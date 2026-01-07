from __future__ import annotations

import datetime
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from pydantic import ConfigDict, Field, PrivateAttr, validate_call

from ..api_types import (
    Configuration,
    Message,
    MessageCreateParam,
    PeerCore,
)
from ..base import PeerBase, SessionBase
from ..http import AsyncHttpClient, AsyncPage
from ..types import DialecticStreamResponse

if TYPE_CHECKING:
    from ..observations import AsyncObservationScope
    from ..types import PeerContext, Representation

from .session import AsyncSession


class AsyncPeer(PeerBase):
    """Represents a peer in the Honcho system with async operations."""

    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: dict[str, object] | None = PrivateAttr(default=None)
    _http: AsyncHttpClient = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        return self._metadata

    @property
    def configuration(self) -> dict[str, object] | None:
        return self._configuration

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        peer_id: str = Field(..., min_length=1),
        workspace_id: str = Field(..., min_length=1),
        http: AsyncHttpClient = Field(...),
        *,
        metadata: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
    ) -> None:
        super().__init__(id=peer_id, workspace_id=workspace_id)
        self._http = http
        self._metadata = metadata
        self._configuration = config

    @classmethod
    async def create(
        cls,
        peer_id: str,
        workspace_id: str,
        http: AsyncHttpClient,
        *,
        config: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "AsyncPeer":
        """Factory method to create and initialize an AsyncPeer with API call."""
        instance = cls(peer_id, workspace_id, http, metadata=metadata, config=config)

        if config is not None or metadata is not None:
            body: dict[str, Any] = {"id": peer_id}
            if config is not None:
                body["configuration"] = config
            if metadata is not None:
                body["metadata"] = metadata

            response = await http.request(
                "POST",
                f"/v2/workspaces/{workspace_id}/peers",
                json=body,
            )
            peer_data = PeerCore.model_validate(response)
            instance._metadata = peer_data.metadata
            instance._configuration = peer_data.configuration

        return instance

    async def chat(
        self,
        query: str,
        *,
        stream: bool = False,
        target: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
    ) -> str | DialecticStreamResponse | None:
        """Query the peer's representation with a natural language question."""
        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )
        resolved_session_id = (
            None
            if session is None
            else (session if isinstance(session, str) else session.id)
        )

        body: dict[str, Any] = {"query": query, "stream": stream}
        if target_id is not None:
            body["target"] = target_id
        if resolved_session_id is not None:
            body["session_id"] = resolved_session_id

        if stream:

            async def stream_response() -> AsyncGenerator[str, None]:
                async for line in self._http.stream(
                    "POST",
                    f"/v2/workspaces/{self.workspace_id}/peers/{self.id}/chat",
                    json=body,
                ):
                    if line.startswith("data: "):
                        json_str = line[6:]
                        try:
                            chunk_data = json.loads(json_str)
                            if chunk_data.get("done"):
                                break
                            delta_obj = chunk_data.get("delta", {})
                            content = delta_obj.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

            return DialecticStreamResponse(stream_response())

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.id}/chat",
            json=body,
        )
        content = response.get("content") if response else None
        if content in ("", None, "None"):
            return None
        return content

    async def get_sessions(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[dict[str, Any], AsyncSession]:
        """Get all sessions this peer is a member of."""

        async def fetch_page(
            page: int = 1, size: int = 50
        ) -> AsyncPage[dict[str, Any], AsyncSession]:
            response = await self._http.request(
                "POST",
                f"/v2/workspaces/{self.workspace_id}/peers/{self.id}/sessions/list",
                json={"filters": filters, "page": page, "size": size},
            )
            return AsyncPage(
                items=response.get("items", []),
                total=response.get("total"),
                page=response.get("page", page),
                size=response.get("size", size),
                pages=response.get("pages"),
                transform_func=lambda s: AsyncSession(
                    s["id"], self.workspace_id, self._http
                ),
                fetch_next=lambda: fetch_page(page + 1, size),
            )

        return await fetch_page()

    @validate_call
    def message(
        self,
        content: str = Field(..., min_length=1),
        *,
        metadata: dict[str, object] | None = None,
        config: Configuration | dict[str, Any] | None = None,
        created_at: datetime.datetime | str | None = None,
    ) -> MessageCreateParam:
        """Create a MessageCreateParam object attributed to this peer."""
        created_at_str: str | None
        if isinstance(created_at, datetime.datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = created_at

        config_dict: dict[str, Any] | None = None
        if config is not None:
            if isinstance(config, Configuration):
                config_dict = config.model_dump()
            else:
                config_dict = config

        return MessageCreateParam(
            peer_id=self.id,
            content=content,
            configuration=config_dict,
            metadata=metadata,
            created_at=created_at_str,
        )

    async def get_metadata(self) -> dict[str, object]:
        """Get the current metadata for this peer."""
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers",
            json={"id": self.id},
        )
        peer = PeerCore.model_validate(response)
        self._metadata = peer.metadata or {}
        return self._metadata

    @validate_call
    async def set_metadata(self, metadata: dict[str, object] = Field(...)) -> None:
        """Set the metadata for this peer."""
        await self._http.request(
            "PUT",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.id}",
            json={"metadata": metadata},
        )
        self._metadata = metadata

    async def get_config(self) -> dict[str, object]:
        """Get the current configuration for this peer."""
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers",
            json={"id": self.id},
        )
        peer = PeerCore.model_validate(response)
        self._configuration = peer.configuration or {}
        return self._configuration

    async def get_peer_config(self) -> dict[str, object]:
        """
        Get the current workspace-level configuration for this peer.

        .. deprecated::
            Use :meth:`get_config` instead.
        """
        return await self.get_config()

    @validate_call
    async def set_config(self, config: dict[str, object] = Field(...)) -> None:
        """Set the configuration for this peer."""
        await self._http.request(
            "PUT",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.id}",
            json={"configuration": config},
        )
        self._configuration = config

    @validate_call
    async def set_peer_config(self, config: dict[str, object] = Field(...)) -> None:
        """
        Set the configuration for this peer.

        .. deprecated::
            Use :meth:`set_config` instead.
        """
        await self.set_config(config)

    async def refresh(self) -> None:
        """Refresh cached metadata and configuration."""
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers",
            json={"id": self.id},
        )
        peer = PeerCore.model_validate(response)
        self._metadata = peer.metadata or {}
        self._configuration = peer.configuration or {}

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1),
        filters: dict[str, object] | None = None,
        limit: int = Field(default=10, ge=1, le=100),
    ) -> list[Message]:
        """Search across all messages with this peer as author."""
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.id}/search",
            json={"query": query, "filters": filters, "limit": limit},
        )
        messages_raw = cast(list[Any], response or [])
        return [Message.model_validate(m) for m in messages_raw]

    async def card(self, target: str | PeerBase | None = None) -> str:
        """Get the peer card for this peer."""
        if isinstance(target, str) and len(target.strip()) == 0:
            raise ValueError("target string cannot be empty")

        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )

        params: dict[str, Any] = {}
        if target_id is not None:
            params["target"] = target_id

        response = await self._http.request(
            "GET",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.id}/card",
            params=params,
        )
        peer_card = response.get("peer_card") if response else None
        if peer_card is None:
            return ""
        return "\n".join(peer_card)

    async def working_rep(
        self,
        session: str | SessionBase | None = None,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_derived: bool | None = None,
        max_observations: int | None = None,
    ) -> "Representation":
        """Get a working representation for this peer."""
        from ..types import Representation as _Representation

        session_id = (
            None
            if session is None
            else session
            if isinstance(session, str)
            else session.id
        )
        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )

        body: dict[str, Any] = {}
        if session_id is not None:
            body["session_id"] = session_id
        if target_id is not None:
            body["target"] = target_id
        if search_query is not None:
            body["search_query"] = search_query
        if search_top_k is not None:
            body["search_top_k"] = search_top_k
        if search_max_distance is not None:
            body["search_max_distance"] = search_max_distance
        if include_most_derived is not None:
            body["include_most_derived"] = include_most_derived
        if max_observations is not None:
            body["max_observations"] = max_observations

        data = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.id}/representation",
            json=body,
        )
        representation = data.get("representation") if data else None
        if representation is not None:
            return _Representation.from_dict(cast(dict[str, object], representation))
        else:
            return _Representation.from_dict(data or {})

    async def get_context(
        self,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_derived: bool | None = None,
        max_observations: int | None = None,
    ) -> "PeerContext":
        """Get context for this peer."""
        from ..types import PeerContext as _PeerContext

        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )

        body: dict[str, Any] = {}
        if target_id is not None:
            body["target"] = target_id
        if search_query is not None:
            body["search_query"] = search_query
        if search_top_k is not None:
            body["search_top_k"] = search_top_k
        if search_max_distance is not None:
            body["search_max_distance"] = search_max_distance
        if include_most_derived is not None:
            body["include_most_derived"] = include_most_derived
        if max_observations is not None:
            body["max_observations"] = max_observations

        response = await self._http.request(
            "GET",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.id}/context",
            params=body,
        )
        return _PeerContext.from_api_response(response or {})

    @property
    def observations(self) -> "AsyncObservationScope":
        """Access this peer's self-observations."""
        from ..observations import AsyncObservationScope as _AsyncObservationScope

        return _AsyncObservationScope(self._http, self.workspace_id, self.id, self.id)

    def observations_of(self, target: str | PeerBase) -> "AsyncObservationScope":
        """Access observations this peer has made about another peer."""
        from ..observations import AsyncObservationScope as _AsyncObservationScope

        target_id = target.id if isinstance(target, PeerBase) else target
        return _AsyncObservationScope(self._http, self.workspace_id, self.id, target_id)

    def __repr__(self) -> str:
        return f"AsyncPeer(id='{self.id}')"

    def __str__(self) -> str:
        return self.id
