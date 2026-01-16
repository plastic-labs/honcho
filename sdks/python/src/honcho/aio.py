# pyright: reportPrivateUsage=false
"""Async view classes for Honcho SDK.

This module provides async accessor classes that wrap the main SDK classes
and provide async versions of all operations. Access via the `.aio` property
on Honcho, Peer, Session, and ConclusionScope instances.

Example:
    ```python
    from honcho import Honcho

    honcho = Honcho(workspace_id="my-workspace")

    # Async operations
    peer = await honcho.aio.peer("user-123")
    await peer.aio.chat("query")
    async for p in honcho.aio.peers():
        print(p.id)
    ```
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict, Field, validate_call

from .api_types import (
    ConclusionResponse,
    MessageCreateParams,
    MessageResponse,
    PeerCardResponse,
    PeerContextResponse,
    PeerResponse,
    QueueStatusResponse,
    RepresentationResponse,
    SessionPeerConfig,
    SessionResponse,
    WorkspaceResponse,
)
from .base import PeerBase, SessionBase
from .http import routes
from .mixins import AsyncMetadataConfigMixin
from .pagination import AsyncPage
from .session_context import SessionContext, SessionSummaries, Summary
from .types import AsyncDialecticStreamResponse
from .utils import (
    datetime_to_iso,
    normalize_peers_to_dict,
    parse_sse_chunk,
    poll_until_complete_async,
    prepare_file_for_upload,
    resolve_id,
)

if TYPE_CHECKING:
    from .client import Honcho
    from .conclusions import ConclusionCreateParams, ConclusionScope

from .conclusions import ConclusionCreateParams
from .peer import Peer
from .session import Session

logger = logging.getLogger(__name__)

__all__ = [
    "HonchoAio",
    "PeerAio",
    "SessionAio",
    "ConclusionScopeAio",
]


class HonchoAio(AsyncMetadataConfigMixin):
    """
    Async view of the Honcho client.

    Access via `honcho.aio`. Provides async versions of all Honcho methods.
    Shares state with the parent Honcho instance.
    """

    __slots__: ClassVar[tuple[str, ...]] = ("_honcho",)
    _honcho: "Honcho"

    def __init__(self, honcho: "Honcho") -> None:
        self._honcho = honcho

    # AsyncMetadataConfigMixin implementation
    def _get_async_http_client(self):
        return self._honcho._async_http_client

    def _get_fetch_route(self) -> str:
        return routes.workspaces()

    def _get_update_route(self) -> str:
        return routes.workspace(self._honcho.workspace_id)

    def _get_fetch_body(self) -> dict[str, Any]:
        return {"id": self._honcho.workspace_id}

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, object], dict[str, object]]:
        workspace = WorkspaceResponse.model_validate(data)
        return workspace.metadata or {}, workspace.configuration or {}

    def _set_metadata(self, metadata: dict[str, object]) -> None:
        self._honcho._metadata = metadata

    def _set_configuration(self, configuration: dict[str, object]) -> None:
        self._honcho._configuration = configuration

    def _get_metadata(self) -> dict[str, object]:
        return self._honcho._metadata or {}

    def _get_configuration(self) -> dict[str, object]:
        return self._honcho._configuration or {}

    async def peer(
        self,
        id: str,
        *,
        metadata: dict[str, object] | None = None,
        configuration: dict[str, object] | None = None,
    ) -> Peer:
        """
        Get or create a peer with the given ID asynchronously.

        Args:
            id: Unique identifier for the peer within the workspace.
            metadata: Optional metadata dictionary to associate with this peer.
            configuration: Optional configuration to set for this peer.

        Returns:
            A Peer object
        """
        if configuration is not None or metadata is not None:
            body: dict[str, Any] = {"id": id}
            if metadata is not None:
                body["metadata"] = metadata
            if configuration is not None:
                body["configuration"] = configuration

            data = await self._honcho._async_http_client.post(
                routes.peers(self._honcho.workspace_id), body=body
            )
            peer_data = PeerResponse.model_validate(data)
            return Peer(
                id,
                self._honcho,
                metadata=peer_data.metadata,
                configuration=peer_data.configuration,
            )

        return Peer(id, self._honcho, metadata=metadata, configuration=configuration)

    async def peers(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[PeerResponse, Peer]:
        """
        Get all peers in the current workspace asynchronously.

        Returns:
            An AsyncPage of Peer objects
        """
        data = await self._honcho._async_http_client.post(
            routes.peers_list(self._honcho.workspace_id),
            body={"filters": filters} if filters else None,
        )

        def transform(peer: PeerResponse) -> Peer:
            return Peer(
                peer.id,
                self._honcho,
                metadata=peer.metadata,
                configuration=peer.configuration,
            )

        async def fetch_next(page: int) -> AsyncPage[PeerResponse, Peer]:
            next_data = await self._honcho._async_http_client.post(
                routes.peers_list(self._honcho.workspace_id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return AsyncPage(next_data, PeerResponse, transform, fetch_next)

        return AsyncPage(data, PeerResponse, transform, fetch_next)

    async def session(
        self,
        id: str,
        *,
        metadata: dict[str, object] | None = None,
        configuration: dict[str, object] | None = None,
    ) -> Session:
        """
        Get or create a session with the given ID asynchronously.

        Args:
            id: Unique identifier for the session within the workspace.
            metadata: Optional metadata dictionary to associate with this session.
            configuration: Optional configuration to set for this session.

        Returns:
            A Session object
        """
        if configuration is not None or metadata is not None:
            body: dict[str, Any] = {"id": id}
            if metadata is not None:
                body["metadata"] = metadata
            if configuration is not None:
                body["configuration"] = configuration

            data = await self._honcho._async_http_client.post(
                routes.sessions(self._honcho.workspace_id), body=body
            )
            session_data = SessionResponse.model_validate(data)
            return Session(
                id,
                self._honcho,
                metadata=session_data.metadata,
                configuration=session_data.configuration,
            )

        return Session(id, self._honcho, metadata=metadata, configuration=configuration)

    async def sessions(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[SessionResponse, Session]:
        """
        Get all sessions in the current workspace asynchronously.

        Returns:
            An AsyncPage of Session objects
        """
        data = await self._honcho._async_http_client.post(
            routes.sessions_list(self._honcho.workspace_id),
            body={"filters": filters} if filters else None,
        )

        def transform(session: SessionResponse) -> Session:
            return Session(
                session.id,
                self._honcho,
                metadata=session.metadata,
                configuration=session.configuration,
            )

        async def fetch_next(page: int) -> AsyncPage[SessionResponse, Session]:
            next_data = await self._honcho._async_http_client.post(
                routes.sessions_list(self._honcho.workspace_id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return AsyncPage(next_data, SessionResponse, transform, fetch_next)

        return AsyncPage(data, SessionResponse, transform, fetch_next)

    async def workspaces(self, filters: dict[str, object] | None = None) -> list[str]:
        """Get all workspace IDs asynchronously."""
        data = await self._honcho._async_http_client.post(
            routes.workspaces_list(),
            body={"filters": filters} if filters else None,
        )
        workspace_ids: list[str] = []
        for item in data.get("items", []):
            workspace = WorkspaceResponse.model_validate(item)
            workspace_ids.append(workspace.id)
        return workspace_ids

    async def delete_workspace(self, workspace_id: str) -> None:
        """Delete a workspace asynchronously."""
        await self._honcho._async_http_client.delete(routes.workspace(workspace_id))

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1, description="The search query to use"),
        filters: dict[str, object] | None = Field(
            None, description="Filters to scope the search"
        ),
        limit: int = Field(
            default=10, ge=1, le=100, description="Number of results to return"
        ),
    ) -> list[MessageResponse]:
        """Search for messages in the current workspace asynchronously."""
        data = await self._honcho._async_http_client.post(
            routes.workspace_search(self._honcho.workspace_id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [MessageResponse.model_validate(item) for item in data]

    async def queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
    ) -> QueueStatusResponse:
        """Get queue processing status asynchronously."""
        resolved_observer_id = resolve_id(observer)
        resolved_sender_id = resolve_id(sender)
        resolved_session_id = resolve_id(session)

        query: dict[str, Any] = {}
        if resolved_observer_id:
            query["observer_id"] = resolved_observer_id
        if resolved_sender_id:
            query["sender_id"] = resolved_sender_id
        if resolved_session_id:
            query["session_id"] = resolved_session_id

        data = await self._honcho._async_http_client.get(
            routes.workspace_queue_status(self._honcho.workspace_id),
            query=query if query else None,
        )
        return QueueStatusResponse.model_validate(data)

    async def poll_queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
        timeout: float = 300.0,
    ) -> QueueStatusResponse:
        """Poll queue status until complete asynchronously."""
        return await poll_until_complete_async(
            lambda: self.queue_status(observer, sender, session),
            timeout=timeout,
        )

    async def list_conclusions(
        self,
        filters: dict[str, object] | None = None,
        reverse: bool = False,
    ) -> AsyncPage[ConclusionResponse, ConclusionResponse]:
        """List all conclusions in the current workspace asynchronously."""
        data = await self._honcho._async_http_client.post(
            routes.conclusions_list(self._honcho.workspace_id),
            body={"filters": filters, "reverse": reverse},
        )
        return AsyncPage(data, ConclusionResponse)

    @validate_call
    async def query_conclusions(
        self,
        query: str = Field(..., min_length=1, description="Semantic search query"),
        observer: str = Field(
            ..., min_length=1, description="Observer peer ID (required)"
        ),
        observed: str = Field(
            ..., min_length=1, description="Observed peer ID (required)"
        ),
        top_k: int = Field(
            default=10, ge=1, le=100, description="Number of results to return"
        ),
        distance: float | None = Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Maximum cosine distance threshold for results",
        ),
        filters: dict[str, object] | None = Field(
            None, description="Additional filters to apply"
        ),
    ) -> list[ConclusionResponse]:
        """Query conclusions using semantic search asynchronously."""
        query_filters: dict[str, object | str] = {
            **(filters or {}),
            "observer": observer,
            "observed": observed,
        }

        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "filters": query_filters,
        }
        if distance is not None:
            body["distance"] = distance

        data = await self._honcho._async_http_client.post(
            routes.conclusions_query(self._honcho.workspace_id),
            body=body,
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    async def delete_conclusion(self, conclusion_id: str) -> None:
        """Delete a specific conclusion by ID asynchronously."""
        await self._honcho._async_http_client.delete(
            routes.conclusion(self._honcho.workspace_id, conclusion_id)
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def update_message(
        self,
        message: MessageResponse | str = Field(
            ..., description="The Message object or message ID to update"
        ),
        metadata: dict[str, object] = Field(
            ..., description="The metadata to update for the message"
        ),
        session: str | SessionBase | None = Field(
            None,
            description="The session (ID string or Session object) - required if message is a string ID",
        ),
    ) -> MessageResponse:
        """Update message metadata asynchronously."""
        if isinstance(message, MessageResponse):
            message_id = message.id
            resolved_session_id = message.session_id
        else:
            message_id = message
            if not session:
                raise ValueError("session is required when message is a string ID")
            resolved_session_id = resolve_id(session)

        data = await self._honcho._async_http_client.put(
            routes.message(self._honcho.workspace_id, resolved_session_id, message_id),
            body={"metadata": metadata},
        )
        return MessageResponse.model_validate(data)


class PeerAio(AsyncMetadataConfigMixin):
    """
    Async view of a Peer.

    Access via `peer.aio`. Provides async versions of all Peer methods.
    Shares state with the parent Peer instance.
    """

    __slots__: ClassVar[tuple[str, ...]] = ("_peer",)
    _peer: "Peer"

    def __init__(self, peer: "Peer") -> None:
        self._peer = peer

    # AsyncMetadataConfigMixin implementation
    def _get_async_http_client(self):
        return self._peer._honcho._async_http_client

    def _get_fetch_route(self) -> str:
        return routes.peers(self._peer.workspace_id)

    def _get_update_route(self) -> str:
        return routes.peer(self._peer.workspace_id, self._peer.id)

    def _get_fetch_body(self) -> dict[str, Any]:
        return {"id": self._peer.id}

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, object], dict[str, object]]:
        peer = PeerResponse.model_validate(data)
        return peer.metadata or {}, peer.configuration or {}

    def _set_metadata(self, metadata: dict[str, object]) -> None:
        self._peer._metadata = metadata

    def _set_configuration(self, configuration: dict[str, object]) -> None:
        self._peer._configuration = configuration

    def _get_metadata(self) -> dict[str, object]:
        return self._peer._metadata or {}

    def _get_configuration(self) -> dict[str, object]:
        return self._peer._configuration or {}

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def chat(
        self,
        query: str = Field(..., min_length=1, description="The natural language query"),
        *,
        target: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
        reasoning_level: Literal["minimal", "low", "medium", "high", "max"]
        | None = None,
    ) -> str | None:
        """Query the peer's representation asynchronously."""
        target_id = resolve_id(target)
        resolved_session_id = resolve_id(session)

        body: dict[str, Any] = {"query": query, "stream": False}
        if target_id:
            body["target"] = target_id
        if resolved_session_id:
            body["session_id"] = resolved_session_id
        if reasoning_level:
            body["reasoning_level"] = reasoning_level

        data = await self._peer._honcho._async_http_client.post(
            routes.peer_chat(self._peer.workspace_id, self._peer.id),
            body=body,
        )
        content = data.get("content")
        if not content:
            return None
        return content

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def chat_stream(
        self,
        query: str = Field(..., min_length=1, description="The natural language query"),
        *,
        target: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
        reasoning_level: Literal["minimal", "low", "medium", "high", "max"]
        | None = None,
    ) -> AsyncDialecticStreamResponse:
        """Query the peer's representation with streaming asynchronously."""
        target_id = resolve_id(target)
        resolved_session_id = resolve_id(session)

        body: dict[str, Any] = {"query": query, "stream": True}
        if target_id:
            body["target"] = target_id
        if resolved_session_id:
            body["session_id"] = resolved_session_id
        if reasoning_level:
            body["reasoning_level"] = reasoning_level

        async def stream_response() -> AsyncGenerator[str, None]:
            async for chunk in self._peer._honcho._async_http_client.stream(
                "POST",
                routes.peer_chat(self._peer.workspace_id, self._peer.id),
                body=body,
            ):
                for content in parse_sse_chunk(chunk):
                    yield content

        return AsyncDialecticStreamResponse(stream_response())

    async def sessions(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[SessionResponse, Session]:
        """Get all sessions this peer is a member of asynchronously."""
        data = await self._peer._honcho._async_http_client.post(
            routes.peer_sessions_list(self._peer.workspace_id, self._peer.id),
            body={"filters": filters} if filters else None,
        )

        def transform(session: SessionResponse) -> Session:
            return Session(session.id, self._peer._honcho)

        async def fetch_next(page: int) -> AsyncPage[SessionResponse, Session]:
            next_data = await self._peer._honcho._async_http_client.post(
                routes.peer_sessions_list(self._peer.workspace_id, self._peer.id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return AsyncPage(next_data, SessionResponse, transform, fetch_next)

        return AsyncPage(data, SessionResponse, transform, fetch_next)

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1, description="The search query to use"),
        filters: dict[str, object] | None = Field(
            None, description="Filters to scope the search"
        ),
        limit: int = Field(
            default=10, ge=1, le=100, description="Number of results to return"
        ),
    ) -> list[MessageResponse]:
        """Search across all messages with this peer as author asynchronously."""
        data = await self._peer._honcho._async_http_client.post(
            routes.peer_search(self._peer.workspace_id, self._peer.id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [MessageResponse.model_validate(item) for item in data]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def card(
        self,
        target: str | PeerBase | None = None,
    ) -> list[str] | None:
        """Get the peer card asynchronously."""
        target_id = resolve_id(target)

        query = {"target": target_id} if target_id else None
        data = await self._peer._honcho._async_http_client.get(
            routes.peer_card(self._peer.workspace_id, self._peer.id),
            query=query,
        )
        response = PeerCardResponse.model_validate(data)
        return response.peer_card

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def representation(
        self,
        session: str | SessionBase | None = None,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = Field(None, ge=1, le=100),
        search_max_distance: float | None = Field(None, ge=0.0, le=1.0),
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = Field(None, ge=1, le=100),
    ) -> str:
        """Get a subset of the representation of the peer asynchronously."""
        session_id = resolve_id(session)
        target_id = resolve_id(target)

        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if target_id:
            body["target"] = target_id
        if search_query is not None:
            body["search_query"] = search_query
        if search_top_k is not None:
            body["search_top_k"] = search_top_k
        if search_max_distance is not None:
            body["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            body["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            body["max_conclusions"] = max_conclusions

        data = await self._peer._honcho._async_http_client.post(
            routes.peer_representation(self._peer.workspace_id, self._peer.id),
            body=body,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def context(
        self,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = Field(None, ge=1, le=100),
        search_max_distance: float | None = Field(None, ge=0.0, le=1.0),
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = Field(None, ge=1, le=100),
    ) -> PeerContextResponse:
        """Get context for this peer asynchronously."""
        target_id = resolve_id(target)

        query: dict[str, Any] = {}
        if target_id:
            query["target"] = target_id
        if search_query is not None:
            query["search_query"] = search_query
        if search_top_k is not None:
            query["search_top_k"] = search_top_k
        if search_max_distance is not None:
            query["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            query["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            query["max_conclusions"] = max_conclusions

        data = await self._peer._honcho._async_http_client.get(
            routes.peer_context(self._peer.workspace_id, self._peer.id),
            query=query if query else None,
        )
        return PeerContextResponse.model_validate(data)


class SessionAio(AsyncMetadataConfigMixin):
    """
    Async view of a Session.

    Access via `session.aio`. Provides async versions of all Session methods.
    Shares state with the parent Session instance.
    """

    __slots__: ClassVar[tuple[str, ...]] = ("_session",)
    _session: "Session"

    def __init__(self, session: "Session") -> None:
        self._session = session

    # AsyncMetadataConfigMixin implementation
    def _get_async_http_client(self):
        return self._session._honcho._async_http_client

    def _get_fetch_route(self) -> str:
        return routes.sessions(self._session.workspace_id)

    def _get_update_route(self) -> str:
        return routes.session(self._session.workspace_id, self._session.id)

    def _get_fetch_body(self) -> dict[str, Any]:
        return {"id": self._session.id}

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, object], dict[str, object]]:
        session = SessionResponse.model_validate(data)
        return session.metadata or {}, session.configuration or {}

    def _set_metadata(self, metadata: dict[str, object]) -> None:
        self._session._metadata = metadata

    def _set_configuration(self, configuration: dict[str, object]) -> None:
        self._session._configuration = configuration

    def _get_metadata(self) -> dict[str, object]:
        return self._session._metadata or {}

    def _get_configuration(self) -> dict[str, object]:
        return self._session._configuration or {}

    async def add_peers(
        self,
        peers: str
        | PeerBase
        | tuple[str, SessionPeerConfig]
        | tuple[PeerBase, SessionPeerConfig]
        | list[PeerBase | str]
        | list[tuple[PeerBase | str, SessionPeerConfig]]
        | list[PeerBase | str | tuple[PeerBase | str, SessionPeerConfig]],
    ) -> None:
        """Add peers to this session asynchronously."""
        await self._session._honcho._async_http_client.post(
            routes.session_peers_add(self._session.workspace_id, self._session.id),
            body=normalize_peers_to_dict(peers),
        )

    async def set_peers(
        self,
        peers: str
        | PeerBase
        | tuple[str, SessionPeerConfig]
        | tuple[PeerBase, SessionPeerConfig]
        | list[PeerBase | str]
        | list[tuple[PeerBase | str, SessionPeerConfig]]
        | list[PeerBase | str | tuple[PeerBase | str, SessionPeerConfig]],
    ) -> None:
        """Set the complete peer list for this session asynchronously."""
        await self._session._honcho._async_http_client.put(
            routes.session_peers_set(self._session.workspace_id, self._session.id),
            body=normalize_peers_to_dict(peers),
        )

    async def remove_peers(
        self,
        peers: str | PeerBase | list[PeerBase | str],
    ) -> None:
        """Remove peers from this session asynchronously."""
        if not isinstance(peers, list):
            peers = [peers]

        peer_ids = [peer if isinstance(peer, str) else peer.id for peer in peers]

        await self._session._honcho._async_http_client.delete(
            routes.session_peers_remove(self._session.workspace_id, self._session.id),
            body=peer_ids,
        )

    async def peers(self) -> list[Peer]:
        """Get all peers in this session asynchronously."""
        data: dict[str, Any] = await self._session._honcho._async_http_client.get(
            routes.session_peers(self._session.workspace_id, self._session.id)
        )

        peers_data: list[Any] = data.get("items", [])
        return [
            Peer(PeerResponse.model_validate(peer).id, self._session._honcho)
            for peer in peers_data
        ]

    async def peer_config(self, peer: str | PeerBase) -> SessionPeerConfig:
        """Get the configuration for a peer in this session asynchronously."""
        peer_id = peer if isinstance(peer, str) else peer.id
        data = await self._session._honcho._async_http_client.get(
            routes.session_peer_config(
                self._session.workspace_id, self._session.id, peer_id
            )
        )
        return SessionPeerConfig(
            observe_others=data.get("observe_others"),
            observe_me=data.get("observe_me"),
        )

    async def set_peer_config(
        self, peer: str | PeerBase, config: SessionPeerConfig
    ) -> None:
        """Set the configuration for a peer in this session asynchronously."""
        peer_id = peer if isinstance(peer, str) else peer.id
        body: dict[str, Any] = {}
        if config.observe_others is not None:
            body["observe_others"] = config.observe_others
        if config.observe_me is not None:
            body["observe_me"] = config.observe_me

        await self._session._honcho._async_http_client.put(
            routes.session_peer_config(
                self._session.workspace_id, self._session.id, peer_id
            ),
            body=body,
        )

    @validate_call
    async def add_messages(
        self,
        messages: MessageCreateParams | list[MessageCreateParams] = Field(
            ..., description="Messages to add to the session"
        ),
    ) -> list[MessageResponse]:
        """Add one or more messages to this session asynchronously."""
        if not isinstance(messages, list):
            messages = [messages]

        messages_data = [
            msg.model_dump(mode="json", exclude_none=True) for msg in messages
        ]

        data = await self._session._honcho._async_http_client.post(
            routes.messages(self._session.workspace_id, self._session.id),
            body={"messages": messages_data},
        )
        return [MessageResponse.model_validate(msg) for msg in data]

    async def messages(
        self,
        *,
        filters: dict[str, object] | None = None,
    ) -> AsyncPage[MessageResponse, MessageResponse]:
        """Get messages from this session asynchronously."""
        data = await self._session._honcho._async_http_client.post(
            routes.messages_list(self._session.workspace_id, self._session.id),
            body={"filters": filters} if filters else None,
        )

        async def fetch_next(page: int) -> AsyncPage[MessageResponse, MessageResponse]:
            next_data = await self._session._honcho._async_http_client.post(
                routes.messages_list(self._session.workspace_id, self._session.id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return AsyncPage(next_data, MessageResponse, None, fetch_next)

        return AsyncPage(data, MessageResponse, None, fetch_next)

    async def delete(self) -> None:
        """Delete this session asynchronously."""
        await self._session._honcho._async_http_client.delete(
            routes.session(self._session.workspace_id, self._session.id)
        )

    async def clone(self, *, message_id: str | None = None) -> Session:
        """Clone this session asynchronously."""
        query: dict[str, Any] = {}
        if message_id is not None:
            query["message_id"] = message_id

        data = await self._session._honcho._async_http_client.post(
            routes.session_clone(self._session.workspace_id, self._session.id),
            query=query if query else None,
        )
        cloned = SessionResponse.model_validate(data)
        return Session(
            cloned.id,
            self._session._honcho,
            metadata=cloned.metadata,
            configuration=cloned.configuration,
        )

    @validate_call
    async def context(
        self,
        *,
        summary: bool = True,
        tokens: int | None = Field(
            None, gt=0, description="Maximum number of tokens to include in the context"
        ),
        peer_target: str | None = Field(
            None,
            description="A peer ID to get context for.",
        ),
        last_user_message: str | MessageResponse | None = Field(
            None,
            description="The most recent message (string or Message object), used to fetch semantically relevant conclusions.",
        ),
        peer_perspective: str | None = Field(
            None,
            description="A peer ID to get context from the perspective of.",
        ),
        limit_to_session: bool = Field(
            False,
            description="Whether to limit the representation to this session only.",
        ),
        search_top_k: int | None = Field(
            None,
            ge=1,
            le=100,
            description="Number of semantically relevant facts to return.",
        ),
        search_max_distance: float | None = Field(
            None,
            ge=0.0,
            le=1.0,
            description="Maximum semantic distance for search results (0.0-1.0).",
        ),
        include_most_frequent: bool | None = Field(
            None,
            description="Whether to include the most frequent conclusions in the representation.",
        ),
        max_conclusions: int | None = Field(
            None,
            ge=1,
            le=100,
            description="Maximum number of conclusions to include in the representation.",
        ),
    ) -> SessionContext:
        """Get optimized context for this session asynchronously."""
        if peer_target is None and peer_perspective is not None:
            raise ValueError(
                "You must provide a `peer_target` when `peer_perspective` is provided"
            )

        if peer_target is None and last_user_message is not None:
            raise ValueError(
                "You must provide a `peer_target` when `last_user_message` is provided"
            )

        last_user_message_id = (
            last_user_message.id
            if isinstance(last_user_message, MessageResponse)
            else last_user_message
        )

        query: dict[str, Any] = {
            "summary": summary,
            "limit_to_session": limit_to_session,
        }
        if tokens is not None:
            query["tokens"] = tokens
        if last_user_message_id is not None:
            query["last_message"] = last_user_message_id
        if peer_target is not None:
            query["peer_target"] = peer_target
        if peer_perspective is not None:
            query["peer_perspective"] = peer_perspective
        if search_top_k is not None:
            query["search_top_k"] = search_top_k
        if search_max_distance is not None:
            query["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            query["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            query["max_conclusions"] = max_conclusions

        data = await self._session._honcho._async_http_client.get(
            routes.session_context(self._session.workspace_id, self._session.id),
            query=query,
        )

        session_summary = None
        if data.get("summary"):
            s = data["summary"]
            session_summary = Summary(
                content=s["content"],
                message_id=s["message_id"],
                summary_type=s["summary_type"],
                created_at=s["created_at"],
                token_count=s["token_count"],
            )

        messages = [
            MessageResponse.model_validate(msg) for msg in data.get("messages", [])
        ]

        return SessionContext(
            session_id=self._session.id,
            messages=messages,
            summary=session_summary,
            peer_representation=str(data.get("peer_representation"))
            if data.get("peer_representation")
            else None,
            peer_card=data.get("peer_card"),
        )

    async def summaries(self) -> SessionSummaries:
        """Get available summaries for this session asynchronously."""
        data = await self._session._honcho._async_http_client.get(
            routes.session_summaries(self._session.workspace_id, self._session.id)
        )

        short_summary = None
        if data.get("short_summary"):
            s = data["short_summary"]
            short_summary = Summary(
                content=s["content"],
                message_id=s["message_id"],
                summary_type=s["summary_type"],
                created_at=s["created_at"],
                token_count=s["token_count"],
            )

        long_summary = None
        if data.get("long_summary"):
            s = data["long_summary"]
            long_summary = Summary(
                content=s["content"],
                message_id=s["message_id"],
                summary_type=s["summary_type"],
                created_at=s["created_at"],
                token_count=s["token_count"],
            )

        return SessionSummaries(
            id=data.get("id") or self._session.id,
            short_summary=short_summary,
            long_summary=long_summary,
        )

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1, description="The search query to use"),
        filters: dict[str, object] | None = Field(
            None, description="Filters to scope the search"
        ),
        limit: int = Field(
            default=10, ge=1, le=100, description="Number of results to return"
        ),
    ) -> list[MessageResponse]:
        """Search for messages in this session asynchronously."""
        data = await self._session._honcho._async_http_client.post(
            routes.session_search(self._session.workspace_id, self._session.id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [MessageResponse.model_validate(msg) for msg in data]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def upload_file(
        self,
        file: tuple[str, bytes, str] | tuple[str, Any, str] | Any = Field(
            ...,
            description="File to upload. Can be a file object, (filename, bytes, content_type) tuple, or (filename, fileobj, content_type) tuple.",
        ),
        peer: str | PeerBase = Field(
            ..., description="The peer creating the messages (ID string or Peer object)"
        ),
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with the messages",
        ),
        configuration: dict[str, Any] | None = Field(
            None,
            description="Optional configuration dictionary to associate with the messages",
        ),
        created_at: str | datetime | None = Field(
            None,
            description="Optional created-at timestamp for the messages.",
        ),
    ) -> list[MessageResponse]:
        """Upload file to create message(s) in this session asynchronously."""
        filename, content_bytes, content_type = prepare_file_for_upload(file)
        resolved_peer_id = peer if isinstance(peer, str) else peer.id

        data_dict: dict[str, str] = {"peer_id": resolved_peer_id}
        if metadata is not None:
            data_dict["metadata"] = json.dumps(metadata)
        if configuration is not None:
            data_dict["configuration"] = json.dumps(configuration)
        created_at_iso = datetime_to_iso(created_at)
        if created_at_iso is not None:
            data_dict["created_at"] = created_at_iso

        response = await self._session._honcho._async_http_client.upload(
            routes.messages_upload(self._session.workspace_id, self._session.id),
            files={"file": (filename, content_bytes, content_type)},
            data=data_dict,
        )

        return [MessageResponse.model_validate(msg) for msg in response]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def representation(
        self,
        peer: str | PeerBase,
        *,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = Field(None, ge=1, le=100),
        search_max_distance: float | None = Field(None, ge=0.0, le=1.0),
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = Field(None, ge=1, le=100),
    ) -> str:
        """Get a subset of the representation of the peer in this session asynchronously."""
        peer_id = resolve_id(peer)
        target_id = resolve_id(target)

        query: dict[str, Any] = {"session_id": self._session.id}
        if target_id:
            query["target"] = target_id
        if search_query is not None:
            query["search_query"] = search_query
        if search_top_k is not None:
            query["search_top_k"] = search_top_k
        if search_max_distance is not None:
            query["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            query["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            query["max_conclusions"] = max_conclusions

        data = await self._session._honcho._async_http_client.post(
            routes.peer_representation(self._session.workspace_id, peer_id),
            body=query,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    async def queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
    ) -> QueueStatusResponse:
        """Get the queue processing status for this session asynchronously."""
        resolved_observer_id = resolve_id(observer)
        resolved_sender_id = resolve_id(sender)

        query: dict[str, Any] = {"session_id": self._session.id}
        if resolved_observer_id:
            query["observer_id"] = resolved_observer_id
        if resolved_sender_id:
            query["sender_id"] = resolved_sender_id

        data = await self._session._honcho._async_http_client.get(
            routes.workspace_queue_status(self._session.workspace_id),
            query=query,
        )
        return QueueStatusResponse.model_validate(data)

    async def poll_queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        timeout: float = 300.0,
    ) -> QueueStatusResponse:
        """Poll queue status until complete asynchronously."""
        return await poll_until_complete_async(
            lambda: self.queue_status(observer, sender),
            timeout=timeout,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def update_message(
        self,
        message: MessageResponse | str = Field(
            ..., description="The Message object or message ID to update"
        ),
        metadata: dict[str, object] = Field(
            ..., description="The metadata to update for the message"
        ),
    ) -> MessageResponse:
        """Update message metadata in this session asynchronously."""
        message_id = message.id if isinstance(message, MessageResponse) else message

        data = await self._session._honcho._async_http_client.put(
            routes.message(self._session.workspace_id, self._session.id, message_id),
            body={"metadata": metadata},
        )
        return MessageResponse.model_validate(data)


class ConclusionScopeAio:
    """
    Async view of a ConclusionScope.

    Access via `scope.aio`. Provides async versions of all ConclusionScope methods.
    Shares state with the parent ConclusionScope instance.
    """

    __slots__: ClassVar[tuple[str, ...]] = ("_scope",)
    _scope: "ConclusionScope"

    def __init__(self, scope: "ConclusionScope") -> None:
        self._scope = scope

    async def list(
        self,
        page: int = 1,
        size: int = 50,
        session: str | SessionBase | None = None,
    ) -> AsyncPage[ConclusionResponse, ConclusionResponse]:
        """List conclusions in this scope asynchronously."""
        resolved_session_id = resolve_id(session)
        filters: dict[str, Any] = {
            "observer": self._scope.observer,
            "observed": self._scope.observed,
        }
        if resolved_session_id:
            filters["session_id"] = resolved_session_id

        data = await self._scope._honcho._async_http_client.post(
            routes.conclusions_list(self._scope.workspace_id),
            body={"filters": filters, "page": page, "size": size},
        )

        async def fetch_next(
            page: int,
        ) -> AsyncPage[ConclusionResponse, ConclusionResponse]:
            next_data = await self._scope._honcho._async_http_client.post(
                routes.conclusions_list(self._scope.workspace_id),
                body={"filters": filters, "page": page, "size": size},
            )
            return AsyncPage(next_data, ConclusionResponse, None, fetch_next)

        return AsyncPage(data, ConclusionResponse, None, fetch_next)

    async def query(
        self,
        query: str,
        top_k: int = 10,
        distance: float | None = None,
    ) -> list[ConclusionResponse]:
        """Semantic search for conclusions asynchronously."""
        filters: dict[str, Any] = {
            "observer": self._scope.observer,
            "observed": self._scope.observed,
        }

        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
        }
        if distance is not None:
            body["distance"] = distance

        data = await self._scope._honcho._async_http_client.post(
            routes.conclusions_query(self._scope.workspace_id),
            body=body,
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    async def delete(self, conclusion_id: str) -> None:
        """Delete a conclusion by ID asynchronously."""
        await self._scope._honcho._async_http_client.delete(
            routes.conclusion(self._scope.workspace_id, conclusion_id)
        )

    async def create(
        self,
        conclusions: list[ConclusionCreateParams | dict[str, Any]],
    ) -> list[ConclusionResponse]:
        """Create conclusions in this scope asynchronously."""
        conclusion_params = [
            {
                "content": c.content
                if isinstance(c, ConclusionCreateParams)
                else c["content"],
                "session_id": c.session_id
                if isinstance(c, ConclusionCreateParams)
                else c["session_id"],
                "observer_id": self._scope.observer,
                "observed_id": self._scope.observed,
            }
            for c in conclusions
        ]

        data = await self._scope._honcho._async_http_client.post(
            routes.conclusions(self._scope.workspace_id),
            body={"conclusions": conclusion_params},
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    async def get_representation(
        self,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> str:
        """Get the computed representation for this scope asynchronously."""
        query: dict[str, Any] = {"target": self._scope.observed}
        if search_query is not None:
            query["search_query"] = search_query
        if search_top_k is not None:
            query["search_top_k"] = search_top_k
        if search_max_distance is not None:
            query["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            query["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            query["max_conclusions"] = max_conclusions

        data = await self._scope._honcho._async_http_client.get(
            routes.peer_representation(self._scope.workspace_id, self._scope.observer),
            query=query,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation
