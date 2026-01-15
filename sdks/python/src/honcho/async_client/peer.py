"""Async Peer class for Honcho SDK."""

from __future__ import annotations

import datetime
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr, validate_call

from ..api_types import (
    MessageCreateParams,
    MessageResponse,
    PeerCardResponse,
    PeerContextResponse,
    PeerResponse,
    RepresentationResponse,
    SessionResponse,
)
from ..base import PeerBase, SessionBase
from ..http import AsyncHonchoHTTPClient, routes
from ..types import DialecticStreamResponse
from .pagination import AsyncPage

if TYPE_CHECKING:
    from .session import AsyncSession


class AsyncPeer(PeerBase):
    """
    Represents a peer in the Honcho system with async operations.

    Peers can send messages, participate in sessions, and maintain both global
    and local representations for contextual interactions.

    Attributes:
        id: Unique identifier for this peer
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this peer.
        configuration: Cached configuration for this peer.
    """

    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: dict[str, object] | None = PrivateAttr(default=None)
    _http: AsyncHonchoHTTPClient = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this peer. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> dict[str, object] | None:
        """Cached configuration for this peer. May be stale. Use get_config() for fresh data."""
        return self._configuration

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        peer_id: str = Field(
            ...,
            min_length=1,
            description="Unique identifier for this peer within the workspace",
        ),
        workspace_id: str = Field(
            ..., min_length=1, description="Workspace ID for scoping operations"
        ),
        http: AsyncHonchoHTTPClient = Field(..., description="HTTP client instance"),
        *,
        metadata: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
    ) -> None:
        """
        Initialize a new AsyncPeer.

        Args:
            peer_id: Unique identifier for this peer within the workspace
            workspace_id: Workspace ID for scoping operations
            http: HTTP client instance
            metadata: Optional metadata to initialize the cached value
            config: Optional configuration to initialize the cached value
        """
        super().__init__(
            id=peer_id,
            workspace_id=workspace_id,
        )
        self._http = http
        self._metadata = metadata
        self._configuration = config

    @classmethod
    async def create(
        cls,
        peer_id: str,
        workspace_id: str,
        http: AsyncHonchoHTTPClient,
        *,
        metadata: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
    ) -> AsyncPeer:
        """
        Create a new AsyncPeer with optional configuration.

        Args:
            peer_id: Unique identifier for this peer
            workspace_id: Workspace ID for scoping operations
            http: HTTP client instance
            metadata: Optional metadata dictionary
            config: Optional configuration

        Returns:
            A new AsyncPeer instance
        """
        if config is not None or metadata is not None:
            body: dict[str, Any] = {"id": peer_id}
            if metadata is not None:
                body["metadata"] = metadata
            if config is not None:
                body["configuration"] = config

            data = await http.post(routes.peers(workspace_id), body=body)
            peer_data = PeerResponse.model_validate(data)
            return cls(
                peer_id,
                workspace_id,
                http,
                metadata=peer_data.metadata,
                config=peer_data.configuration,
            )

        return cls(peer_id, workspace_id, http)

    async def chat(
        self,
        query: str,
        *,
        stream: bool = False,
        target: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
        reasoning_level: Literal["minimal", "low", "medium", "high", "max"]
        | None = None,
    ) -> str | DialecticStreamResponse | None:
        """
        Query the peer's representation with a natural language question.

        Args:
            query: The natural language question to ask.
            stream: Whether to stream the response
            target: Optional target peer for local representation query.
            session: Optional session to scope the query to.
            reasoning_level: Optional reasoning level for the query.

        Returns:
            For non-streaming: Response string, or None if no relevant information
            For streaming: DialecticStreamResponse object
        """
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
        if target_id:
            body["target"] = target_id
        if resolved_session_id:
            body["session_id"] = resolved_session_id
        if reasoning_level:
            body["reasoning_level"] = reasoning_level

        if stream:

            async def stream_response() -> AsyncGenerator[str, None]:
                async for chunk in self._http.stream(
                    "POST",
                    routes.peer_chat(self.workspace_id, self.id),
                    body=body,
                ):
                    # Parse SSE data
                    for line in chunk.decode("utf-8").split("\n"):
                        if line.startswith("data: "):
                            json_str = line[6:]
                            try:
                                chunk_data = json.loads(json_str)
                                if chunk_data.get("done"):
                                    return
                                delta_obj = chunk_data.get("delta", {})
                                content = delta_obj.get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

            return DialecticStreamResponse(stream_response())

        data = await self._http.post(
            routes.peer_chat(self.workspace_id, self.id),
            body=body,
        )
        content = data.get("content")
        if content in ("", None, "None"):
            return None
        return content

    async def get_sessions(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[SessionResponse, "AsyncSession"]:
        """
        Get all sessions this peer is a member of.

        Returns:
            An async paginated list of AsyncSession objects
        """
        from .session import AsyncSession

        data = await self._http.post(
            routes.peer_sessions_list(self.workspace_id, self.id),
            body={"filters": filters} if filters else None,
        )

        def transform(session: SessionResponse) -> AsyncSession:
            return AsyncSession(session.id, self.workspace_id, self._http)

        async def fetch_next(page: int) -> AsyncPage[SessionResponse, AsyncSession]:
            next_data = await self._http.post(
                routes.peer_sessions_list(self.workspace_id, self.id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return AsyncPage(next_data, SessionResponse, transform, fetch_next)

        return AsyncPage(data, SessionResponse, transform, fetch_next)

    @validate_call
    def message(
        self,
        content: str = Field(
            ..., min_length=1, description="The text content for the message"
        ),
        *,
        config: dict[str, Any] | None = Field(
            None,
            description="Optional configuration dictionary",
        ),
        metadata: dict[str, object] | None = Field(
            None, description="Optional metadata dictionary"
        ),
        created_at: datetime.datetime | str | None = Field(
            None,
            description="Optional created-at timestamp for the message.",
        ),
    ) -> MessageCreateParams:
        """
        Create a MessageCreateParams object attributed to this peer.

        Args:
            content: The text content for the message
            config: Optional configuration dictionary
            metadata: Optional metadata dictionary
            created_at: Optional created-at timestamp

        Returns:
            A new MessageCreateParams object
        """
        from ..api_types import MessageConfiguration

        created_at_dt: datetime.datetime | None
        if isinstance(created_at, str):
            created_at_dt = datetime.datetime.fromisoformat(created_at)
        else:
            created_at_dt = created_at

        config_obj = MessageConfiguration(**config) if config else None

        return MessageCreateParams(
            peer_id=self.id,
            content=content,
            configuration=config_obj,
            metadata=metadata,
            created_at=created_at_dt,
        )

    async def get_metadata(self) -> dict[str, object]:
        """
        Get the current metadata for this peer.

        Returns:
            A dictionary containing the peer's metadata.
        """
        data = await self._http.post(
            routes.peers(self.workspace_id),
            body={"id": self.id},
        )
        peer = PeerResponse.model_validate(data)
        self._metadata = peer.metadata or {}
        return self._metadata

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(
            ..., description="Metadata dictionary to associate with this peer"
        ),
    ) -> None:
        """
        Set the metadata for this peer.

        Args:
            metadata: A dictionary of metadata
        """
        await self._http.put(
            routes.peer(self.workspace_id, self.id),
            body={"metadata": metadata},
        )
        self._metadata = metadata

    async def get_config(self) -> dict[str, object]:
        """
        Get the current workspace-level configuration for this peer.

        Returns:
            A dictionary containing the peer's configuration
        """
        data = await self._http.post(
            routes.peers(self.workspace_id),
            body={"id": self.id},
        )
        peer = PeerResponse.model_validate(data)
        self._configuration = peer.configuration or {}
        return self._configuration

    @validate_call
    async def set_config(
        self,
        config: dict[str, object] = Field(..., description="Configuration dictionary"),
    ) -> None:
        """
        Set the configuration for this peer.

        Args:
            config: A dictionary of configuration
        """
        await self._http.put(
            routes.peer(self.workspace_id, self.id),
            body={"configuration": config},
        )
        self._configuration = config

    async def get_peer_config(self) -> dict[str, object]:
        """
        Get the current workspace-level configuration for this peer.

        .. deprecated::
            Use :meth:`get_config` instead.
        """
        return await self.get_config()

    @validate_call
    async def set_peer_config(
        self,
        config: dict[str, object] = Field(..., description="Configuration dictionary"),
    ) -> None:
        """
        Set the configuration for this peer.

        .. deprecated::
            Use :meth:`set_config` instead.
        """
        return await self.set_config(config)

    async def refresh(self) -> None:
        """
        Refresh cached metadata and configuration for this peer.
        """
        data = await self._http.post(
            routes.peers(self.workspace_id),
            body={"id": self.id},
        )
        peer = PeerResponse.model_validate(data)
        self._metadata = peer.metadata or {}
        self._configuration = peer.configuration or {}

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1, description="The search query"),
        filters: dict[str, object] | None = Field(
            None, description="Filters to scope the search"
        ),
        limit: int = Field(
            default=10, ge=1, le=100, description="Number of results to return"
        ),
    ) -> list[MessageResponse]:
        """
        Search across all messages with this peer as author.

        Args:
            query: The search query to use
            filters: Filters to scope the search.
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of MessageResponse objects.
        """
        data = await self._http.post(
            routes.peer_search(self.workspace_id, self.id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [MessageResponse.model_validate(item) for item in data]

    async def card(
        self,
        target: str | PeerBase | None = None,
    ) -> str:
        """
        Get the peer card for this peer.

        Args:
            target: Optional target peer for local card.

        Returns:
            A string containing the peer card, or empty string if none
        """
        if isinstance(target, str) and len(target.strip()) == 0:
            raise ValueError("target string cannot be empty")

        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )

        query = {"target": target_id} if target_id else None
        data = await self._http.get(
            routes.peer_card(self.workspace_id, self.id),
            query=query,
        )
        response = PeerCardResponse.model_validate(data)

        if response.peer_card is None:
            return ""

        return "\n".join(response.peer_card)

    async def get_representation(
        self,
        session: str | SessionBase | None = None,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> str:
        """
        Get a subset of the representation of the peer.

        Args:
            session: Optional session to scope the representation to.
            target: Optional target peer to get the representation of.
            search_query: Semantic search query to filter relevant conclusions
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results
            include_most_frequent: Whether to include the most frequent conclusions
            max_conclusions: Maximum number of conclusions to include

        Returns:
            A representation string
        """
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

        data = await self._http.post(
            routes.peer_representation(self.workspace_id, self.id),
            body=body,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    async def get_context(
        self,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> PeerContextResponse:
        """
        Get context for this peer, including representation and peer card.

        Args:
            target: Optional target peer to get context for.
            search_query: Semantic search query to filter relevant conclusions
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results
            include_most_frequent: Whether to include the most frequent conclusions
            max_conclusions: Maximum number of conclusions to include

        Returns:
            A PeerContextResponse object
        """
        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )

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

        data = await self._http.get(
            routes.peer_context(self.workspace_id, self.id),
            query=query if query else None,
        )
        return PeerContextResponse.model_validate(data)

    @property
    def conclusions(self) -> "AsyncConclusionScope":
        """
        Access this peer's self-conclusions (where observer == observed == self).

        Returns:
            An AsyncConclusionScope scoped to this peer's self-conclusions
        """
        from ..conclusions import AsyncConclusionScope

        return AsyncConclusionScope(self._http, self.workspace_id, self.id, self.id)

    def conclusions_of(self, target: str | PeerBase) -> "AsyncConclusionScope":
        """
        Access conclusions this peer has made about another peer.

        Args:
            target: The target peer (either a Peer object or peer ID string)

        Returns:
            An AsyncConclusionScope scoped to this peer's conclusions of the target
        """
        from ..conclusions import AsyncConclusionScope

        target_id = target.id if isinstance(target, PeerBase) else target
        return AsyncConclusionScope(self._http, self.workspace_id, self.id, target_id)

    def __repr__(self) -> str:
        return f"AsyncPeer(id='{self.id}')"

    def __str__(self) -> str:
        return self.id


# Import for type hints
from ..conclusions import AsyncConclusionScope  # noqa: E402
