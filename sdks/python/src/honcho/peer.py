# pyright: reportPrivateUsage=false
"""Sync Peer class for Honcho SDK."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr, validate_call

from .api_types import (
    MessageCreateParams,
    MessageResponse,
    PeerCardResponse,
    PeerConfig,
    PeerContextResponse,
    PeerResponse,
    RepresentationResponse,
    SessionResponse,
)
from .base import PeerBase, SessionBase
from .conclusions import ConclusionScope
from .http import routes
from .message import Message
from .mixins import MetadataConfigMixin
from .pagination import SyncPage
from .types import DialecticStreamResponse
from .utils import parse_datetime, parse_sse_stream, resolve_id

if TYPE_CHECKING:
    from .aio import PeerAio
    from .client import Honcho
    from .session import Session

logger = logging.getLogger(__name__)


class Peer(PeerBase, MetadataConfigMixin):
    """
    Represents a peer in the Honcho system.

    Peers can send messages, participate in sessions, and maintain both global
    and local representations for contextual interactions. A peer represents
    an entity (user, assistant, etc.) that can communicate within the system.

    Attributes:
        id: Unique identifier for this peer
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this peer. May be stale if not recently
            fetched. Call get_metadata() for fresh data.
        configuration: Cached configuration for this peer. May be stale if not
            recently fetched. Call get_configuration() for fresh data.
    """

    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: PeerConfig | None = PrivateAttr(default=None)
    _honcho: "Honcho" = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this peer. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> PeerConfig | None:
        """Cached configuration for this peer. May be stale. Use get_configuration() for fresh data."""
        return self._configuration

    # MetadataConfigMixin implementation
    def _get_http_client(self):
        self._honcho._ensure_workspace()
        return self._honcho._http

    def _get_fetch_route(self) -> str:
        return routes.peers(self.workspace_id)

    def _get_update_route(self) -> str:
        return routes.peer(self.workspace_id, self.id)

    def _get_fetch_body(self) -> dict[str, Any]:
        return {"id": self.id}

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, object], dict[str, object]]:
        peer = PeerResponse.model_validate(data)
        # Return configuration as dict for mixin compatibility
        return peer.metadata or {}, peer.configuration.model_dump(exclude_none=True)

    def get_configuration(self) -> PeerConfig:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Get configuration from the server and update the cache.

        Returns:
            A PeerConfig object containing the configuration settings.
        """
        self._honcho._ensure_workspace()
        data = self._get_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        peer = PeerResponse.model_validate(data)
        self._metadata = peer.metadata or {}
        self._configuration = peer.configuration
        return self._configuration

    @validate_call
    def set_configuration(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        configuration: PeerConfig = Field(..., description="Configuration to set"),
    ) -> None:
        """
        Set configuration on the server and update the cache.

        Args:
            configuration: A PeerConfig object with configuration settings.
        """
        self._get_http_client().put(
            self._get_update_route(),
            body={"configuration": configuration.model_dump(exclude_none=True)},
        )
        self._configuration = configuration

    @property
    def aio(self) -> "PeerAio":
        """
        Access async versions of all Peer methods.

        Returns a PeerAio view that provides async versions of all methods
        while sharing state with this Peer instance.

        Example:
            ```python
            peer = honcho.peer("user-123")

            # Async operations
            await peer.aio.chat("query")
            await peer.aio.get_metadata()
            ```
        """
        # Import here to avoid circular import (aio.py imports Peer)
        from .aio import PeerAio

        return PeerAio(self)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        peer_id: str = Field(
            ...,
            min_length=1,
            description="Unique identifier for this peer within the workspace",
        ),
        honcho: Any = Field(..., description="Honcho client instance"),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this peer. If set, will get/create peer immediately with metadata.",
        ),
        configuration: PeerConfig | None = Field(
            None,
            description="Optional configuration to set for this peer. If set, will get/create peer immediately with flags.",
        ),
    ) -> None:
        """
        Initialize a new Peer.

        Provided metadata and configuration will overwrite any existing data in those
        locations if given.

        Args:
            peer_id: Unique identifier for this peer within the workspace
            honcho: Honcho client instance
            metadata: Optional metadata dictionary to associate with this peer.
                If set, will get/create peer immediately with metadata.
            configuration: Optional configuration to set for this peer.
                If set, will get/create peer immediately with flags.
        """
        super().__init__(
            id=peer_id,
            workspace_id=honcho.workspace_id,
        )
        self._honcho = honcho
        self._metadata = metadata
        self._configuration = configuration

        if configuration is not None or metadata is not None:
            self._honcho._ensure_workspace()
            body: dict[str, Any] = {"id": peer_id}
            if metadata is not None:
                body["metadata"] = metadata
            if configuration is not None:
                body["configuration"] = configuration.model_dump(exclude_none=True)

            data = honcho._http.post(routes.peers(honcho.workspace_id), body=body)
            peer_data = PeerResponse.model_validate(data)
            # Update cached values with API response
            self._metadata = peer_data.metadata
            self._configuration = peer_data.configuration  # pyright: ignore[reportIncompatibleVariableOverride]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def chat(
        self,
        query: str = Field(..., min_length=1, description="The natural language query"),
        *,
        target: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
        reasoning_level: Literal["minimal", "low", "medium", "high", "max"]
        | None = None,
    ) -> str | None:
        """
        Query the peer's representation with a natural language question.

        Makes an API call to the Honcho dialectic endpoint to query either the peer's
        global representation (all content associated with this peer) or their local
        representation of another peer (what this peer knows about the target peer).

        Args:
            query: The natural language question to ask.
            target: Optional target peer for local representation query. If provided,
                    queries what this peer knows about the target peer rather than
                    querying the peer's global representation. Can be a peer ID string
                    or a Peer object.
            session: Optional session to scope the query to. If provided, only
                     information from that session is considered. Can be a session
                     ID string or a Session object.
            reasoning_level: Optional reasoning level for the query: "minimal", "low", "medium",
                             "high", or "max". Defaults to "low" if not provided.

        Returns:
            Response string containing the answer, or None if no relevant information
        """
        self._honcho._ensure_workspace()
        target_id = resolve_id(target)
        resolved_session_id = resolve_id(session)

        body: dict[str, Any] = {"query": query, "stream": False}
        if target_id:
            body["target"] = target_id
        if resolved_session_id:
            body["session_id"] = resolved_session_id
        if reasoning_level:
            body["reasoning_level"] = reasoning_level

        data = self._honcho._http.post(
            routes.peer_chat(self.workspace_id, self.id),
            body=body,
        )
        content = data.get("content")
        if not content:
            return None
        return content

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def chat_stream(
        self,
        query: str = Field(..., min_length=1, description="The natural language query"),
        *,
        target: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
        reasoning_level: Literal["minimal", "low", "medium", "high", "max"]
        | None = None,
    ) -> DialecticStreamResponse:
        """
        Query the peer's representation with a natural language question, streaming the response.

        Makes an API call to the Honcho dialectic endpoint to query either the peer's
        global representation (all content associated with this peer) or their local
        representation of another peer (what this peer knows about the target peer).

        Args:
            query: The natural language question to ask.
            target: Optional target peer for local representation query. If provided,
                    queries what this peer knows about the target peer rather than
                    querying the peer's global representation. Can be a peer ID string
                    or a Peer object.
            session: Optional session to scope the query to. If provided, only
                     information from that session is considered. Can be a session
                     ID string or a Session object.
            reasoning_level: Optional reasoning level for the query: "minimal", "low", "medium",
                             "high", or "max". Defaults to "low" if not provided.

        Returns:
            DialecticStreamResponse object that can be iterated over and provides final response
        """
        self._honcho._ensure_workspace()
        target_id = resolve_id(target)
        resolved_session_id = resolve_id(session)

        body: dict[str, Any] = {"query": query, "stream": True}
        if target_id:
            body["target"] = target_id
        if resolved_session_id:
            body["session_id"] = resolved_session_id
        if reasoning_level:
            body["reasoning_level"] = reasoning_level

        def stream_response() -> Generator[str, None, None]:
            yield from parse_sse_stream(
                self._honcho._http.stream(
                    "POST",
                    routes.peer_chat(self.workspace_id, self.id),
                    body=body,
                )
            )

        return DialecticStreamResponse(stream_response())

    def sessions(
        self, filters: dict[str, object] | None = None
    ) -> SyncPage[SessionResponse, "Session"]:
        """
        Get all sessions this peer is a member of.

        Makes an API call to retrieve all sessions where this peer is an active participant.
        Sessions are created when peers are added to them or send messages to them.

        Returns:
            A paginated list of Session objects this peer belongs to. Returns an empty
            list if the peer is not a member of any sessions
        """
        self._honcho._ensure_workspace()
        # Import here to avoid circular import (session.py imports Peer)
        from .session import Session

        data = self._honcho._http.post(
            routes.peer_sessions_list(self.workspace_id, self.id),
            body={"filters": filters} if filters else None,
        )

        def transform(session: SessionResponse) -> Session:
            return Session(session.id, self._honcho)

        def fetch_next(page: int) -> SyncPage[SessionResponse, Session]:
            next_data = self._honcho._http.post(
                routes.peer_sessions_list(self.workspace_id, self.id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return SyncPage(next_data, SessionResponse, transform, fetch_next)

        return SyncPage(data, SessionResponse, transform, fetch_next)

    @validate_call
    def message(
        self,
        content: str = Field(
            ..., min_length=0, description="The text content for the message"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None, description="Optional metadata dictionary"
        ),
        configuration: dict[str, Any] | None = Field(
            None,
            description="Optional configuration dictionary to associate with the message",
        ),
        created_at: datetime.datetime | str | None = Field(
            None,
            description="Optional created-at timestamp for the message. Accepts a datetime which will be converted to an ISO 8601 string, or a preformatted string.",
        ),
    ) -> MessageCreateParams:
        """
        Build a message object attributed to this peer (synchronous, no API call).

        This is a convenience method for creating message objects with this peer's ID
        already set. The returned object can then be passed to `session.add_messages()`.

        Note:
            This method is synchronous and does NOT send the message to Honcho.
            To actually create the message on the server, pass the returned object to
            `session.add_messages()`.

        Args:
            content: The text content for the message
            metadata: Optional metadata dictionary to associate with the message
            configuration: Optional configuration dictionary (e.g., reasoning settings)
            created_at: Optional created-at timestamp

        Returns:
            A MessageCreateParams object ready to be passed to `session.add_messages()`

        Example:
            ```python
            msg = peer.message("Hello!")
            await session.add_messages(msg)

            # Or batch multiple messages:
            await session.add_messages([
                alice.message("Hi Bob"),
                bob.message("Hey Alice!"),
            ])
            ```
        """
        from .api_types import MessageConfiguration

        if content != "" and content.strip() == "":
            raise ValueError("Message content cannot be only whitespace")

        created_at_dt = parse_datetime(created_at)

        config_obj = MessageConfiguration(**configuration) if configuration else None

        return MessageCreateParams(
            peer_id=self.id,
            content=content,
            configuration=config_obj,
            metadata=metadata,
            created_at=created_at_dt,
        )

    @validate_call
    def search(
        self,
        query: str = Field(..., min_length=1, description="The search query to use"),
        filters: dict[str, object] | None = Field(
            None, description="Filters to scope the search"
        ),
        limit: int = Field(
            default=10, ge=1, le=100, description="Number of results to return"
        ),
    ) -> list[Message]:
        """
        Search across all messages in the workspace with this peer as author.

        Makes an API call to search endpoint.

        Args:
            query: The search query to use
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        self._honcho._ensure_workspace()
        data = self._honcho._http.post(
            routes.peer_search(self.workspace_id, self.id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [
            Message.from_api_response(MessageResponse.model_validate(item))
            for item in data
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def card(
        self,
        target: str | PeerBase | None = None,
    ) -> list[str] | None:
        """
        Get the peer card for this peer.

        Makes an API call to retrieve the peer card, which contains a representation
        of what this peer knows. If a target is provided, returns this peer's local
        representation of the target peer.

        Args:
            target: Optional target peer for local card. If provided, returns this
                    peer's card of the target peer. Can be a Peer object or peer ID string.

        Returns:
            A list of strings representing the peer card, or None if none is available
        """
        self._honcho._ensure_workspace()
        target_id = resolve_id(target)

        query = {"target": target_id} if target_id else None
        data = self._honcho._http.get(
            routes.peer_card(self.workspace_id, self.id),
            query=query,
        )
        response = PeerCardResponse.model_validate(data)

        return response.peer_card

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def representation(
        self,
        session: str | SessionBase | None = None,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = Field(None, ge=1, le=100),
        search_max_distance: float | None = Field(None, ge=0.0, le=1.0),
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = Field(None, ge=1, le=100),
    ) -> str:
        """
        Get a subset of the representation of the peer.

        Args:
            session: Optional session to scope the representation to.
            target: Optional target peer to get the representation of. If provided,
            returns the representation of the target from the perspective of this peer.
            search_query: Semantic search query to filter relevant conclusions
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_frequent: Whether to include the most frequent conclusions
            max_conclusions: Maximum number of conclusions to include

        Returns:
            A Representation string

        Example:
            ```python
            # Get global representation
            rep = peer.representation()
            print(rep)

            # Get representation scoped to a session
            session_rep = peer.representation(session='session-123')

            # Get representation with semantic search
            searched_rep = peer.representation(
                search_query='preferences',
                search_top_k=10,
                max_conclusions=50
            )
            ```
        """
        self._honcho._ensure_workspace()
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

        data = self._honcho._http.post(
            routes.peer_representation(self.workspace_id, self.id),
            body=body,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def context(
        self,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = Field(None, ge=1, le=100),
        search_max_distance: float | None = Field(None, ge=0.0, le=1.0),
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = Field(None, ge=1, le=100),
    ) -> PeerContextResponse:
        """
        Get context for this peer, including representation and peer card.

        This is a convenience method that retrieves both the working representation
        and peer card in a single API call.

        Args:
            target: Optional target peer to get context for. If provided, returns
                   the context for the target from this peer's perspective.
                   Can be a Peer object or peer ID string.
            search_query: Semantic search query to filter relevant conclusions
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_frequent: Whether to include the most frequent conclusions
            max_conclusions: Maximum number of conclusions to include

        Returns:
            A PeerContext object containing the representation and peer card

        Example:
            ```python
            # Get own context
            context = peer.context()
            print(context.representation)
            print(context.peer_card)

            # Get context for another peer
            context = peer.context(target='other-peer-id')

            # Get context with semantic search
            context = peer.context(
                search_query='preferences',
                search_top_k=10
            )
            ```
        """
        self._honcho._ensure_workspace()
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

        data = self._honcho._http.get(
            routes.peer_context(self.workspace_id, self.id),
            query=query if query else None,
        )
        return PeerContextResponse.model_validate(data)

    @property
    def conclusions(self) -> ConclusionScope:
        """
        Access this peer's self-conclusions (where observer == observed == self).

        This property provides a convenient way to access conclusions that this peer
        has made about themselves. Use this for self-conclusion scenarios.

        Returns:
            A ConclusionScope scoped to this peer's self-conclusions

        Example:
            ```python
            # List self-conclusions
            obs_list = peer.conclusions.list()

            # Search self-conclusions
            results = peer.conclusions.query("preferences")

            # Delete a self-conclusion
            peer.conclusions.delete("obs-123")
            ```
        """
        return ConclusionScope(self._honcho, self.workspace_id, self.id, self.id)

    def conclusions_of(self, target: str | PeerBase) -> ConclusionScope:
        """
        Access conclusions this peer has made about another peer.

        This method provides scoped access to conclusions where this peer is the
        observer and the target is the observed peer.

        Args:
            target: The target peer (either a Peer object or peer ID string)

        Returns:
            A ConclusionScope scoped to this peer's conclusions of the target

        Example:
            ```python
            # Get conclusions about another peer
            bob_conclusions = peer.conclusions_of("bob")

            # List conclusions
            obs_list = bob_conclusions.list()

            # Search conclusions
            results = bob_conclusions.query("work history")

            # Get the representation from these conclusions
            rep = bob_conclusions.get_representation()
            ```
        """
        target_id = target.id if isinstance(target, PeerBase) else target
        return ConclusionScope(self._honcho, self.workspace_id, self.id, target_id)

    def __repr__(self) -> str:
        """
        Return a string representation of the Peer.

        Returns:
            A string representation suitable for debugging
        """
        return f"Peer(id='{self.id}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Peer.

        Returns:
            The peer's ID
        """
        return self.id
