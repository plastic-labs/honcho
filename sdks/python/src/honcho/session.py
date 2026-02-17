# pyright: reportPrivateUsage=false
"""Sync Session class for Honcho SDK."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field, PrivateAttr, validate_call

from .api_types import (
    MessageCreateParams,
    MessageResponse,
    PeerResponse,
    QueueStatusResponse,
    RepresentationResponse,
    SessionConfiguration,
    SessionPeerConfig,
    SessionResponse,
)
from .base import PeerBase, SessionBase
from .http import routes
from .message import Message
from .mixins import MetadataConfigMixin
from .pagination import SyncPage
from .peer import Peer
from .session_context import SessionContext, SessionSummaries, Summary
from .utils import (
    datetime_to_iso,
    normalize_peers_to_dict,
    prepare_file_for_upload,
    resolve_id,
)

if TYPE_CHECKING:
    from .aio import SessionAio
    from .client import Honcho

logger = logging.getLogger(__name__)

__all__ = ["Session", "SessionPeerConfig"]


class Session(SessionBase, MetadataConfigMixin):
    """
    Represents a session in Honcho.

    Sessions are scoped to a set of peers and contain messages/content.
    They create bidirectional relationships between peers and provide
    a context for multi-party conversations and interactions.

    Attributes:
        id: Unique identifier for this session
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this session. May be stale if not recently
            fetched. Call get_metadata() for fresh data.
        configuration: Cached configuration for this session. May be stale if not
            recently fetched. Call get_configuration() for fresh data.
    """

    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: SessionConfiguration | None = PrivateAttr(default=None)
    _honcho: "Honcho" = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this session. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> SessionConfiguration | None:
        """Cached configuration for this session. May be stale. Use get_configuration() for fresh data."""
        return self._configuration

    # MetadataConfigMixin implementation
    def _get_http_client(self):
        self._honcho._ensure_workspace()
        return self._honcho._http

    def _get_fetch_route(self) -> str:
        return routes.sessions(self.workspace_id)

    def _get_update_route(self) -> str:
        return routes.session(self.workspace_id, self.id)

    def _get_fetch_body(self) -> dict[str, Any]:
        return {"id": self.id}

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, object], dict[str, object]]:
        session = SessionResponse.model_validate(data)
        # Return configuration as dict for mixin compatibility
        return session.metadata or {}, session.configuration.model_dump(
            exclude_none=True
        )

    def get_configuration(self) -> SessionConfiguration:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Get configuration from the server and update the cache.

        Returns:
            A SessionConfiguration object containing the configuration settings.
        """
        self._honcho._ensure_workspace()
        data = self._get_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        session = SessionResponse.model_validate(data)
        self._metadata = session.metadata or {}
        self._configuration = session.configuration
        return self._configuration

    @validate_call
    def set_configuration(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        configuration: SessionConfiguration = Field(
            ..., description="Configuration to set"
        ),
    ) -> None:
        """
        Set configuration on the server and update the cache.

        Args:
            configuration: A SessionConfiguration object with configuration settings.
        """
        self._get_http_client().put(
            self._get_update_route(),
            body={"configuration": configuration.model_dump(exclude_none=True)},
        )
        self._configuration = configuration

    @property
    def aio(self) -> "SessionAio":
        """
        Access async versions of all Session methods.

        Returns a SessionAio view that provides async versions of all methods
        while sharing state with this Session instance.

        Example:
            ```python
            session = honcho.session("session-123")

            # Async operations
            await session.aio.add_messages(peer.message("Hello"))
            async for msg in session.aio.messages():
                print(msg.content)
            ```
        """
        # Import here to avoid circular import (aio.py imports Session)
        from .aio import SessionAio

        return SessionAio(self)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        session_id: str = Field(
            ..., min_length=1, description="Unique identifier for this session"
        ),
        honcho: Any = Field(..., description="Honcho client instance"),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this session. If set, will get/create session immediately with metadata.",
        ),
        configuration: SessionConfiguration | None = Field(
            None,
            description="Optional configuration to set for this session. If set, will get/create session immediately with flags.",
        ),
    ) -> None:
        """
        Initialize a new Session.

        Provided metadata and configuration will overwrite any existing data in those
        locations if given.

        Args:
            session_id: Unique identifier for this session within the workspace
            honcho: Honcho client instance
            metadata: Optional metadata dictionary to associate with this session.
                If set, will get/create session immediately with metadata.
            configuration: Optional configuration to set for this session.
                If set, will get/create session immediately with flags.
        """
        super().__init__(
            id=session_id,
            workspace_id=honcho.workspace_id,
        )
        self._honcho = honcho
        self._metadata = metadata
        self._configuration = configuration

        if configuration is not None or metadata is not None:
            self._honcho._ensure_workspace()
            body: dict[str, Any] = {"id": session_id}
            if metadata is not None:
                body["metadata"] = metadata
            if configuration is not None:
                body["configuration"] = configuration.model_dump(exclude_none=True)

            data = honcho._http.post(routes.sessions(honcho.workspace_id), body=body)
            session_data = SessionResponse.model_validate(data)
            # Update cached values with API response
            self._metadata = session_data.metadata
            self._configuration = session_data.configuration  # pyright: ignore[reportIncompatibleVariableOverride]

    def add_peers(
        self,
        peers: str
        | PeerBase
        | tuple[str, SessionPeerConfig]
        | tuple[PeerBase, SessionPeerConfig]
        | list[PeerBase | str]
        | list[tuple[PeerBase | str, SessionPeerConfig]]
        | list[PeerBase | str | tuple[PeerBase | str, SessionPeerConfig]] = Field(
            ..., description="Peers to add to the session"
        ),
    ) -> None:
        """
        Add peers to this session.

        Makes an API call to add one or more peers to this session. Adding peers
        creates bidirectional relationships and allows them to participate in
        the session's conversations.

        Args:
            peers: Peers to add to the session. Can be:
                - str: Single peer ID
                - Peer: Single Peer object
                - List[Union[Peer, str]]: List of Peer objects and/or peer IDs
                - tuple[str, SessionPeerConfig]: Single peer ID and SessionPeerConfig
                - tuple[Peer, SessionPeerConfig]: Single Peer object and SessionPeerConfig
                - List[tuple[Union[Peer, str], SessionPeerConfig]]: List of Peer objects and/or peer IDs and SessionPeerConfig
                - Mixed lists with peers and tuples/lists containing peer+config combinations
        """
        self._honcho._ensure_workspace()
        self._honcho._http.post(
            routes.session_peers(self.workspace_id, self.id),
            body=normalize_peers_to_dict(peers),
        )

    def set_peers(
        self,
        peers: str
        | PeerBase
        | tuple[str, SessionPeerConfig]
        | tuple[PeerBase, SessionPeerConfig]
        | list[PeerBase | str]
        | list[tuple[PeerBase | str, SessionPeerConfig]]
        | list[PeerBase | str | tuple[PeerBase | str, SessionPeerConfig]] = Field(
            ..., description="Peers to set for the session"
        ),
    ) -> None:
        """
        Set the complete peer list for this session.

        Makes an API call to replace the current peer list with the provided peers.
        This will remove any peers not in the new list and add any that are missing.

        Args:
            peers: Peers to set for the session. Can be:
                - str: Single peer ID
                - Peer: Single Peer object
                - List[Union[Peer, str]]: List of Peer objects and/or peer IDs
                - tuple[str, SessionPeerConfig]: Single peer ID and SessionPeerConfig
                - tuple[Peer, SessionPeerConfig]: Single Peer object and SessionPeerConfig
                - List[tuple[Union[Peer, str], SessionPeerConfig]]: List of Peer objects and/or peer IDs and SessionPeerConfig
                - Mixed lists with peers and tuples/lists containing peer+config combinations
        """
        self._honcho._ensure_workspace()
        self._honcho._http.put(
            routes.session_peers(self.workspace_id, self.id),
            body=normalize_peers_to_dict(peers),
        )

    def remove_peers(
        self,
        peers: str | PeerBase | list[PeerBase | str] = Field(
            ..., description="Peers to remove from the session"
        ),
    ) -> None:
        """
        Remove peers from this session.

        Makes an API call to remove one or more peers from this session.
        Removed peers will no longer be able to participate in the session
        unless added back.

        Args:
            peers: Peers to remove from the session. Can be:
                   - str: Single peer ID
                   - Peer: Single Peer object
                   - List[Union[Peer, str]]: List of Peer objects and/or peer IDs
        """
        self._honcho._ensure_workspace()
        if not isinstance(peers, list):
            peers = [peers]

        peer_ids = [peer if isinstance(peer, str) else peer.id for peer in peers]

        self._honcho._http.delete(
            routes.session_peers(self.workspace_id, self.id),
            body=peer_ids,
        )

    def peers(self) -> list[Peer]:
        """
        Get all peers in this session.

        Makes an API call to retrieve the list of peer IDs that are currently
        members of this session. Automatically converts the paginated response
        into a list for us -- the max number of peers in a session is usually 10.

        Returns:
            A list of Peer objects that are members of this session
        """
        self._honcho._ensure_workspace()
        data: dict[str, Any] = self._honcho._http.get(
            routes.session_peers(self.workspace_id, self.id)
        )

        peers_data: list[Any] = data.get("items", [])
        return [
            Peer(PeerResponse.model_validate(peer).id, self._honcho)
            for peer in peers_data
        ]

    def get_peer_configuration(self, peer: str | PeerBase) -> SessionPeerConfig:
        """
        Get the configuration for a peer in this session.
        """
        self._honcho._ensure_workspace()
        peer_id = peer if isinstance(peer, str) else peer.id
        data = self._honcho._http.get(
            routes.session_peer_config(self.workspace_id, self.id, peer_id)
        )
        return SessionPeerConfig(
            observe_others=data.get("observe_others"),
            observe_me=data.get("observe_me"),
        )

    def set_peer_configuration(
        self, peer: str | PeerBase, configuration: SessionPeerConfig
    ) -> None:
        """
        Set the configuration for a peer in this session.
        """
        self._honcho._ensure_workspace()
        peer_id = peer if isinstance(peer, str) else peer.id
        body: dict[str, Any] = {}
        if configuration.observe_others is not None:
            body["observe_others"] = configuration.observe_others
        if configuration.observe_me is not None:
            body["observe_me"] = configuration.observe_me

        self._honcho._http.put(
            routes.session_peer_config(self.workspace_id, self.id, peer_id),
            body=body,
        )

    @validate_call
    def add_messages(
        self,
        messages: MessageCreateParams | list[MessageCreateParams] = Field(
            ..., description="Messages to add to the session"
        ),
    ) -> list[Message]:
        """
        Add one or more messages to this session.

        Makes an API call to store messages in this session. Any message added
        to a session will automatically add the creating peer to the session
        if they are not already a member.

        Args:
            messages: Messages to add to the session. Can be:
                      - MessageCreateParams: Single MessageCreateParams object
                      - List[MessageCreateParams]: List of MessageCreateParams objects
        """
        self._honcho._ensure_workspace()
        if not isinstance(messages, list):
            messages = [messages]

        messages_data = [
            msg.model_dump(mode="json", exclude_none=True) for msg in messages
        ]

        data = self._honcho._http.post(
            routes.messages(self.workspace_id, self.id),
            body={"messages": messages_data},
        )
        return [
            Message.from_api_response(MessageResponse.model_validate(msg))
            for msg in data
        ]

    @validate_call
    def messages(
        self,
        *,
        filters: dict[str, object] | None = Field(
            None, description="Dictionary of filter criteria"
        ),
    ) -> SyncPage[MessageResponse, Message]:
        """
        Get messages from this session with optional filtering.

        Makes an API call to retrieve messages from this session. Results can be
        filtered based on various criteria.

        Args:
            filters: Dictionary of filter criteria. Supported filters include:
                    - peer_id: Filter messages by the peer who created them
                    - metadata: Filter messages by metadata key-value pairs
                    - timestamp_start: Filter messages after a specific timestamp
                    - timestamp_end: Filter messages before a specific timestamp

        Returns:
            A list of Message objects matching the specified criteria, ordered by
            creation time (most recent first)
        """
        self._honcho._ensure_workspace()
        data = self._honcho._http.post(
            routes.messages_list(self.workspace_id, self.id),
            body={"filters": filters} if filters else None,
        )

        def transform(response: MessageResponse) -> Message:
            return Message.from_api_response(response)

        def fetch_next(page: int) -> SyncPage[MessageResponse, Message]:
            next_data = self._honcho._http.post(
                routes.messages_list(self.workspace_id, self.id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return SyncPage(next_data, MessageResponse, transform, fetch_next)

        return SyncPage(data, MessageResponse, transform, fetch_next)

    def delete(self) -> None:
        """
        Delete this session and all associated data.

        Makes an API call to permanently delete this session and all related data including:
        - Messages
        - Message embeddings
        - Conclusions
        - Session-Peer associations
        - Background processing queue items

        This action cannot be undone.
        """
        self._honcho._ensure_workspace()
        self._honcho._http.delete(routes.session(self.workspace_id, self.id))

    def clone(
        self,
        *,
        message_id: str | None = None,
    ) -> "Session":
        """
        Clone this session, optionally up to a specific message.

        Makes an API call to create a copy of this session with a new ID.
        All messages and peers from the original session are copied to the new session.
        If a message_id is provided, only messages up to and including that message
        are copied.

        Args:
            message_id: Optional message ID to cut off the clone at. If provided,
                       the cloned session will only contain messages up to and
                       including this message.

        Returns:
            A new Session object representing the cloned session

        Example:
            ```python
            # Clone entire session
            cloned = session.clone()

            # Clone session up to a specific message
            cloned = session.clone(message_id="msg_abc123")
            ```
        """
        self._honcho._ensure_workspace()
        query: dict[str, Any] = {}
        if message_id is not None:
            query["message_id"] = message_id

        data = self._honcho._http.post(
            routes.session_clone(self.workspace_id, self.id),
            query=query if query else None,
        )
        cloned = SessionResponse.model_validate(data)
        return Session(
            cloned.id,
            self._honcho,
            metadata=cloned.metadata,
            configuration=cloned.configuration,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def context(
        self,
        *,
        summary: bool = True,
        tokens: int | None = Field(
            None, gt=0, description="Maximum number of tokens to include in the context"
        ),
        peer_target: str | None = Field(
            None,
            description="A peer ID to get context for. If given *without* `peer_perspective`, a representation and peer card will be included from the omniscient Honcho-level view of `peer_target`. If given *with* `peer_perspective`, will get the representation and card for `peer_target` *from the perspective of `peer_perspective`*.",
        ),
        search_query: str | Message | None = Field(
            None,
            description="A query string (or Message object) used to fetch semantically relevant conclusions. Use this alongside `peer_target` to get a more focused context -- does nothing if `peer_target` is not provided.",
        ),
        peer_perspective: str | None = Field(
            None,
            description="A peer ID to get context *from the perspective of*. If given, response will attempt to include representation and card from the perspective of `peer_perspective`. Must be provided with `peer_target`.",
        ),
        limit_to_session: bool = Field(
            False,
            description="Whether to limit the representation to this session only. If True, only conclusions from this session will be included.",
        ),
        search_top_k: int | None = Field(
            None,
            ge=1,
            le=100,
            description="Number of semantically relevant facts to return when searching with `search_query`.",
        ),
        search_max_distance: float | None = Field(
            None,
            ge=0.0,
            le=1.0,
            description="Maximum semantic distance for search results (0.0-1.0) when searching with `search_query`.",
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
        """
        Get optimized context for this session within a token limit.

        Makes an API call to retrieve a curated list of messages that provides
        optimal context for the conversation while staying within the specified
        token limit. Uses tiktoken for token counting, so results should be
        compatible with OpenAI models.

        Args:
            summary: Whether to include summary information
            tokens: Maximum number of tokens to include in the context. Will default
            to Honcho server configuration if not provided.
            peer_target: A peer ID to get context for.
            search_query: A query string for semantic search.
            peer_perspective: A peer ID to get context from the perspective of.
            limit_to_session: Whether to limit the representation to this session only.
            search_top_k: Number of semantically relevant facts to return.
            search_max_distance: Maximum semantic distance for search results.
            include_most_frequent: Whether to include the most frequent conclusions.
            max_conclusions: Maximum number of conclusions to include.

        Returns:
            A SessionContext object containing the optimized message history and
            summary, if available, that maximizes conversational context while
            respecting the token limit

        Note:
            Token counting is performed using tiktoken. For models using different
            tokenizers, you may need to adjust the token limit accordingly.
        """
        self._honcho._ensure_workspace()

        if peer_target is None and peer_perspective is not None:
            raise ValueError(
                "You must provide a `peer_target` when `peer_perspective` is provided"
            )

        if peer_target is None and search_query is not None:
            raise ValueError(
                "You must provide a `peer_target` when `search_query` is provided"
            )

        search_query_text = (
            search_query.content if isinstance(search_query, Message) else search_query
        )

        query: dict[str, Any] = {
            "summary": summary,
            "limit_to_session": limit_to_session,
        }
        if tokens is not None:
            query["tokens"] = tokens
        if search_query_text is not None:
            query["search_query"] = search_query_text
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

        data = self._honcho._http.get(
            routes.session_context(self.workspace_id, self.id),
            query=query,
        )

        # Convert summary if present
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
            Message.from_api_response(MessageResponse.model_validate(msg))
            for msg in data.get("messages", [])
        ]

        return SessionContext(
            session_id=self.id,
            messages=messages,
            summary=session_summary,
            peer_representation=str(data.get("peer_representation"))
            if data.get("peer_representation")
            else None,
            peer_card=data.get("peer_card"),
        )

    def summaries(self) -> SessionSummaries:
        """
        Get available summaries for this session.

        Makes an API call to retrieve both short and long summaries for this session,
        if they are available. Summaries are created asynchronously by the backend
        as messages are added to the session.

        Returns:
            A SessionSummaries object containing:
            - id: The session ID
            - short_summary: The short summary if available, including metadata
            - long_summary: The long summary if available, including metadata

        Note:
            Summaries may be None if:
            - Not enough messages have been added to trigger summary generation
            - The summary generation is still in progress
            - Summary generation is disabled for this session
        """
        self._honcho._ensure_workspace()
        data = self._honcho._http.get(
            routes.session_summaries(self.workspace_id, self.id)
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
            id=data.get("id") or self.id,
            short_summary=short_summary,
            long_summary=long_summary,
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
        Search for messages in this session.

        Makes an API call to search for messages in this session.

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
            routes.session_search(self.workspace_id, self.id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [
            Message.from_api_response(MessageResponse.model_validate(msg))
            for msg in data
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def upload_file(
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
            description="Optional created-at timestamp for the messages. Should be an ISO 8601 formatted string.",
        ),
    ) -> list[Message]:
        """
        Upload file to create message(s) in this session.

        Accepts a flexible payload:
        - File objects (opened in binary mode)
        - (filename, bytes, content_type) tuples
        - (filename, fileobj, content_type) tuples

        Files are normalized to (filename, fileobj, content_type) tuples for the HTTP client.

        Args:
            file: File to upload. Can be:
                - a file object (must have .name and .read())
                - a tuple (filename, bytes, content_type)
                - a tuple (filename, fileobj, content_type)
            peer: The peer who will be attributed as the creator of the messages.
                Can be a peer ID string or a Peer object.
            metadata: Optional metadata dictionary to associate with the messages
            configuration: Optional configuration dictionary to associate with the messages
            created_at: Optional created-at timestamp for the messages. Should be an ISO 8601 formatted string.

        Returns:
            A list of Message objects representing the created messages

        Note:
            Supported file types include PDFs, text files, and JSON documents.
            Large files will be automatically split into multiple messages to fit
            within message size limits.
        """
        self._honcho._ensure_workspace()

        # Prepare file for upload using shared utility
        filename, content_bytes, content_type = prepare_file_for_upload(file)

        # Extract peer ID from Peer object if needed
        resolved_peer_id = peer if isinstance(peer, str) else peer.id

        # Build form data
        data_dict: dict[str, str] = {"peer_id": resolved_peer_id}
        if metadata is not None:
            data_dict["metadata"] = json.dumps(metadata)
        if configuration is not None:
            data_dict["configuration"] = json.dumps(configuration)
        created_at_iso = datetime_to_iso(created_at)
        if created_at_iso is not None:
            data_dict["created_at"] = created_at_iso

        response = self._honcho._http.upload(
            routes.messages_upload(self.workspace_id, self.id),
            files={"file": (filename, content_bytes, content_type)},
            data=data_dict,
        )

        return [
            Message.from_api_response(MessageResponse.model_validate(msg))
            for msg in response
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def representation(
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
        """
        Get a subset of the representation of the peer in this session.

        Args:
            peer: Peer to get the representation of.
            target: Optional target peer to get the representation of. If provided,
            queries what `peer` knows about the `target`.
            search_query: Semantic search query to filter relevant conclusions
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_frequent: Whether to include the most frequent conclusions
            max_conclusions: Maximum number of conclusions to include

        Returns:
            A Representation string

        Example:
            ```python
            # Get peer's representation in this session
            rep = session.representation('user123')
            print(rep)

            # Get what user123 knows about assistant in this session
            local_rep = session.representation('user123', target='assistant')

            # Get representation with semantic search
            searched_rep = session.representation(
                'user123',
                search_query='preferences',
                search_top_k=10
            )
            ```
        """
        self._honcho._ensure_workspace()
        peer_id = resolve_id(peer)
        target_id = resolve_id(target)

        query: dict[str, Any] = {"session_id": self.id}
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

        data = self._honcho._http.post(
            routes.peer_representation(self.workspace_id, peer_id),
            body=query,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
    ) -> QueueStatusResponse:
        """
        Get the queue processing status, optionally scoped to an observer, sender, and/or session.

        Args:
            observer: Optional observer (ID string or Peer object) to scope the status check
            sender: Optional sender (ID string or Peer object) to scope the status check
        """
        self._honcho._ensure_workspace()
        resolved_observer_id = resolve_id(observer)
        resolved_sender_id = resolve_id(sender)

        query: dict[str, Any] = {"session_id": self.id}
        if resolved_observer_id:
            query["observer_id"] = resolved_observer_id
        if resolved_sender_id:
            query["sender_id"] = resolved_sender_id

        data = self._honcho._http.get(
            routes.workspace_queue_status(self.workspace_id),
            query=query,
        )
        return QueueStatusResponse.model_validate(data)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def update_message(
        self,
        message: Message | str = Field(
            ..., description="The Message object or message ID to update"
        ),
        metadata: dict[str, object] = Field(
            ..., description="The metadata to update for the message"
        ),
    ) -> Message:
        """
        Update the metadata of a message in this session.

        Makes an API call to update the metadata of a specific message within this session.

        Args:
            message: Either a Message object or a message ID string
            metadata: The metadata to update for the message

        Returns:
            The updated Message object
        """
        self._honcho._ensure_workspace()
        message_id = message.id if isinstance(message, Message) else message

        data = self._honcho._http.put(
            routes.message(self.workspace_id, self.id, message_id),
            body={"metadata": metadata},
        )
        return Message.from_api_response(MessageResponse.model_validate(data))

    def __repr__(self) -> str:
        """
        Return a string representation of the Session.

        Returns:
            A string representation suitable for debugging
        """
        return f"Session(id='{self.id}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Session.

        Returns:
            The session's ID
        """
        return self.id
