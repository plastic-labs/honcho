from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from honcho_core import Honcho as HonchoCore
from honcho_core._types import omit
from honcho_core.types import DeriverStatus
from honcho_core.types.workspaces.sessions import MessageCreateParam
from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from .pagination import SyncPage
from .session_context import SessionContext, SessionSummaries, Summary
from .utils import prepare_file_for_upload

if TYPE_CHECKING:
    from .peer import Peer
    from .types import Representation

logger = logging.getLogger(__name__)


class SessionPeerConfig(BaseModel):
    observe_others: bool | None = Field(
        None,
        description="Whether this peer should form a session-level theory-of-mind representation of other peers in the session",
    )
    observe_me: bool | None = Field(
        None,
        description="Whether other peers in this session should try to form a session-level theory-of-mind representation of this peer",
    )


class Session(BaseModel):
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
            recently fetched. Call get_config() for fresh data.
    """

    id: str = Field(..., min_length=1, description="Unique identifier for this session")
    workspace_id: str = Field(
        ..., min_length=1, description="Workspace ID for scoping operations"
    )
    metadata: dict[str, object] | None = Field(
        None,
        frozen=True,
        description="Cached metadata for this session. May be stale. Use get_metadata() for fresh data.",
    )
    configuration: dict[str, object] | None = Field(
        None,
        frozen=True,
        description="Cached configuration for this session. May be stale. Use get_config() for fresh data.",
    )
    _client: HonchoCore = PrivateAttr()

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        session_id: str = Field(
            ..., min_length=1, description="Unique identifier for this session"
        ),
        workspace_id: str = Field(
            ..., min_length=1, description="Workspace ID for scoping operations"
        ),
        client: HonchoCore = Field(
            ..., description="Reference to the parent Honcho client instance"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this session. If set, will get/create session immediately with metadata.",
        ),
        config: dict[str, object] | None = Field(
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
            workspace_id: Workspace ID for scoping operations
            client: Reference to the parent Honcho client instance
            metadata: Optional metadata dictionary to associate with this session.
            If set, will get/create session immediately with metadata.
            config: Optional configuration to set for this session.
            If set, will get/create session immediately with flags.
        """
        super().__init__(
            id=session_id,
            workspace_id=workspace_id,
            metadata=metadata,
            configuration=config,
        )
        self._client = client

        if config is not None or metadata is not None:
            session_data = self._client.workspaces.sessions.get_or_create(
                workspace_id=workspace_id,
                id=session_id,
                configuration=config if config is not None else omit,
                metadata=metadata if metadata is not None else omit,
            )
            # Update cached values with API response
            object.__setattr__(self, "metadata", session_data.metadata)
            object.__setattr__(self, "configuration", session_data.configuration)

    def add_peers(
        self,
        peers: str
        | Peer
        | tuple[str, SessionPeerConfig]
        | tuple[Peer, SessionPeerConfig]
        | list[Peer | str]
        | list[tuple[Peer | str, SessionPeerConfig]]
        | list[Peer | str | tuple[Peer | str, SessionPeerConfig]] = Field(
            ..., description="Peers to add to the session"
        ),
    ) -> None:
        """
        Add peers to this session.

        Makes an async API call to add one or more peers to this session. Adding peers
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
        if not isinstance(peers, list):
            peers = [peers]

        peer_dict: dict[str, Any] = {}
        for peer in peers:
            if isinstance(peer, tuple):
                # Handle tuple[str/Peer, SessionPeerConfig]
                peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
                peer_config = peer[1]
                peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
            else:
                # Handle direct str or Peer
                peer_id = peer if isinstance(peer, str) else peer.id
                peer_dict[peer_id] = {}

        self._client.workspaces.sessions.peers.add(
            session_id=self.id,
            workspace_id=self.workspace_id,
            body=peer_dict,
        )

    def set_peers(
        self,
        peers: str
        | Peer
        | tuple[str, SessionPeerConfig]
        | tuple[Peer, SessionPeerConfig]
        | list[Peer | str]
        | list[tuple[Peer | str, SessionPeerConfig]]
        | list[Peer | str | tuple[Peer | str, SessionPeerConfig]] = Field(
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
        if not isinstance(peers, list):
            peers = [peers]

        peer_dict: dict[str, Any] = {}
        for peer in peers:
            if isinstance(peer, tuple):
                # Handle tuple[str/Peer, SessionPeerConfig]
                peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
                peer_config = peer[1]
                peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
            else:
                # Handle direct str or Peer
                peer_id = peer if isinstance(peer, str) else peer.id
                peer_dict[peer_id] = {}

        self._client.workspaces.sessions.peers.set(
            session_id=self.id,
            workspace_id=self.workspace_id,
            body=peer_dict,
        )

    def remove_peers(
        self,
        peers: str | Peer | list[Peer | str] = Field(
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
        if not isinstance(peers, list):
            peers = [peers]

        peer_ids = [peer if isinstance(peer, str) else peer.id for peer in peers]

        self._client.workspaces.sessions.peers.remove(
            session_id=self.id,
            workspace_id=self.workspace_id,
            body=peer_ids,
        )

    def get_peers(self) -> list[Peer]:
        """
        Get all peers in this session.

        Makes an API call to retrieve the list of peer IDs that are currently
        members of this session. Automatically converts the paginated response
        into a list for us -- the max number of peers in a session is usually 10.

        Returns:
            A list of Peer objects that are members of this session
        """
        from .peer import Peer

        peers_page = self._client.workspaces.sessions.peers.list(
            session_id=self.id,
            workspace_id=self.workspace_id,
        )
        return [
            Peer(peer.id, self.workspace_id, self._client) for peer in peers_page.items
        ]

    def get_peer_config(self, peer: str | Peer) -> SessionPeerConfig:
        """
        Get the configuration for a peer in this session.
        """
        from .peer import Peer

        peer_get_config_response = self._client.workspaces.sessions.peers.get_config(
            peer_id=str(peer.id) if isinstance(peer, Peer) else peer,
            workspace_id=self.workspace_id,
            session_id=self.id,
        )
        return SessionPeerConfig(
            observe_others=peer_get_config_response.observe_others,
            observe_me=peer_get_config_response.observe_me,
        )

    def set_peer_config(self, peer: str | Peer, config: SessionPeerConfig) -> None:
        """
        Set the configuration for a peer in this session.
        """
        from .peer import Peer

        self._client.workspaces.sessions.peers.set_config(
            peer_id=str(peer.id) if isinstance(peer, Peer) else peer,
            workspace_id=self.workspace_id,
            session_id=self.id,
            observe_others=omit
            if config.observe_others is None
            else config.observe_others,
            observe_me=omit if config.observe_me is None else config.observe_me,
        )

    @validate_call
    def add_messages(
        self,
        messages: MessageCreateParam | list[MessageCreateParam] = Field(
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
                      - MessageCreateParam: Single MessageCreateParam object
                      - List[MessageCreateParam]: List of MessageCreateParam objects
        """
        if not isinstance(messages, list):
            messages = [messages]

        return self._client.workspaces.sessions.messages.create(
            session_id=self.id,
            workspace_id=self.workspace_id,
            messages=[MessageCreateParam(**message) for message in messages],
        )

    @validate_call
    def get_messages(
        self,
        *,
        filters: dict[str, object] | None = Field(
            None, description="Dictionary of filter criteria"
        ),
    ) -> SyncPage[Message]:
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
        messages_page = self._client.workspaces.sessions.messages.list(
            session_id=self.id,
            workspace_id=self.workspace_id,
            filters=filters,
        )
        return SyncPage(messages_page)

    def get_metadata(self) -> dict[str, object]:
        """
        Get metadata for this session.

        Makes an API call to retrieve the current metadata associated with this session.
        Metadata can include custom attributes, settings, or any other key-value data.
        This method also updates the cached metadata attribute.

        Returns:
            A dictionary containing the session's metadata. Returns an empty dictionary
            if no metadata is set
        """
        session_data = self._client.workspaces.sessions.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        metadata = session_data.metadata or {}
        object.__setattr__(self, "metadata", metadata)
        return metadata

    def delete(self) -> None:
        """
        Delete this session

        Makes an API call to mark this session as inactive.
        """
        self._client.workspaces.sessions.delete(
            session_id=self.id,
            workspace_id=self.workspace_id,
        )

    @validate_call
    def set_metadata(
        self,
        metadata: dict[str, object] = Field(
            ..., description="Metadata dictionary to associate with this session"
        ),
    ) -> None:
        """
        Set metadata for this session.

        Makes an API call to update the metadata associated with this session.
        This will overwrite any existing metadata with the provided values.
        This method also updates the cached metadata attribute.

        Args:
            metadata: A dictionary of metadata to associate with this session.
                     Keys must be strings, values can be any JSON-serializable type
        """
        self._client.workspaces.sessions.update(
            session_id=self.id,
            workspace_id=self.workspace_id,
            metadata=metadata,
        )
        object.__setattr__(self, "metadata", metadata)

    def get_config(self) -> dict[str, object]:
        """
        Get configuration for this session.

        Makes an API call to retrieve the current configuration associated with this session.
        Configuration includes settings that control session behavior.
        This method also updates the cached configuration attribute.

        Returns:
            A dictionary containing the session's configuration. Returns an empty dictionary
            if no configuration is set
        """
        session_data = self._client.workspaces.sessions.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        configuration = session_data.configuration or {}
        object.__setattr__(self, "configuration", configuration)
        return configuration

    @validate_call
    def set_config(
        self,
        configuration: dict[str, object] = Field(
            ..., description="Configuration dictionary to associate with this session"
        ),
    ) -> None:
        """
        Set configuration for this session.

        Makes an API call to update the configuration associated with this session.
        This will overwrite any existing configuration with the provided values.
        This method also updates the cached configuration attribute.

        Args:
            configuration: A dictionary of configuration to associate with this session.
                          Keys must be strings, values can be any JSON-serializable type
        """
        self._client.workspaces.sessions.update(
            session_id=self.id,
            workspace_id=self.workspace_id,
            configuration=configuration,
        )
        object.__setattr__(self, "configuration", configuration)

    def refresh(self) -> None:
        """
        Refresh cached metadata and configuration for this session.

        Makes a single API call to retrieve the latest metadata and configuration
        associated with this session and updates the cached attributes.
        """
        session_data = self._client.workspaces.sessions.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        metadata = session_data.metadata or {}
        configuration = session_data.configuration or {}
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "configuration", configuration)

    @validate_call
    def get_context(
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
        last_user_message: str | Message | None = Field(
            None,
            description="The most recent message (string or Message object), used to fetch semantically relevant observations and returned as part of the context object. Use this alongside `peer_target` to get a more focused context -- does nothing if `peer_target` is not provided.",
        ),
        peer_perspective: str | None = Field(
            None,
            description="A peer ID to get context *from the perspective of*. If given, response will attempt to include representation and card from the perspective of `peer_perspective`. Must be provided with `peer_target`.",
        ),
        limit_to_session: bool = Field(
            False,
            description="Whether to limit the representation to this session only. If True, only observations from this session will be included.",
        ),
        search_top_k: int | None = Field(
            None,
            ge=1,
            le=100,
            description="Number of semantically relevant facts to return when searching with `last_user_message`.",
        ),
        search_max_distance: float | None = Field(
            None,
            ge=0.0,
            le=1.0,
            description="Maximum semantic distance for search results (0.0-1.0) when searching with `last_user_message`.",
        ),
        include_most_derived: bool | None = Field(
            None,
            description="Whether to include the most derived observations in the representation.",
        ),
        max_observations: int | None = Field(
            None,
            ge=1,
            le=100,
            description="Maximum number of observations to include in the representation.",
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
            peer_target: A peer ID to get context for. If given *without* `peer_perspective`, a representation and peer card will be included from the omniscient Honcho-level view of `peer_target`. If given *with* `peer_perspective`, will get the representation and card for `peer_target` *from the perspective of `peer_perspective`*.
            last_user_message: The most recent message (string or Message object), used to fetch semantically relevant observations and returned as part of the context object. Use this alongside `peer_target` to get a more focused context -- does nothing if `peer_target` is not provided.
            peer_perspective: A peer ID to get context *from the perspective of*. If given, response will attempt to include representation and card from the perspective of `peer_perspective`. Must be provided with `peer_target`.
            limit_to_session: Whether to limit the representation to this session only. If True, only observations from this session will be included.
            search_top_k: Number of semantically relevant facts to return when searching with `last_user_message`.
            search_max_distance: Maximum semantic distance for search results (0.0-1.0) when searching with `last_user_message`.
            include_most_derived: Whether to include the most derived observations in the representation.
            max_observations: Maximum number of observations to include in the representation.

        Returns:
            A SessionContext object containing the optimized message history and
            summary, if available, that maximizes conversational context while
            respecting the token limit

        Note:
            Token counting is performed using tiktoken. For models using different
            tokenizers, you may need to adjust the token limit accordingly.
        """

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
            if isinstance(last_user_message, Message)
            else last_user_message
        )
        context = self._client.workspaces.sessions.get_context(
            session_id=self.id,
            workspace_id=self.workspace_id,
            tokens=tokens if tokens is not None else omit,
            summary=summary,
            last_message=last_user_message_id
            if last_user_message_id is not None
            else omit,
            peer_target=peer_target if peer_target is not None else omit,
            peer_perspective=peer_perspective if peer_perspective is not None else omit,
            limit_to_session=limit_to_session,
            search_top_k=search_top_k if search_top_k is not None else omit,
            search_max_distance=search_max_distance
            if search_max_distance is not None
            else omit,
            include_most_derived=include_most_derived
            if include_most_derived is not None
            else omit,
            max_observations=max_observations if max_observations is not None else omit,
        )

        # Convert the honcho_core summary to our Summary if it exists
        session_summary = None
        if context.summary:
            session_summary = Summary(
                content=context.summary.content,
                message_id=context.summary.message_id,
                summary_type=context.summary.summary_type,
                created_at=context.summary.created_at,
                token_count=context.summary.token_count,
            )

        return SessionContext(
            session_id=self.id,
            messages=context.messages,
            summary=session_summary,
            peer_representation=str(context.peer_representation)
            if context.peer_representation
            else None,
            peer_card=context.peer_card,
        )

    def get_summaries(self) -> SessionSummaries:
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
        # Use the honcho_core client to get summaries
        response = self._client.workspaces.sessions.summaries(
            session_id=self.id,
            workspace_id=self.workspace_id,
        )

        # Create Summary objects from the response data
        short_summary = None
        if response.short_summary:
            short_summary = Summary(
                content=response.short_summary.content,
                message_id=response.short_summary.message_id,
                summary_type=response.short_summary.summary_type,
                created_at=response.short_summary.created_at,
                token_count=response.short_summary.token_count,
            )

        long_summary = None
        if response.long_summary:
            long_summary = Summary(
                content=response.long_summary.content,
                message_id=response.long_summary.message_id,
                summary_type=response.long_summary.summary_type,
                created_at=response.long_summary.created_at,
                token_count=response.long_summary.token_count,
            )

        return SessionSummaries(
            id=response.id or self.id,
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
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        return self._client.workspaces.sessions.search(
            self.id,
            workspace_id=self.workspace_id,
            query=query,
            filters=filters,
            limit=limit,
        )

    @validate_call
    def upload_file(
        self,
        file: tuple[str, bytes, str] | tuple[str, Any, str] | Any = Field(
            ...,
            description="File to upload. Can be a file object, (filename, bytes, content_type) tuple, or (filename, fileobj, content_type) tuple.",
        ),
        peer_id: str = Field(..., description="ID of the peer creating the messages"),
    ) -> list[Message]:
        """
        Upload file to create message(s) in this session.

        Accepts a flexible payload:
        - File objects (opened in binary mode)
        - (filename, bytes, content_type) tuples
        - (filename, fileobj, content_type) tuples

        Files are normalized to (filename, fileobj, content_type) tuples for the Stainless client.

        Args:
            file: File to upload. Can be:
                - a file object (must have .name and .read())
                - a tuple (filename, bytes, content_type)
                - a tuple (filename, fileobj, content_type)
            peer_id: ID of the peer who will be attributed as the creator of the messages

        Returns:
            A list of Message objects representing the created messages

        Note:
            Supported file types include PDFs, text files, and JSON documents.
            Large files will be automatically split into multiple messages to fit
            within message size limits.
        """

        # Prepare file for upload using shared utility
        filename, content_bytes, content_type = prepare_file_for_upload(file)

        # Call the upload endpoint
        response = self._client.workspaces.sessions.messages.upload(
            session_id=self.id,
            workspace_id=self.workspace_id,
            file=(filename, content_bytes, content_type),
            peer_id=peer_id,
        )

        return [Message.model_validate(msg) for msg in response]

    def working_rep(
        self,
        peer: str | Peer,
        *,
        target: str | Peer | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_derived: bool | None = None,
        max_observations: int | None = None,
    ) -> "Representation":
        """
        Get the current working representation of the peer in this session.

        Args:
            peer: Peer to get the working representation of.
            target: Optional target peer to get the representation of. If provided,
            queries what `peer` knows about the `target`.
            search_query: Semantic search query to filter relevant observations
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_derived: Whether to include the most derived observations
            max_observations: Maximum number of observations to include

        Returns:
            A Representation object containing explicit and deductive observations

        Example:
            ```python
            # Get peer's representation in this session
            rep = session.working_rep('user123')
            print(rep)

            # Get what user123 knows about assistant in this session
            local_rep = session.working_rep('user123', target='assistant')

            # Get representation with semantic search
            searched_rep = session.working_rep(
                'user123',
                search_query='preferences',
                search_top_k=10
            )
            ```
        """
        from .peer import Peer as _Peer
        from .types import Representation as _Representation

        data = self._client.workspaces.peers.working_representation(
            str(peer.id) if isinstance(peer, _Peer) else peer,
            workspace_id=self.workspace_id,
            session_id=self.id,
            target=str(target.id) if isinstance(target, _Peer) else target,
            search_query=search_query if search_query is not None else omit,
            search_top_k=search_top_k if search_top_k is not None else omit,
            search_max_distance=search_max_distance
            if search_max_distance is not None
            else omit,
            include_most_derived=include_most_derived
            if include_most_derived is not None
            else omit,
            max_observations=max_observations if max_observations is not None else omit,
        )
        return _Representation.from_dict(data)  # type: ignore

    @validate_call
    def get_deriver_status(
        self,
        observer_id: str | None = None,
        sender_id: str | None = None,
    ) -> DeriverStatus:
        """
        Get the deriver processing status, optionally scoped to an observer, sender, and/or session
        """
        return self._client.workspaces.deriver_status(
            workspace_id=self.workspace_id,
            observer_id=observer_id,
            sender_id=sender_id,
            session_id=self.id,
        )

    @validate_call
    def poll_deriver_status(
        self,
        observer_id: str | None = None,
        sender_id: str | None = None,
        timeout: float = Field(
            300.0,
            gt=0,
            description="Maximum time to poll in seconds. Defaults to 5 minutes (300 seconds).",
        ),
    ) -> DeriverStatus:
        """
        Poll get_deriver_status until pending_work_units and in_progress_work_units are both 0.
        This allows you to guarantee that all messages have been processed by the deriver for
        use with the dialectic endpoint.

        The polling estimates sleep time by assuming each work unit takes 1 second.

        Args:
            observer_id: Optional observer ID to scope the status check
            sender_id: Optional sender ID to scope the status check
            timeout: Maximum time to poll in seconds. Defaults to 5 minutes (300 seconds).

        Returns:
            DeriverStatus when all work units are complete

        Raises:
            TimeoutError: If timeout is exceeded before work units complete
            Exception: If get_deriver_status fails repeatedly
        """
        start_time = time.time()

        while True:
            try:
                status = self.get_deriver_status(observer_id, sender_id)
            except Exception as e:
                logger.warning(f"Failed to get deriver status: {e}")
                # Sleep briefly before retrying
                time.sleep(1)

                # Check timeout after error
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise TimeoutError(
                        f"Polling timeout exceeded after {timeout}s. "
                        + f"Error during status check: {e}"
                    ) from e
                continue

            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return status

            # Check timeout before sleeping
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. "
                    + f"Current status: {status.pending_work_units} pending, "
                    + f"{status.in_progress_work_units} in progress work units."
                )

            # Sleep for the expected time to complete all current work units
            # Assuming each pending and in-progress work unit takes 1 second
            total_work_units = status.pending_work_units + status.in_progress_work_units
            sleep_time = max(1, total_work_units)

            # Don't sleep past the timeout
            remaining_time = timeout - elapsed_time
            sleep_time = min(sleep_time, remaining_time)
            if sleep_time <= 0:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. "
                    + f"Current status: {status.pending_work_units} pending, "
                    + f"{status.in_progress_work_units} in progress work units."
                )

            time.sleep(sleep_time)

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
