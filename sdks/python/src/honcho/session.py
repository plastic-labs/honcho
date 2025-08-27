from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honcho_core import Honcho as HonchoCore
from honcho_core._types import NOT_GIVEN
from honcho_core.types.workspaces.sessions import MessageCreateParam
from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from .pagination import SyncPage
from .session_context import SessionContext, SessionSummaries, Summary
from .utils import prepare_file_for_upload

if TYPE_CHECKING:
    from .peer import Peer


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
        _honcho: Reference to the parent Honcho client instance
        anonymous: Whether this is an anonymous session
        summarize: Whether automatic summarization is enabled
    """

    id: str = Field(..., min_length=1, description="Unique identifier for this session")
    workspace_id: str = Field(
        ..., min_length=1, description="Workspace ID for scoping operations"
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
        )
        self._client = client

        if config or metadata:
            self._client.workspaces.sessions.get_or_create(
                workspace_id=workspace_id,
                id=session_id,
                configuration=config if config is not None else NOT_GIVEN,
                metadata=metadata if metadata is not None else NOT_GIVEN,
            )

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
            observe_others=NOT_GIVEN
            if config.observe_others is None
            else config.observe_others,
            observe_me=NOT_GIVEN if config.observe_me is None else config.observe_me,
        )

    @validate_call
    def add_messages(
        self,
        messages: MessageCreateParam | list[MessageCreateParam] = Field(
            ..., description="Messages to add to the session"
        ),
    ) -> None:
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

        self._client.workspaces.sessions.messages.create(
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

    @validate_call
    def set_message_metadata(
        self,
        message_id: str = Field(
            ..., min_length=1, description="ID of the message to update"
        ),
        metadata: dict[str, object] = Field(
            ..., description="Metadata dictionary to associate with the message"
        ),
    ) -> Message:
        """
        Update metadata for a specific message in this session.

        Makes an API call to update the metadata associated with a message.
        This will overwrite any existing metadata with the provided values.

        Args:
            message_id: ID of the message to update
            metadata: A dictionary of metadata to associate with the message.
                     Keys must be strings, values can be any JSON-serializable type

        Returns:
            The updated Message object
        """
        return self._client.workspaces.sessions.messages.update(
            session_id=self.id,
            workspace_id=self.workspace_id,
            message_id=message_id,
            metadata=metadata,
        )

    def get_metadata(self) -> dict[str, object]:
        """
        Get metadata for this session.

        Makes an API call to retrieve the current metadata associated with this session.
        Metadata can include custom attributes, settings, or any other key-value data.

        Returns:
            A dictionary containing the session's metadata. Returns an empty dictionary
            if no metadata is set
        """
        return (
            self._client.workspaces.sessions.get_or_create(
                workspace_id=self.workspace_id,
                id=self.id,
            ).metadata
            or {}
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

        Args:
            metadata: A dictionary of metadata to associate with this session.
                     Keys must be strings, values can be any JSON-serializable type
        """
        self._client.workspaces.sessions.update(
            session_id=self.id,
            workspace_id=self.workspace_id,
            metadata=metadata,
        )

    def delete(self) -> None:
        """
        Delete this session.

        Makes an API call to mark this session as inactive. The session and its
        messages will no longer be accessible through normal operations.
        """
        self._client.workspaces.sessions.delete(
            session_id=self.id,
            workspace_id=self.workspace_id,
        )

    @validate_call
    def get_context(
        self,
        *,
        summary: bool = True,
        tokens: int | None = Field(
            None, gt=0, description="Maximum number of tokens to include in the context"
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

        Returns:
            A SessionContext object containing the optimized message history and
            summary, if available, that maximizes conversational context while
            respecting the token limit

        Note:
            Token counting is performed using tiktoken. For models using different
            tokenizers, you may need to adjust the token limit accordingly.
        """
        context = self._client.workspaces.sessions.get_context(
            session_id=self.id,
            workspace_id=self.workspace_id,
            tokens=tokens if tokens is not None else NOT_GIVEN,
            summary=summary,
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
            session_id=self.id, messages=context.messages, summary=session_summary
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
    ) -> dict[str, object]:
        """
        Get the current working representation of the peer in this session.

        Args:
            peer: Peer to get the working representation of.
            target: Optional target peer to get the representation of. If provided,
            queries what `peer` knows about the `target`.

        Returns:
            A dictionary containing information about the peer.
        """
        from .peer import Peer

        return self._client.workspaces.peers.working_representation(
            str(peer.id) if isinstance(peer, Peer) else peer,
            workspace_id=self.workspace_id,
            session_id=self.id,
            target=str(target.id) if isinstance(target, Peer) else target,
        )

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
