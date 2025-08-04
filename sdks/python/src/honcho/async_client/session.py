from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from honcho_core import AsyncHoncho as AsyncHonchoCore
from honcho_core._types import NOT_GIVEN
from honcho_core.types.workspaces.sessions import MessageCreateParam
from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from ..session_context import SessionContext
from ..utils import prepare_file_for_upload
from .pagination import AsyncPage

if TYPE_CHECKING:
    from .peer import AsyncPeer


try:
    env_val = os.getenv("HONCHO_DEFAULT_CONTEXT_TOKENS")
    _default_context_tokens = int(env_val) if env_val else None
except (ValueError, TypeError):
    _default_context_tokens = None


class SessionPeerConfig(BaseModel):
    observe_others: bool | None = Field(
        None,
        description="Whether this peer should form a session-level theory-of-mind representation of other peers in the session",
    )
    observe_me: bool | None = Field(
        None,
        description="Whether other peers in this session should try to form a session-level theory-of-mind representation of this peer",
    )


class AsyncSession(BaseModel):
    """
    Represents a session in Honcho with async operations.

    Sessions are scoped to a set of peers and contain messages/content.
    They create bidirectional relationships between peers and provide
    a context for multi-party conversations and interactions.

    Attributes:
        id: Unique identifier for this session
        _client: Reference to the parent AsyncHoncho client instance
        anonymous: Whether this is an anonymous session
        summarize: Whether automatic summarization is enabled
    """

    id: str = Field(..., min_length=1, description="Unique identifier for this session")
    workspace_id: str = Field(
        ..., min_length=1, description="Workspace ID for scoping operations"
    )
    _client: AsyncHonchoCore = PrivateAttr()

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        session_id: str = Field(
            ..., min_length=1, description="Unique identifier for this session"
        ),
        workspace_id: str = Field(
            ..., min_length=1, description="Workspace ID for scoping operations"
        ),
        client: AsyncHonchoCore = Field(
            ..., description="Reference to the parent AsyncHoncho client instance"
        ),
    ) -> None:
        """
        Initialize a new AsyncSession.

        Args:
            session_id: Unique identifier for this session within the workspace
            workspace_id: Workspace ID for scoping operations
            client: Reference to the parent AsyncHoncho client instance
        """
        super().__init__(
            id=session_id,
            workspace_id=workspace_id,
        )
        self._client = client

    @classmethod
    async def create(
        cls,
        session_id: str,
        workspace_id: str,
        client: AsyncHonchoCore,
        *,
        config: dict[str, object] | None = None,
    ) -> AsyncSession:
        """
        Create a new AsyncSession with optional configuration.

        Args:
            session_id: Unique identifier for this session within the workspace
            workspace_id: Workspace ID for scoping operations
            client: Reference to the parent AsyncHoncho client instance
            config:
                Optional configuration to set for this session. If set, will get/create session immediately with flags.

        Returns:
            A new AsyncSession instance
        """
        session = cls(session_id, workspace_id, client)

        if config:
            await client.workspaces.sessions.get_or_create(
                workspace_id=workspace_id,
                id=session_id,
                configuration=config,
            )

        return session

    async def add_peers(
        self,
        peers: str
        | AsyncPeer
        | tuple[str, SessionPeerConfig]
        | tuple[AsyncPeer, SessionPeerConfig]
        | list[AsyncPeer | str]
        | list[tuple[AsyncPeer | str, SessionPeerConfig]]
        | list[AsyncPeer | str | tuple[AsyncPeer | str, SessionPeerConfig]] = Field(
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
                   - AsyncPeer: Single AsyncPeer object
                   - List[Union[AsyncPeer, str]]: List of AsyncPeer objects and/or peer IDs
                   - tuple[str, SessionPeerConfig]: Single peer ID and SessionPeerConfig
                   - tuple[AsyncPeer, SessionPeerConfig]: Single AsyncPeer object and SessionPeerConfig
                   - List[tuple[Union[AsyncPeer, str], SessionPeerConfig]]: List of AsyncPeer objects and/or peer IDs and SessionPeerConfig
                   - Mixed lists with peers and tuples/lists containing peer+config combinations
        """
        if not isinstance(peers, list):
            peers = [peers]

        peer_dict: dict[str, Any] = {}
        for peer in peers:
            if isinstance(peer, tuple):
                # Handle tuple[str/AsyncPeer, SessionPeerConfig]
                peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
                peer_config = peer[1]
                peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
            else:
                # Handle direct str or AsyncPeer
                peer_id = peer if isinstance(peer, str) else peer.id
                peer_dict[peer_id] = {}

        await self._client.workspaces.sessions.peers.add(
            session_id=self.id,
            workspace_id=self.workspace_id,
            body=peer_dict,
        )

    async def set_peers(
        self,
        peers: str
        | AsyncPeer
        | tuple[str, SessionPeerConfig]
        | tuple[AsyncPeer, SessionPeerConfig]
        | list[AsyncPeer | str]
        | list[tuple[AsyncPeer | str, SessionPeerConfig]]
        | list[AsyncPeer | str | tuple[AsyncPeer | str, SessionPeerConfig]] = Field(
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
                  - AsyncPeer: Single AsyncPeer object
                  - List[Union[AsyncPeer, str]]: List of AsyncPeer objects and/or peer IDs
                  - tuple[str, SessionPeerConfig]: Single peer ID and SessionPeerConfig
                  - tuple[AsyncPeer, SessionPeerConfig]: Single AsyncPeer object and SessionPeerConfig
                  - List[tuple[Union[AsyncPeer, str], SessionPeerConfig]]: List of AsyncPeer objects and/or peer IDs and SessionPeerConfig
                  - Mixed lists with peers and tuples/lists containing peer+config combinations
        """
        if not isinstance(peers, list):
            peers = [peers]

        peer_dict: dict[str, Any] = {}
        for peer in peers:
            if isinstance(peer, tuple):
                # Handle tuple[str/AsyncPeer, SessionPeerConfig]
                peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
                peer_config = peer[1]
                peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
            else:
                # Handle direct str or AsyncPeer
                peer_id = peer if isinstance(peer, str) else peer.id
                peer_dict[peer_id] = {}

        await self._client.workspaces.sessions.peers.set(
            session_id=self.id,
            workspace_id=self.workspace_id,
            body=peer_dict,
        )

    async def remove_peers(
        self,
        peers: str | AsyncPeer | list[AsyncPeer | str] = Field(
            ..., description="Peers to remove from the session"
        ),
    ) -> None:
        """
        Remove peers from this session.

        Makes an async API call to remove one or more peers from this session.
        Removed peers will no longer be able to participate in the session
        unless added back.

        Args:
            peers: Peers to remove from the session. Can be:
                   - str: Single peer ID
                   - AsyncPeer: Single AsyncPeer object
                   - List[Union[AsyncPeer, str]]: List of AsyncPeer objects and/or peer IDs
        """
        if not isinstance(peers, list):
            peers = [peers]

        peer_ids = [peer if isinstance(peer, str) else peer.id for peer in peers]

        await self._client.workspaces.sessions.peers.remove(
            session_id=self.id,
            workspace_id=self.workspace_id,
            body=peer_ids,
        )

    async def get_peers(self) -> list[AsyncPeer]:
        """
        Get all peers in this session.

        Makes an async API call to retrieve the list of peer IDs that are currently
        members of this session. Automatically converts the paginated response
        into a list for us -- the max number of peers in a session is usually 10.

        Returns:
            A list of AsyncPeer objects that are members of this session
        """
        from .peer import AsyncPeer

        peers_page = await self._client.workspaces.sessions.peers.list(
            session_id=self.id,
            workspace_id=self.workspace_id,
        )
        return [
            AsyncPeer(peer.id, self.workspace_id, self._client)
            for peer in peers_page.items
        ]

    async def get_peer_config(self, peer: str | AsyncPeer) -> SessionPeerConfig:
        """
        Get the configuration for a peer in this session.
        """
        from .peer import AsyncPeer

        peer_get_config_response = (
            await self._client.workspaces.sessions.peers.get_config(
                peer_id=str(peer.id) if isinstance(peer, AsyncPeer) else peer,
                workspace_id=self.workspace_id,
                session_id=self.id,
            )
        )
        return SessionPeerConfig(
            observe_others=peer_get_config_response.observe_others,
            observe_me=peer_get_config_response.observe_me,
        )

    async def set_peer_config(
        self, peer: str | AsyncPeer, config: SessionPeerConfig
    ) -> None:
        """
        Set the configuration for a peer in this session.
        """
        from .peer import AsyncPeer

        await self._client.workspaces.sessions.peers.set_config(
            peer_id=str(peer.id) if isinstance(peer, AsyncPeer) else peer,
            workspace_id=self.workspace_id,
            session_id=self.id,
            observe_others=NOT_GIVEN
            if config.observe_others is None
            else config.observe_others,
            observe_me=NOT_GIVEN if config.observe_me is None else config.observe_me,
        )

    @validate_call
    async def add_messages(
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

        await self._client.workspaces.sessions.messages.create(
            session_id=self.id,
            workspace_id=self.workspace_id,
            messages=[MessageCreateParam(**message) for message in messages],
        )

    @validate_call
    async def get_messages(
        self,
        *,
        filters: dict[str, object] | None = Field(
            None, description="Dictionary of filter criteria"
        ),
    ) -> AsyncPage[Message]:
        """
        Get messages from this session with optional filtering.

        Makes an async API call to retrieve messages from this session. Results can be
        filtered based on various criteria.

        Args:
            filters: Dictionary of filter criteria. Supported filters include:
                    - peer_id: Filter messages by the peer who created them
                    - metadata: Filter messages by metadata key-value pairs
                    - timestamp_start: Filter messages after a specific timestamp
                    - timestamp_end: Filter messages before a specific timestamp

        Returns:
            An async paginated list of Message objects matching the specified criteria, ordered by
            creation time (most recent first)
        """
        messages_page = await self._client.workspaces.sessions.messages.list(
            session_id=self.id,
            workspace_id=self.workspace_id,
            filter=filters,
        )
        return AsyncPage(messages_page)

    async def get_metadata(self) -> dict[str, object]:
        """
        Get metadata for this session.

        Makes an async API call to retrieve the current metadata associated with this session.
        Metadata can include custom attributes, settings, or any other key-value data.

        Returns:
            A dictionary containing the session's metadata. Returns an empty dictionary
            if no metadata is set
        """
        session = await self._client.workspaces.sessions.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        return session.metadata or {}

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(
            ..., description="Metadata dictionary to associate with this session"
        ),
    ) -> None:
        """
        Set metadata for this session.

        Makes an async API call to update the metadata associated with this session.
        This will overwrite any existing metadata with the provided values.

        Args:
            metadata: A dictionary of metadata to associate with this session.
                     Keys must be strings, values can be any JSON-serializable type
        """
        await self._client.workspaces.sessions.update(
            session_id=self.id,
            workspace_id=self.workspace_id,
            metadata=metadata,
        )

    @validate_call
    async def get_context(
        self,
        *,
        summary: bool = True,
        tokens: int | None = Field(
            None, gt=0, description="Maximum number of tokens to include in the context"
        ),
    ) -> SessionContext:
        """
        Get optimized context for this session within a token limit.

        Makes an async API call to retrieve a curated list of messages that provides
        optimal context for the conversation while staying within the specified
        token limit. Uses tiktoken for token counting, so results should be
        compatible with OpenAI models.

        Args:
            summary: Whether to include summary information
            tokens: Maximum number of tokens to include in the context.
                    Defaults to HONCHO_default_context_tokens environment
                    variable if it exists.

        Returns:
            A SessionContext object containing the optimized message history
            that maximizes conversational context while respecting the token limit

        Note:
            Token counting is performed using tiktoken. For models using different
            tokenizers, you may need to adjust the token limit accordingly.
        """
        if not tokens:
            tokens = _default_context_tokens
        context = await self._client.workspaces.sessions.get_context(
            session_id=self.id,
            workspace_id=self.workspace_id,
            tokens=tokens,
            summary=summary,
        )

        return SessionContext(
            session_id=self.id, messages=context.messages, summary=context.summary
        )

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1, description="The search query to use"),
        limit: int = Field(default=10, ge=1, le=100, description="Number of results to return"),
    ) -> list[Message]:
        """
        Search for messages in this session.

        Makes an async API call to search for messages in this session.

        Args:
            query: The search query to use
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        messages = await self._client.workspaces.sessions.search(
            self.id, workspace_id=self.workspace_id, body={"query": query, "limit": limit}
        )
        return messages

    @validate_call
    async def upload_file(
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
        response = await self._client.workspaces.sessions.messages.upload(
            session_id=self.id,
            workspace_id=self.workspace_id,
            file=(filename, content_bytes, content_type),
            peer_id=peer_id,
        )

        return [Message.model_validate(msg) for msg in response]

    async def working_rep(
        self,
        peer: str | AsyncPeer,
        *,
        target: str | AsyncPeer | None = None,
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
        from .peer import AsyncPeer

        return await self._client.workspaces.peers.working_representation(
            str(peer.id) if isinstance(peer, AsyncPeer) else peer,
            workspace_id=self.workspace_id,
            session_id=self.id,
            target=str(target.id) if isinstance(target, AsyncPeer) else target,
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the AsyncSession.

        Returns:
            A string representation suitable for debugging
        """
        return f"AsyncSession(id='{self.id}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the AsyncSession.

        Returns:
            The session's ID
        """
        return self.id
