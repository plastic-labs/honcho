from __future__ import annotations

from typing import TYPE_CHECKING

from honcho_core import AsyncHoncho as AsyncHonchoCore
from honcho_core.types.workspaces.sessions import MessageCreateParam
from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from .pagination import AsyncPage

if TYPE_CHECKING:
    from .session import AsyncSession


class AsyncPeer(BaseModel):
    """
    Represents a peer in the Honcho system with async operations.

    Peers can send messages, participate in sessions, and maintain both global
    and local representations for contextual interactions. A peer represents
    an entity (user, assistant, etc.) that can communicate within the system.

    Attributes:
        id: Unique identifier for this peer
        _client: Reference to the parent AsyncHoncho client instance
    """

    id: str = Field(..., min_length=1, description="Unique identifier for this peer")
    workspace_id: str = Field(
        ..., min_length=1, description="Workspace ID for scoping operations"
    )
    _client: AsyncHonchoCore = PrivateAttr()

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
        client: AsyncHonchoCore = Field(
            ..., description="Reference to the parent AsyncHoncho client instance"
        ),
    ) -> None:
        """
        Initialize a new AsyncPeer.

        Args:
            peer_id: Unique identifier for this peer within the workspace
            workspace_id: Workspace ID for scoping operations
            client: Reference to the parent AsyncHoncho client instance
        """
        super().__init__(id=peer_id, workspace_id=workspace_id)
        self._client = client

    @classmethod
    async def create(
        cls,
        peer_id: str,
        workspace_id: str,
        client: AsyncHonchoCore,
        *,
        config: dict[str, object] | None = None,
    ) -> AsyncPeer:
        """
        Create a new AsyncPeer with optional configuration.

        Args:
            peer_id: Unique identifier for this peer within the workspace
            workspace_id: Workspace ID for scoping operations
            client: Reference to the parent AsyncHoncho client instance
            config: Optional configuration to set for this peer.
                           If set, will get/create peer immediately with flags.

        Returns:
            A new AsyncPeer instance
        """
        peer = cls(peer_id, workspace_id, client)

        if config:
            await client.workspaces.peers.get_or_create(
                workspace_id=workspace_id,
                id=peer_id,
                configuration=config,
            )

        return peer

    async def chat(
        self,
        query: str,
        *,
        stream: bool = False,
        target: str | AsyncPeer | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """
        Query the peer's representation with a natural language question.

        Makes an async API call to the Honcho dialectic endpoint to query either the peer's
        global representation (all content associated with this peer) or their local
        representation of another peer (what this peer knows about the target peer).

        Args:
            query: The natural language question to ask.
            stream: Whether to stream the response
            target: Optional target peer for local representation query. If provided,
                    queries what this peer knows about the target peer rather than
                    querying the peer's global representation
            session_id: Optional session ID to scope the query to a specific session.
                        If provided, only information from that session is considered

        Returns:
            Response string containing the answer to the query, or None if no
            relevant information is available
        """
        response = await self._client.workspaces.peers.chat(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            query=query,
            stream=stream,
            target=str(target.id) if isinstance(target, AsyncPeer) else target,
            session_id=session_id,
        )
        # "If the context provided doesn't help address the query, write absolutely NOTHING but "None""
        if response.content in ("", None, "None"):
            return None
        return response.content

    async def get_sessions(self) -> AsyncPage[AsyncSession]:
        """
        Get all sessions this peer is a member of.

        Makes an async API call to retrieve all sessions where this peer is an active participant.
        Sessions are created when peers are added to them or send messages to them.

        Returns:
            An async paginated list of AsyncSession objects this peer belongs to. Returns an empty
            list if the peer is not a member of any sessions
        """
        from .session import AsyncSession

        sessions_page = await self._client.workspaces.peers.sessions.list(
            peer_id=self.id,
            workspace_id=self.workspace_id,
        )
        return AsyncPage(
            sessions_page,
            lambda session: AsyncSession(session.id, self.workspace_id, self._client),
        )

    @validate_call
    def message(
        self,
        content: str = Field(
            ..., min_length=1, description="The text content for the message"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None, description="Optional metadata dictionary"
        ),
    ) -> MessageCreateParam:
        """
        Create a MessageCreateParam object attributed to this peer.

        This is a convenience method for creating MessageCreateParam objects with this peer's ID.
        The created MessageCreateParam can then be added to sessions or used in other operations.

        Args:
            content: The text content for the message
            metadata: Optional metadata dictionary to associate with the message

        Returns:
            A new MessageCreateParam object with this peer's ID and the provided content
        """
        return MessageCreateParam(peer_id=self.id, content=content, metadata=metadata)

    async def get_metadata(self) -> dict[str, object]:
        """
        Get the current metadata for this peer.

        Makes an async API call to retrieve metadata associated with this peer. Metadata
        can include custom attributes, settings, or any other key-value data
        associated with the peer.

        Returns:
            A dictionary containing the peer's metadata. Returns an empty dictionary
            if no metadata is set
        """
        peer = await self._client.workspaces.peers.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        return peer.metadata or {}

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(
            ..., description="Metadata dictionary to associate with this peer"
        ),
    ) -> None:
        """
        Set the metadata for this peer.

        Makes an async API call to update the metadata associated with this peer.
        This will overwrite any existing metadata with the provided values.

        Args:
            metadata: A dictionary of metadata to associate with this peer.
                      Keys must be strings, values can be any JSON-serializable type
        """
        await self._client.workspaces.peers.update(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            metadata=metadata,
        )

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1, description="The search query to use"),
    ) -> AsyncPage[Message]:
        """
        Search across all messages in the workspace with this peer as author.

        Makes an API call to search endpoint.

        Args:
            query: The search query to use

        Returns:
            An AsyncPage of Message objects representing the search results.
            Returns an empty page if no messages are found.
        """
        messages_page = await self._client.workspaces.peers.search(
            self.id, workspace_id=self.workspace_id, query=query
        )
        return AsyncPage(messages_page)

    def __repr__(self) -> str:
        """
        Return a string representation of the AsyncPeer.

        Returns:
            A string representation suitable for debugging
        """
        return f"AsyncPeer(id='{self.id}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the AsyncPeer.

        Returns:
            The peer's ID
        """
        return self.id
