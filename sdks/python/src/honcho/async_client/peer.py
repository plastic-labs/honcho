from __future__ import annotations

import datetime
from typing import TYPE_CHECKING
from collections.abc import AsyncGenerator

from honcho_core import AsyncHoncho as AsyncHonchoCore
from honcho_core._types import NOT_GIVEN
from honcho_core.types.workspaces.sessions import MessageCreateParam
from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from ..types import DialecticStreamResponse
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
        metadata: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
    ) -> AsyncPeer:
        """
        Create a new AsyncPeer with optional configuration.

        Provided metadata and configuration will overwrite any existing data in those
        locations if given.

        Args:
            peer_id: Unique identifier for this peer within the workspace
            workspace_id: Workspace ID for scoping operations
            client: Reference to the parent AsyncHoncho client instance
            metadata: Optional metadata dictionary to associate with this peer.
            If set, will get/create peer immediately with metadata.
            config: Optional configuration to set for this peer.
            If set, will get/create peer immediately with flags.

        Returns:
            A new AsyncPeer instance
        """
        peer = cls(peer_id, workspace_id, client)

        if config or metadata:
            await client.workspaces.peers.get_or_create(
                workspace_id=workspace_id,
                id=peer_id,
                configuration=config if config is not None else NOT_GIVEN,
                metadata=metadata if metadata is not None else NOT_GIVEN,
            )

        return peer

    async def chat(
        self,
        query: str,
        *,
        stream: bool = False,
        target: str | AsyncPeer | None = None,
        session_id: str | None = None,
    ) -> str | DialecticStreamResponse | None:
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
            For non-streaming: Response string containing the answer, or None if no relevant information
            For streaming: DialecticStreamResponse object that can be iterated over and provides final response
        """
        if stream:

            async def stream_response() -> AsyncGenerator[str]:
                import json

                # Use core SDK with_streaming_response
                async with self._client.workspaces.peers.with_streaming_response.chat(
                    peer_id=self.id,
                    workspace_id=self.workspace_id,
                    query=query,
                    stream=True,
                    target=str(target.id) if isinstance(target, AsyncPeer) else target,
                    session_id=session_id,
                ) as response:
                    async for line in response.iter_lines():
                        if line.startswith("data: "):
                            json_str = line[6:]  # Remove "data: " prefix
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

    async def get_sessions(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[AsyncSession]:
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
            filters=filters,
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
        created_at: datetime.datetime | str | None = Field(
            None,
            description="Optional created-at timestamp for the message. Accepts a datetime which will be converted to an ISO 8601 string, or a preformatted string.",
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
        created_at_str: str | None
        if isinstance(created_at, datetime.datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = created_at

        return MessageCreateParam(
            peer_id=self.id,
            content=content,
            metadata=metadata,
            created_at=created_at_str,
        )

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

    async def get_peer_config(self) -> dict[str, object]:
        """
        Get the current workspace-level configuration for this peer.

        Makes an API call to retrieve configuration associated with this peer.
        Configuration currently includes one optional flag, `observe_me`.

        Returns:
            A dictionary containing the peer's configuration
        """
        peer = await self._client.workspaces.peers.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        return peer.configuration or {}

    @validate_call
    async def set_peer_config(
        self,
        config: dict[str, object] = Field(
            ..., description="Configuration dictionary to associate with this peer"
        ),
    ) -> None:
        """
        Set the configuration for this peer. Currently the only supported config
        value is the `observe_me` flag, which controls whether derivation tasks
        should be created for this peer's global representation. Default is True.

        Makes an API call to update the configuration associated with this peer.
        This will overwrite any existing configuration with the provided values.

        Args:
            config: A dictionary of configuration to associate with this peer.
            Keys must be strings, values can be any JSON-serializable type
        """
        await self._client.workspaces.peers.update(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            configuration=config,
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
    ) -> list[Message]:
        """
        Search across all messages in the workspace with this peer as author.

        Makes an API call to search endpoint.

        Args:
            query: The search query to use
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        return await self._client.workspaces.peers.search(
            self.id,
            workspace_id=self.workspace_id,
            query=query,
            filters=filters,
            limit=limit,
        )

    async def card(
        self,
        target: str | AsyncPeer | None = None,
    ) -> str:
        """
        Get the peer card for this peer.

        Makes an API call to retrieve the peer card, which contains a representation
        of what this peer knows. If a target is provided, returns this peer's local
        representation of the target peer.

        Args:
            target: Optional target peer for local card. If provided, returns this
                    peer's card of the target peer. Can be an AsyncPeer object or peer ID string.

        Returns:
            A PeerCardResponse object containing the peer card
        """
        # Validate target parameter
        if target is not None and not isinstance(target, (str, AsyncPeer)):
            raise TypeError(
                f"target must be str, AsyncPeer, or None, got {type(target)}"
            )

        if isinstance(target, str) and len(target.strip()) == 0:
            raise ValueError("target string cannot be empty")

        response = await self._client.workspaces.peers.card(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            target=str(target.id) if isinstance(target, AsyncPeer) else target,
        )

        if response.peer_card is None:
            return ""

        items: list[str] = response.peer_card
        return "\n".join(items)

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
