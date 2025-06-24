import logging
import os
from typing import List, Literal, Mapping, Optional

from honcho_core import AsyncHoncho as AsyncHonchoCore
from honcho_core import Honcho as HonchoCore
from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, Field, PrivateAttr, validate_call

from .pagination import AsyncPage
from .peer import AsyncPeer
from .session import AsyncSession

logger = logging.getLogger(__name__)


class AsyncHoncho(BaseModel):
    """
    Main async client for the Honcho SDK.

    Provides async access to peers, sessions, and workspace operations with configuration
    from environment variables or explicit parameters. This is the primary entry
    point for interacting with the Honcho conversational memory platform asynchronously.

    Attributes:
        api_key: API key for authentication
        base_url: Base URL for the Honcho API
        workspace_id: Workspace ID for scoping operations
    """

    workspace_id: str = Field(
        ...,
        min_length=1,
        description="Workspace ID for scoping operations",
    )
    _client: AsyncHonchoCore = PrivateAttr()

    @validate_call
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[Literal["local", "production", "demo"]] = None,
        base_url: Optional[str] = Field(
            None, description="Base URL for the Honcho API"
        ),
        workspace_id: Optional[str] = Field(
            None, min_length=1, description="Workspace ID for scoping operations"
        ),
        timeout: Optional[float] = Field(None, gt=0, description="Timeout in seconds"),
        max_retries: Optional[int] = Field(
            None, ge=0, description="Maximum number of retries"
        ),
        default_headers: Optional[Mapping[str, str]] = None,
        default_query: Optional[Mapping[str, object]] = None,
    ) -> None:
        """
        Initialize the AsyncHoncho client.

        Args:
            api_key:
                API key for authentication. If not provided, will attempt to
                read from HONCHO_API_KEY environment variable
            environment:
                Environment to use (local or production)
            base_url:
                Base URL for the Honcho API. If not provided, will attempt to
                read from HONCHO_URL environment variable or default to the
                production API URL
            workspace_id:
                Workspace ID to use for operations. If not provided, will
                attempt to read from HONCHO_WORKSPACE_ID environment variable
                or default to "default"
            timeout:
                Optional custom timeout for the HTTP client.
            max_retries:
                Optional custom maximum number of retries for the HTTP client.
            default_headers:
                Optional custom default headers for the HTTP client.
            default_query:
                Optional custom default query parameters for the HTTP client.
        """
        # Resolve workspace_id before calling super().__init__
        resolved_workspace_id = workspace_id or os.getenv(
            "HONCHO_WORKSPACE_ID", "default"
        )

        super().__init__(workspace_id=resolved_workspace_id)

        # Build client kwargs, excluding None values that AsyncHonchoCore doesn't handle well
        client_kwargs = {}

        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if environment is not None:
            client_kwargs["environment"] = environment
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        if default_headers is not None:
            client_kwargs["default_headers"] = default_headers
        if default_query is not None:
            client_kwargs["default_query"] = default_query

        self._client = AsyncHonchoCore(**client_kwargs)

        # Get or create the workspace using synchronous client
        sync_client = HonchoCore(**client_kwargs)
        sync_client.workspaces.get_or_create(id=self.workspace_id)

    @validate_call
    async def peer(
        self,
        id: str = Field(
            ..., min_length=1, description="Unique identifier for the peer"
        ),
        *,
        config: Optional[dict[str, object]] = Field(
            None,
            description="Optional configuration to set for this peer. If set, will get/create peer immediately with flags.",
        ),
    ) -> AsyncPeer:
        """
        Get or create a peer with the given ID.

        Creates an AsyncPeer object that can be used to interact with the specified peer.
        This method does not make an API call - the peer is created lazily when
        its methods are first used.

        Args:
            id: Unique identifier for the peer within the workspace. Should be a
                stable identifier that can be used consistently across sessions
            config:
                Optional configuration to set for this peer. If set, will get/create peer immediately with flags.

        Returns:
            An AsyncPeer object that can be used to send messages, join sessions, and
            query the peer's knowledge representations

        Raises:
            ValidationError: If the peer ID is empty or invalid
        """
        if config:
            return await AsyncPeer.create(
                id, self.workspace_id, self._client, config=config
            )
        else:
            return AsyncPeer(id, self.workspace_id, self._client)

    async def get_peers(self) -> AsyncPage[AsyncPeer]:
        """
        Get all peers in the current workspace.

        Makes an async API call to retrieve all peers that have been created or used
        within the current workspace. Returns a paginated result that transforms
        inner client Peer objects to SDK AsyncPeer objects as they are consumed.

        Returns:
            An AsyncPage of AsyncPeer objects representing all peers in the workspace.
            The page preserves pagination functionality while transforming objects
        """
        peers_page = await self._client.workspaces.peers.list(
            workspace_id=self.workspace_id
        )
        return AsyncPage(
            peers_page, lambda peer: AsyncPeer(peer.id, self.workspace_id, self._client)
        )

    @validate_call
    async def session(
        self,
        id: str = Field(
            ..., min_length=1, description="Unique identifier for the session"
        ),
        *,
        config: Optional[dict[str, object]] = Field(
            None,
            description="Optional configuration to set for this session. If set, will get/create session immediately with flags.",
        ),
    ) -> AsyncSession:
        """
        Get or create a session with the given ID.

        Creates an AsyncSession object that can be used to manage conversations between
        multiple peers. This method does not make an API call - the session is
        created lazily when its methods are first used.

        Args:
            id: Unique identifier for the session within the workspace. Should be a
                stable identifier that can be used consistently to reference the
                same conversation
            config:
                Optional configuration to set for this session. If set, will get/create session immediately with flags.
        Returns:
            An AsyncSession object that can be used to add peers, send messages, and
            manage conversation context

        Raises:
            ValidationError: If the session ID is empty or invalid
        """
        if config:
            return await AsyncSession.create(
                id, self.workspace_id, self._client, config=config
            )
        else:
            return AsyncSession(id, self.workspace_id, self._client)

    async def get_sessions(self) -> AsyncPage[AsyncSession]:
        """
        Get all sessions in the current workspace.

        Makes an async API call to retrieve all sessions that have been created within
        the current workspace.

        Returns:
            An AsyncPage of AsyncSession objects representing all sessions in the workspace.
            Returns an empty page if no sessions exist
        """
        sessions_page = await self._client.workspaces.sessions.list(
            workspace_id=self.workspace_id
        )
        return AsyncPage(
            sessions_page,
            lambda session: AsyncSession(session.id, self.workspace_id, self._client),
        )

    async def get_metadata(self) -> dict[str, object]:
        """
        Get metadata for the current workspace.

        Makes an async API call to retrieve metadata associated with the current workspace.
        Workspace metadata can include settings, configuration, or any other
        key-value data associated with the workspace.

        Returns:
            A dictionary containing the workspace's metadata. Returns an empty
            dictionary if no metadata is set
        """
        workspace = await self._client.workspaces.get_or_create(id=self.workspace_id)
        return workspace.metadata or {}

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(..., description="Metadata dictionary"),
    ) -> None:
        """
        Set metadata for the current workspace.

        Makes an async API call to update the metadata associated with the current workspace.
        This will overwrite any existing metadata with the provided values.

        Args:
            metadata: A dictionary of metadata to associate with the workspace.
                      Keys must be strings, values can be any JSON-serializable type
        """
        await self._client.workspaces.update(self.workspace_id, metadata=metadata)

    async def get_workspaces(self) -> List[str]:
        """
        Get all workspace IDs from the Honcho instance.

        Makes an async API call to retrieve all workspace IDs that the authenticated
        user has access to.

        Returns:
            A list of workspace ID strings. Returns an empty list if no workspaces
            are accessible or none exist
        """
        workspaces_page = await self._client.workspaces.list()
        workspace_ids = []
        async for workspace in workspaces_page:
            workspace_ids.append(workspace.id)
        return workspace_ids

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1, description="The search query to use"),
    ) -> AsyncPage[Message]:
        """
        Search for messages in the current workspace.

        Makes an async API call to search for messages in the current workspace.

        Args:
            query: The search query to use

        Returns:
            An AsyncPage of Message objects representing the search results.
            Returns an empty page if no messages are found.
        """
        messages_page = await self._client.workspaces.search(
            self.workspace_id, body=query
        )
        return AsyncPage(messages_page)

    def __repr__(self) -> str:
        """
        Return a string representation of the AsyncHoncho client.

        Returns:
            A string representation suitable for debugging
        """
        return f"AsyncHoncho(workspace_id='{self.workspace_id}', base_url='{self._client._base_url}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the AsyncHoncho client.

        Returns:
            A string showing the workspace ID
        """
        return f"AsyncHoncho Client (workspace: {self.workspace_id})"
