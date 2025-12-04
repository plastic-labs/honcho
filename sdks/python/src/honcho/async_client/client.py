import asyncio
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Literal

import httpx
from honcho_core import AsyncHoncho as AsyncHonchoCore
from honcho_core import Honcho as HonchoCore
from honcho_core.types import DeriverStatus, Workspace
from honcho_core.types.workspaces.peer import Peer as PeerCore
from honcho_core.types.workspaces.session import Session as SessionCore
from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from ..base import PeerBase, SessionBase
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

    For advanced usage, the underlying honcho_core client can be accessed via the
    `core` property to use functionality not exposed through this SDK.

    Attributes:
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this workspace. May be stale if not recently
            fetched. Call get_metadata() for fresh data.
        configuration: Cached configuration for this workspace. May be stale if not
            recently fetched. Call get_config() for fresh data.
        core: Access to the underlying honcho_core client for advanced usage
    """

    model_config = ConfigDict(extra="allow")  # pyright: ignore

    workspace_id: str = Field(
        ...,
        min_length=1,
        description="Workspace ID for scoping operations",
    )
    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: dict[str, object] | None = PrivateAttr(default=None)
    _client: AsyncHonchoCore = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this workspace. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> dict[str, object] | None:
        """Cached configuration for this workspace. May be stale. Use get_config() for fresh data."""
        return self._configuration

    @property
    def core(self) -> AsyncHonchoCore:
        """
        Access the underlying honcho_core client. The honcho_core client is the raw Stainless-generated client,
        allowing users to access functionality that is not exposed through this SDK.

        Returns:
            The underlying AsyncHonchoCore client instance

        Example:
            ```python
            from honcho import AsyncHoncho

            client = AsyncHoncho()

            workspace = await client.core.workspaces.get_or_create(id="custom-workspace-id")
            ```
        """
        return self._client

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        api_key: str | None = None,
        environment: Literal["local", "production", "demo"] | None = None,
        base_url: str | None = Field(None, description="Base URL for the Honcho API"),
        workspace_id: str | None = Field(
            None, min_length=1, description="Workspace ID for scoping operations"
        ),
        timeout: float | None = Field(None, gt=0, description="Timeout in seconds"),
        max_retries: int | None = Field(
            None, ge=0, description="Maximum number of retries"
        ),
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        async_http_client: httpx.AsyncClient | None = Field(
            None, description="Custom HTTP client"
        ),
        http_client: httpx.Client | None = Field(
            None, description="Custom HTTP client"
        ),
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
            http_client:
                Optional custom httpx client.
        """
        # Resolve workspace_id before calling super().__init__
        resolved_workspace_id = workspace_id or os.getenv(
            "HONCHO_WORKSPACE_ID", "default"
        )

        super().__init__(workspace_id=resolved_workspace_id)

        # Build client kwargs, excluding None values that AsyncHonchoCore doesn't handle well
        client_kwargs: dict[str, Any] = {}

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

        sync_client_kwargs = client_kwargs.copy()
        async_client_kwargs = client_kwargs.copy()

        if http_client is not None:
            sync_client_kwargs["http_client"] = http_client
        if async_http_client is not None:
            async_client_kwargs["http_client"] = async_http_client

        self._client = AsyncHonchoCore(**async_client_kwargs)

        # Get or create the workspace using synchronous client
        sync_client = HonchoCore(**sync_client_kwargs)
        sync_client.workspaces.get_or_create(id=self.workspace_id)

    @validate_call
    async def peer(
        self,
        id: str = Field(
            ..., min_length=1, description="Unique identifier for the peer"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this peer. If set, will get/create peer immediately with metadata.",
        ),
        config: dict[str, object] | None = Field(
            None,
            description="Optional configuration to set for this peer. If set, will get/create peer immediately with flags.",
        ),
    ) -> AsyncPeer:
        """
        Get or create a peer with the given ID.

        Creates an AsyncPeer object that can be used to interact with the specified peer.
        This method does not make an API call unless `config` or `metadata` is
        provided.

        Args:
            id: Unique identifier for the peer within the workspace. Should be a
            stable identifier that can be used consistently across sessions
            metadata: Optional metadata dictionary to associate with this peer.
            If set, will get/create peer immediately with metadata.
            config: Optional configuration to set for this peer.
            If set, will get/create peer immediately with flags.

        Returns:
            An AsyncPeer object that can be used to send messages, join sessions, and
            query the peer's knowledge representations

        Raises:
            ValidationError: If the peer ID is empty or invalid
        """
        if config or metadata:
            return await AsyncPeer.create(
                id, self.workspace_id, self._client, config=config, metadata=metadata
            )
        return AsyncPeer(id, self.workspace_id, self._client)

    async def get_peers(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[PeerCore, AsyncPeer]:
        """
        Get all peers in the current workspace.

        Makes an async API call to retrieve all peers that have been created or used
        within the current workspace. Returns a paginated result that transforms
        inner client Peer objects to SDK AsyncPeer objects as they are consumed.

        Returns:
            An AsyncPage of AsyncPeer objects representing all peers in the workspace
        """
        peers_page = await self._client.workspaces.peers.list(
            workspace_id=self.workspace_id, filters=filters
        )
        return AsyncPage(
            peers_page,
            lambda peer: AsyncPeer(
                peer.id,
                self.workspace_id,
                self._client,
                metadata=peer.metadata,
                config=peer.configuration,
            ),
        )

    @validate_call
    async def session(
        self,
        id: str = Field(
            ..., min_length=1, description="Unique identifier for the session"
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
    ) -> AsyncSession:
        """
        Get or create a session with the given ID.

        Creates an AsyncSession object that can be used to manage conversations between
        multiple peers. This method does not make an API call unless `config` or
        `metadata` is provided.

        Args:
            id: Unique identifier for the session within the workspace. Should be a
            stable identifier that can be used consistently to reference the
            same conversation
            metadata: Optional metadata dictionary to associate with this session.
            If set, will get/create session immediately with metadata.
            config: Optional configuration to set for this session.
            If set, will get/create session immediately with flags.
        Returns:
            An AsyncSession object that can be used to add peers, send messages, and
            manage conversation context

        Raises:
            ValidationError: If the session ID is empty or invalid
        """
        if config or metadata:
            return await AsyncSession.create(
                id, self.workspace_id, self._client, config=config, metadata=metadata
            )
        return AsyncSession(id, self.workspace_id, self._client)

    async def get_sessions(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[SessionCore, AsyncSession]:
        """
        Get all sessions in the current workspace.

        Makes an async API call to retrieve all sessions that have been created within
        the current workspace.

        Returns:
            An AsyncPage of AsyncSession objects representing all sessions in the workspace.
            Returns an empty page if no sessions exist
        """
        sessions_page = await self._client.workspaces.sessions.list(
            workspace_id=self.workspace_id, filters=filters
        )
        return AsyncPage(
            sessions_page,
            lambda session: AsyncSession(
                session.id,
                self.workspace_id,
                self._client,
                metadata=session.metadata,
                config=session.configuration,
            ),
        )

    async def get_metadata(self) -> dict[str, object]:
        """
        Get metadata for the current workspace.

        Makes an async API call to retrieve metadata associated with the current workspace.
        Workspace metadata can include settings, configuration, or any other
        key-value data associated with the workspace. This method also updates the
        cached metadata attribute.

        Returns:
            A dictionary containing the workspace's metadata. Returns an empty
            dictionary if no metadata is set
        """
        workspace = await self._client.workspaces.get_or_create(id=self.workspace_id)
        self._metadata = workspace.metadata or {}
        return self._metadata

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(..., description="Metadata dictionary"),
    ) -> None:
        """
        Set metadata for the current workspace.

        Makes an async API call to update the metadata associated with the current workspace.
        This will overwrite any existing metadata with the provided values.
        This method also updates the cached metadata attribute.

        Args:
            metadata: A dictionary of metadata to associate with the workspace.
                      Keys must be strings, values can be any JSON-serializable type
        """
        await self._client.workspaces.update(self.workspace_id, metadata=metadata)
        self._metadata = metadata

    async def get_config(self) -> dict[str, object]:
        """
        Get configuration for the current workspace.

        Makes an async API call to retrieve configuration associated with the current workspace.
        Configuration includes settings that control workspace behavior.
        This method also updates the cached configuration attribute.

        Returns:
            A dictionary containing the workspace's configuration. Returns an empty
            dictionary if no configuration is set
        """
        workspace = await self._client.workspaces.get_or_create(id=self.workspace_id)
        self._configuration = workspace.configuration or {}
        return self._configuration

    @validate_call
    async def set_config(
        self,
        configuration: dict[str, object] = Field(
            ..., description="Configuration dictionary"
        ),
    ) -> None:
        """
        Set configuration for the current workspace.

        Makes an async API call to update the configuration associated with the current workspace.
        This will overwrite any existing configuration with the provided values.
        This method also updates the cached configuration attribute.

        Args:
            configuration: A dictionary of configuration to associate with the workspace.
                          Keys must be strings, values can be any JSON-serializable type
        """
        await self._client.workspaces.update(
            self.workspace_id, configuration=configuration
        )
        self._configuration = configuration

    async def refresh(self) -> None:
        """
        Refresh cached metadata and configuration for the current workspace.

        Makes a single async API call to retrieve the latest metadata and configuration
        associated with the current workspace and updates the cached attributes.
        """
        workspace = await self._client.workspaces.get_or_create(id=self.workspace_id)
        self._metadata = workspace.metadata or {}
        self._configuration = workspace.configuration or {}

    async def get_workspaces(
        self, filters: dict[str, object] | None = None
    ) -> list[str]:
        """
        Get all workspace IDs from the Honcho instance.

        Makes an async API call to retrieve all workspace IDs that the authenticated
        user has access to.

        Returns:
            A list of workspace ID strings. Returns an empty list if no workspaces
            are accessible or none exist
        """
        workspaces_page = await self._client.workspaces.list(filters=filters)
        workspace_ids: list[str] = []
        async for workspace in workspaces_page:
            workspace_ids.append(workspace.id)
        return workspace_ids

    @validate_call
    async def delete_workspace(
        self,
        workspace_id: str = Field(
            ..., min_length=1, description="ID of the workspace to delete"
        ),
    ) -> Workspace:
        """
        Delete a workspace.

        Makes an async API call to delete the specified workspace.

        Args:
            workspace_id: The ID of the workspace to delete

        Returns:
            The deleted Workspace object
        """
        return await self._client.workspaces.delete(workspace_id)

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
        Search for messages in the current workspace.

        Makes an async API call to search for messages in the current workspace.

        Args:
            query: The search query to use
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        return await self._client.workspaces.search(
            self.workspace_id,
            query=query,
            filters=filters,
            limit=limit,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def get_deriver_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
    ) -> DeriverStatus:
        """
        Get the deriver processing status, optionally scoped to an observer, sender, and/or session.

        Args:
            observer: Optional observer (ID string or Peer object) to scope the status check
            sender: Optional sender (ID string or Peer object) to scope the status check
            session: Optional session (ID string or Session object) to scope the status check
        """
        resolved_observer_id = (
            None
            if observer is None
            else (observer if isinstance(observer, str) else observer.id)
        )
        resolved_sender_id = (
            None
            if sender is None
            else (sender if isinstance(sender, str) else sender.id)
        )
        resolved_session_id = (
            None
            if session is None
            else (session if isinstance(session, str) else session.id)
        )

        return await self._client.workspaces.deriver_status(
            workspace_id=self.workspace_id,
            observer_id=resolved_observer_id,
            sender_id=resolved_sender_id,
            session_id=resolved_session_id,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def poll_deriver_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
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
            observer: Optional observer (ID string or AsyncPeer object) to scope the status check
            sender: Optional sender (ID string or AsyncPeer object) to scope the status check
            session: Optional session (ID string or AsyncSession object) to scope the status check
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
                status = await self.get_deriver_status(observer, sender, session)
            except Exception as e:
                logger.warning(f"Failed to get deriver status: {e}")
                # Sleep briefly before retrying
                await asyncio.sleep(1)

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

            await asyncio.sleep(sleep_time)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def update_message(
        self,
        message: Message | str = Field(
            ..., description="The Message object or message ID to update"
        ),
        metadata: dict[str, object] = Field(
            ..., description="The metadata to update for the message"
        ),
        session: str | SessionBase | None = Field(
            None,
            description="The session (ID string or Session object) - required if message is a string ID",
        ),
    ) -> Message:
        """
        Update the metadata of a message.

        Makes an API call to update the metadata of a specific message within a session.

        Args:
            message: Either a Message object or a message ID string
            metadata: The metadata to update for the message
            session: The session (ID string or Session object) - required if message is a string ID, ignored if message is a Message object

        Returns:
            The updated Message object

        Raises:
            ValidationError: If message is a string ID but session_id is not provided
        """
        if isinstance(message, Message):
            message_id = message.id
            resolved_session_id = message.session_id
        else:
            message_id = message
            if not session:
                raise ValueError("session is required when message is a string ID")
            resolved_session_id = session if isinstance(session, str) else session.id

        return await self._client.workspaces.sessions.messages.update(
            message_id=message_id,
            workspace_id=self.workspace_id,
            session_id=resolved_session_id,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the AsyncHoncho client.

        Returns:
            A string representation suitable for debugging
        """
        return f"AsyncHoncho(workspace_id='{self.workspace_id}', base_url='{self._client.base_url}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the AsyncHoncho client.

        Returns:
            A string showing the workspace ID
        """
        return f"AsyncHoncho Client (workspace: {self.workspace_id})"
