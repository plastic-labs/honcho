"""Sync Honcho client."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from .aio import HonchoAio
from .api_types import (
    MessageResponse,
    PeerConfig,
    PeerResponse,
    QueueStatusResponse,
    SessionConfiguration,
    SessionResponse,
    WorkspaceConfiguration,
    WorkspaceResponse,
)
from .base import PeerBase, SessionBase
from .http import AsyncHonchoHTTPClient, HonchoHTTPClient, routes
from .message import Message
from .mixins import MetadataConfigMixin
from .pagination import SyncPage
from .peer import Peer
from .session import Session
from .utils import resolve_id

logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENTS = {
    "local": "http://localhost:8000",
    "production": "https://api.honcho.dev",
}


class Honcho(BaseModel, MetadataConfigMixin):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """
    Main client for the Honcho SDK.

    Provides access to peers, sessions, and workspace operations with configuration
    from environment variables or explicit parameters. This is the primary entry
    point for interacting with the Honcho conversational memory platform.

    Attributes:
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this workspace. May be stale if not recently
            fetched. Call get_metadata() for fresh data.
        configuration: Cached configuration for this workspace. May be stale if not
            recently fetched. Call get_configuration() for fresh data.
    """

    model_config = ConfigDict(extra="allow")  # pyright: ignore

    workspace_id: str = Field(
        ...,
        min_length=1,
        description="Workspace ID for scoping operations",
    )
    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: WorkspaceConfiguration | None = PrivateAttr(default=None)
    _http: HonchoHTTPClient = PrivateAttr()
    _async_http: AsyncHonchoHTTPClient | None = PrivateAttr(default=None)
    _http_config: dict[str, Any] = PrivateAttr()
    _base_url: str = PrivateAttr()
    _workspace_ensured: bool = PrivateAttr(default=False)

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this workspace. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> WorkspaceConfiguration | None:
        """Cached configuration for this workspace. May be stale. Use get_configuration() for fresh data."""
        return self._configuration

    # MetadataConfigMixin implementation
    def _get_http_client(self):
        return self._http

    def _get_fetch_route(self) -> str:
        return routes.workspaces()

    def _get_update_route(self) -> str:
        return routes.workspace(self.workspace_id)

    def _get_fetch_body(self) -> dict[str, Any]:
        return {"id": self.workspace_id}

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, object], dict[str, object]]:
        workspace = WorkspaceResponse.model_validate(data)
        # Return configuration as dict for mixin compatibility
        return workspace.metadata or {}, workspace.configuration.model_dump(
            exclude_none=True
        )

    def get_configuration(self) -> WorkspaceConfiguration:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Get configuration from the server and update the cache.

        Returns:
            A WorkspaceConfiguration object containing the configuration settings.
        """
        data = self._get_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        workspace = WorkspaceResponse.model_validate(data)
        self._metadata = workspace.metadata or {}
        self._configuration = workspace.configuration
        return self._configuration

    @validate_call
    def set_configuration(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        configuration: WorkspaceConfiguration = Field(
            ..., description="Configuration to set"
        ),
    ) -> None:
        """
        Set configuration on the server and update the cache.

        Args:
            configuration: A WorkspaceConfiguration object with configuration settings.
        """
        self._get_http_client().put(
            self._get_update_route(),
            body={"configuration": configuration.model_dump(exclude_none=True)},
        )
        self._configuration = configuration  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def base_url(self) -> str:
        """The base URL of the Honcho API."""
        return self._base_url

    @property
    def _async_http_client(self) -> AsyncHonchoHTTPClient:
        """Lazily create and return the async HTTP client."""
        if self._async_http is None:
            self._async_http = AsyncHonchoHTTPClient(**self._http_config)
        return self._async_http

    @property
    def aio(self) -> HonchoAio:
        """
        Access async versions of all Honcho methods.

        Returns an HonchoAio view that provides async versions of all methods
        while sharing state with this Honcho instance.

        Example:
            ```python
            honcho = Honcho(workspace_id="my-workspace")

            # Async operations
            peer = await honcho.aio.peer("user-123")
            async for p in honcho.aio.peers():
                print(p.id)
            ```
        """
        return HonchoAio(self)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        api_key: str | None = None,
        environment: Literal["local", "production"] | None = None,
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
        http_client: httpx.Client | None = Field(
            None, description="Custom HTTP client"
        ),
    ) -> None:
        """
        Initialize the Honcho client.

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

        # Resolve API key
        resolved_api_key = api_key or os.getenv("HONCHO_API_KEY")

        # Resolve base URL
        if base_url:
            resolved_base_url = base_url
        elif environment:
            resolved_base_url = ENVIRONMENTS[environment]
        else:
            resolved_base_url = os.getenv("HONCHO_URL", ENVIRONMENTS["production"])

        self._base_url = resolved_base_url

        # Build HTTP client kwargs
        http_kwargs: dict[str, Any] = {
            "base_url": resolved_base_url,
            "api_key": resolved_api_key,
        }

        if timeout is not None:
            http_kwargs["timeout"] = timeout
        if max_retries is not None:
            http_kwargs["max_retries"] = max_retries
        if default_headers is not None:
            http_kwargs["default_headers"] = dict(default_headers)
        if default_query is not None:
            http_kwargs["default_query"] = dict(default_query)

        # Store config for lazy async client creation (without custom http_client)
        self._http_config = dict(http_kwargs)

        if http_client is not None:
            http_kwargs["http_client"] = http_client

        self._http = HonchoHTTPClient(**http_kwargs)

    def _ensure_workspace(self) -> None:
        """
        Ensure the workspace exists on the server.

        The Honcho API uses get-or-create semantics for workspaces via
        `POST /v3/workspaces`. This SDK uses that endpoint once per client
        instance to guarantee that subsequent workspace-scoped calls (peers,
        sessions, queue status, etc.) operate on an existing workspace.
        """
        if self._workspace_ensured:
            return
        self._http.post(routes.workspaces(), body={"id": self.workspace_id})
        self._workspace_ensured = True

    async def _ensure_workspace_async(self) -> None:
        """
        Async version of `_ensure_workspace`.

        This performs the same get-or-create call, but via the async HTTP client
        used by `honcho.aio`.
        """
        if self._workspace_ensured:
            return
        await self._async_http_client.post(
            routes.workspaces(), body={"id": self.workspace_id}
        )
        self._workspace_ensured = True

    @validate_call
    def peer(
        self,
        id: str = Field(
            ..., min_length=1, description="Unique identifier for the peer"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this peer. If set, will get/create peer immediately with metadata.",
        ),
        configuration: PeerConfig | None = Field(
            None,
            description="Optional configuration to set for this peer. If set, will get/create peer immediately with flags.",
        ),
    ) -> Peer:
        """
        Get or create a peer with the given ID.

        Creates a Peer object that can be used to interact with the specified peer.
        This method does not make an API call unless `configuration` or `metadata` is
        provided.

        Args:
            id: Unique identifier for the peer within the workspace. Should be a
                stable identifier that can be used consistently across sessions.
            metadata: Optional metadata dictionary to associate with this peer.
                If set, will get/create peer immediately with metadata.
            configuration: Optional configuration to set for this peer.
                If set, will get/create peer immediately with flags.

        Returns:
            A Peer object that can be used to send messages, join sessions, and
            query the peer's knowledge representations

        Raises:
            ValidationError: If the peer ID is empty or invalid
        """
        return Peer(id, self, configuration=configuration, metadata=metadata)

    def peers(
        self, filters: dict[str, object] | None = None
    ) -> SyncPage[PeerResponse, Peer]:
        """
        Get all peers in the current workspace.

        Makes an API call to retrieve all peers that have been created or used
        within the current workspace. Returns a paginated result that transforms
        inner client Peer objects to SDK Peer objects as they are consumed.

        Returns:
            A SyncPage of Peer objects representing all peers in the workspace
        """
        self._ensure_workspace()
        data = self._http.post(
            routes.peers_list(self.workspace_id),
            body={"filters": filters} if filters else None,
        )

        def transform(peer: PeerResponse) -> Peer:
            return Peer(
                peer.id,
                self,
                metadata=peer.metadata,
                configuration=peer.configuration,
            )

        def fetch_next(page: int) -> SyncPage[PeerResponse, Peer]:
            next_data = self._http.post(
                routes.peers_list(self.workspace_id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return SyncPage(next_data, PeerResponse, transform, fetch_next)

        return SyncPage(data, PeerResponse, transform, fetch_next)

    @validate_call
    def session(
        self,
        id: str = Field(
            ..., min_length=1, description="Unique identifier for the session"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this session. If set, will get/create session immediately with metadata.",
        ),
        configuration: SessionConfiguration | None = Field(
            None,
            description="Optional configuration to set for this session. If set, will get/create session immediately with flags.",
        ),
    ) -> Session:
        """
        Get or create a session with the given ID.

        Creates a Session object that can be used to manage conversations between
        multiple peers. This method does not make an API call unless `configuration` or
        `metadata` is provided.

        Args:
            id: Unique identifier for the session within the workspace. Should be a
                stable identifier that can be used consistently to reference the
                same conversation
            metadata: Optional metadata dictionary to associate with this session.
                If set, will get/create session immediately with metadata.
            configuration: Optional configuration to set for this session.
                If set, will get/create session immediately with flags.

        Returns:
            A Session object that can be used to add peers, send messages, and
            manage conversation context

        Raises:
            ValidationError: If the session ID is empty or invalid
        """
        return Session(id, self, configuration=configuration, metadata=metadata)

    def sessions(
        self, filters: dict[str, object] | None = None
    ) -> SyncPage[SessionResponse, Session]:
        """
        Get all sessions in the current workspace.

        Makes an API call to retrieve all sessions that have been created within
        the current workspace.

        Returns:
            A SyncPage of Session objects representing all sessions in the workspace.
            Returns an empty page if no sessions exist
        """
        self._ensure_workspace()
        data = self._http.post(
            routes.sessions_list(self.workspace_id),
            body={"filters": filters} if filters else None,
        )

        def transform(session: SessionResponse) -> Session:
            return Session(
                session.id,
                self,
                metadata=session.metadata,
                configuration=session.configuration,
            )

        def fetch_next(page: int) -> SyncPage[SessionResponse, Session]:
            next_data = self._http.post(
                routes.sessions_list(self.workspace_id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return SyncPage(next_data, SessionResponse, transform, fetch_next)

        return SyncPage(data, SessionResponse, transform, fetch_next)

    def workspaces(
        self, filters: dict[str, object] | None = None
    ) -> SyncPage[WorkspaceResponse, str]:
        """
        Get all workspace IDs from the Honcho instance.

        Makes an API call to retrieve all workspace IDs that the authenticated
        user has access to.

        Returns:
            A paginated SyncPage of workspace ID strings
        """
        data = self._http.post(
            routes.workspaces_list(),
            body={"filters": filters} if filters else None,
        )

        def transform(workspace: WorkspaceResponse) -> str:
            return workspace.id

        def fetch_next(page: int) -> SyncPage[WorkspaceResponse, str]:
            next_data = self._http.post(
                routes.workspaces_list(),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return SyncPage(next_data, WorkspaceResponse, transform, fetch_next)

        return SyncPage(data, WorkspaceResponse, transform, fetch_next)

    @validate_call
    def delete_workspace(
        self,
        workspace_id: str = Field(
            ..., min_length=1, description="ID of the workspace to delete"
        ),
    ) -> None:
        """
        Delete a workspace.

        Makes an API call to delete the specified workspace. This action cannot be undone.

        Args:
            workspace_id: The ID of the workspace to delete
        """
        self._http.delete(routes.workspace(workspace_id))

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
        Search for messages in the current workspace.

        Makes an API call to search for messages in the current workspace.

        Args:
            query: The search query to use
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        self._ensure_workspace()
        data = self._http.post(
            routes.workspace_search(self.workspace_id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [
            Message.from_api_response(MessageResponse.model_validate(item))
            for item in data
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
    ) -> QueueStatusResponse:
        """
        Get the queue processing status, optionally scoped to an observer, sender, and/or session.

        Args:
            observer: Optional observer (ID string or Peer object) to scope the status check
            sender: Optional sender (ID string or Peer object) to scope the status check
            session: Optional session (ID string or Session object) to scope the status check
        """
        self._ensure_workspace()
        resolved_observer_id = resolve_id(observer)
        resolved_sender_id = resolve_id(sender)
        resolved_session_id = resolve_id(session)

        query: dict[str, Any] = {}
        if resolved_observer_id:
            query["observer_id"] = resolved_observer_id
        if resolved_sender_id:
            query["sender_id"] = resolved_sender_id
        if resolved_session_id:
            query["session_id"] = resolved_session_id

        data = self._http.get(
            routes.workspace_queue_status(self.workspace_id),
            query=query if query else None,
        )
        return QueueStatusResponse.model_validate(data)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def schedule_dream(
        self,
        observer: str | PeerBase,
        session: str | SessionBase | None = None,
        observed: str | PeerBase | None = None,
    ) -> None:
        """
        Schedule a dream task for memory consolidation.

        Dreams are background processes that consolidate observations into higher-level
        insights and update peer cards. This method schedules a dream task for immediate
        processing.

        Args:
            observer: The observer peer (ID string or Peer object) whose perspective
                to use for the dream.
            session: Optional session (ID string or Session object) to scope the dream to.
            observed: Optional observed peer (ID string or Peer object). If not provided,
                defaults to the observer (self-reflection).
        """
        self._ensure_workspace()
        resolved_observer_id = resolve_id(observer)
        resolved_session_id = resolve_id(session)
        resolved_observed_id = (
            resolve_id(observed) if observed else resolved_observer_id
        )

        self._http.post(
            routes.workspace_schedule_dream(self.workspace_id),
            body={
                "observer": resolved_observer_id,
                "observed": resolved_observed_id,
                "session_id": resolved_session_id,
                "dream_type": "omni",
            },
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the Honcho client.

        Returns:
            A string representation suitable for debugging
        """
        return (
            f"Honcho(workspace_id='{self.workspace_id}', base_url='{self._base_url}')"
        )

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Honcho client.

        Returns:
            A string showing the workspace ID
        """
        return f"Honcho Client (workspace: {self.workspace_id})"
