"""Async Honcho client."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from ..api_types import (
    MessageResponse,
    PeerResponse,
    QueueStatusResponse,
    SessionResponse,
    WorkspaceResponse,
)
from ..base import PeerBase, SessionBase
from ..http import AsyncHonchoHTTPClient, routes
from .pagination import AsyncPage
from .peer import AsyncPeer
from .session import AsyncSession

logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENTS = {
    "local": "http://localhost:8000",
    "production": "https://api.honcho.dev",
    "demo": "https://demo.honcho.dev",
}


class AsyncHoncho(BaseModel):
    """
    Main async client for the Honcho SDK.

    Provides async access to peers, sessions, and workspace operations with configuration
    from environment variables or explicit parameters. This is the primary entry
    point for interacting with the Honcho conversational memory platform asynchronously.

    Attributes:
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this workspace. May be stale if not recently
            fetched. Call get_metadata() for fresh data.
        configuration: Cached configuration for this workspace. May be stale if not
            recently fetched. Call get_config() for fresh data.
    """

    model_config = ConfigDict(extra="allow")  # pyright: ignore

    workspace_id: str = Field(
        ...,
        min_length=1,
        description="Workspace ID for scoping operations",
    )
    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: dict[str, object] | None = PrivateAttr(default=None)
    _http: AsyncHonchoHTTPClient = PrivateAttr()
    _base_url: str = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this workspace. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> dict[str, object] | None:
        """Cached configuration for this workspace. May be stale. Use get_config() for fresh data."""
        return self._configuration

    @property
    def base_url(self) -> str:
        """The base URL of the Honcho API."""
        return self._base_url

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
        http_client: httpx.AsyncClient | None = Field(
            None, description="Custom async HTTP client"
        ),
        sync_http_client: httpx.Client | None = Field(
            None, description="Custom sync HTTP client for initialization"
        ),
        # Deprecated parameter for backwards compatibility
        async_http_client: httpx.AsyncClient | None = Field(
            None, description="Deprecated: use http_client instead"
        ),
    ) -> None:
        """
        Initialize the AsyncHoncho client.

        Args:
            api_key:
                API key for authentication. If not provided, will attempt to
                read from HONCHO_API_KEY environment variable
            environment:
                Environment to use (local, production, or demo)
            base_url:
                Base URL for the Honcho API. If not provided, will attempt to
                read from HONCHO_URL environment variable or default to the
                production API URL
            workspace_id:
                Workspace ID to use for operations. If not provided, will
                attempt to read from HONCHO_WORKSPACE_ID environment variable
                or default to "default"
            timeout:
                Optional custom timeout in seconds.
            max_retries:
                Optional custom maximum number of retries.
            default_headers:
                Optional custom default headers.
            default_query:
                Optional custom default query parameters.
            http_client:
                Optional custom httpx.AsyncClient.
            sync_http_client:
                Optional custom httpx.Client for workspace initialization.
            async_http_client:
                Deprecated: use http_client instead.
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

        # Support deprecated async_http_client parameter
        actual_http_client = http_client or async_http_client
        if actual_http_client is not None:
            http_kwargs["http_client"] = actual_http_client

        self._http = AsyncHonchoHTTPClient(**http_kwargs)

        # Get or create the workspace synchronously using a temporary sync client
        # This ensures workspace exists on construction
        from ..http import HonchoHTTPClient

        sync_kwargs = {
            k: v
            for k, v in http_kwargs.items()
            if k != "http_client"  # Don't pass async client to sync client
        }
        if sync_http_client is not None:
            sync_kwargs["http_client"] = sync_http_client

        sync_http = HonchoHTTPClient(**sync_kwargs)
        try:
            sync_http.post(routes.workspaces(), body={"id": self.workspace_id})
        finally:
            sync_http.close()

    @validate_call
    async def peer(
        self,
        id: str = Field(
            ..., min_length=1, description="Unique identifier for the peer"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this peer.",
        ),
        config: dict[str, object] | None = Field(
            None,
            description="Optional configuration to set for this peer.",
        ),
    ) -> AsyncPeer:
        """
        Get or create a peer with the given ID.

        Creates an AsyncPeer object that can be used to interact with the specified peer.
        This method does not make an API call unless `config` or `metadata` is
        provided.

        Args:
            id: Unique identifier for the peer within the workspace.
            metadata: Optional metadata dictionary to associate with this peer.
            config: Optional configuration to set for this peer.

        Returns:
            An AsyncPeer object
        """
        if config or metadata:
            return await AsyncPeer.create(
                id, self.workspace_id, self._http, config=config, metadata=metadata
            )
        return AsyncPeer(id, self.workspace_id, self._http)

    async def get_peers(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[PeerResponse, AsyncPeer]:
        """
        Get all peers in the current workspace.

        Returns:
            An AsyncPage of AsyncPeer objects
        """
        data = await self._http.post(
            routes.peers_list(self.workspace_id),
            body={"filters": filters} if filters else None,
        )

        def transform(peer: PeerResponse) -> AsyncPeer:
            return AsyncPeer(
                peer.id,
                self.workspace_id,
                self._http,
                metadata=peer.metadata,
                config=peer.configuration,
            )

        async def fetch_next(page: int) -> AsyncPage[PeerResponse, AsyncPeer]:
            next_data = await self._http.post(
                routes.peers_list(self.workspace_id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return AsyncPage(next_data, PeerResponse, transform, fetch_next)

        return AsyncPage(data, PeerResponse, transform, fetch_next)

    @validate_call
    async def session(
        self,
        id: str = Field(
            ..., min_length=1, description="Unique identifier for the session"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this session.",
        ),
        config: dict[str, object] | None = Field(
            None,
            description="Optional configuration to set for this session.",
        ),
    ) -> AsyncSession:
        """
        Get or create a session with the given ID.

        Args:
            id: Unique identifier for the session within the workspace.
            metadata: Optional metadata dictionary to associate with this session.
            config: Optional configuration to set for this session.

        Returns:
            An AsyncSession object
        """
        if config or metadata:
            return await AsyncSession.create(
                id, self.workspace_id, self._http, config=config, metadata=metadata
            )
        return AsyncSession(id, self.workspace_id, self._http)

    async def get_sessions(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[SessionResponse, AsyncSession]:
        """
        Get all sessions in the current workspace.

        Returns:
            An AsyncPage of AsyncSession objects
        """
        data = await self._http.post(
            routes.sessions_list(self.workspace_id),
            body={"filters": filters} if filters else None,
        )

        def transform(session: SessionResponse) -> AsyncSession:
            return AsyncSession(
                session.id,
                self.workspace_id,
                self._http,
                metadata=session.metadata,
                config=session.configuration,
            )

        async def fetch_next(page: int) -> AsyncPage[SessionResponse, AsyncSession]:
            next_data = await self._http.post(
                routes.sessions_list(self.workspace_id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return AsyncPage(next_data, SessionResponse, transform, fetch_next)

        return AsyncPage(data, SessionResponse, transform, fetch_next)

    async def get_metadata(self) -> dict[str, object]:
        """
        Get metadata for the current workspace.

        Returns:
            A dictionary containing the workspace's metadata.
        """
        data = await self._http.post(
            routes.workspaces(), body={"id": self.workspace_id}
        )
        workspace = WorkspaceResponse.model_validate(data)
        self._metadata = workspace.metadata or {}
        return self._metadata

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(..., description="Metadata dictionary"),
    ) -> None:
        """
        Set metadata for the current workspace.

        Args:
            metadata: A dictionary of metadata to associate with the workspace.
        """
        await self._http.put(
            routes.workspace(self.workspace_id),
            body={"metadata": metadata},
        )
        self._metadata = metadata

    async def get_config(self) -> dict[str, object]:
        """
        Get configuration for the current workspace.

        Returns:
            A dictionary containing the workspace's configuration.
        """
        data = await self._http.post(
            routes.workspaces(), body={"id": self.workspace_id}
        )
        workspace = WorkspaceResponse.model_validate(data)
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

        Args:
            configuration: A dictionary of configuration to associate with the workspace.
        """
        await self._http.put(
            routes.workspace(self.workspace_id),
            body={"configuration": configuration},
        )
        self._configuration = configuration

    async def refresh(self) -> None:
        """
        Refresh cached metadata and configuration for the current workspace.
        """
        data = await self._http.post(
            routes.workspaces(), body={"id": self.workspace_id}
        )
        workspace = WorkspaceResponse.model_validate(data)
        self._metadata = workspace.metadata or {}
        self._configuration = workspace.configuration or {}

    async def get_workspaces(
        self, filters: dict[str, object] | None = None
    ) -> list[str]:
        """
        Get all workspace IDs from the Honcho instance.

        Returns:
            A list of workspace ID strings.
        """
        data = await self._http.post(
            routes.workspaces_list(),
            body={"filters": filters} if filters else None,
        )
        workspace_ids: list[str] = []
        for item in data.get("items", []):
            workspace = WorkspaceResponse.model_validate(item)
            workspace_ids.append(workspace.id)
        return workspace_ids

    @validate_call
    async def delete_workspace(
        self,
        workspace_id: str = Field(
            ..., min_length=1, description="ID of the workspace to delete"
        ),
    ) -> None:
        """
        Delete a workspace.

        Args:
            workspace_id: The ID of the workspace to delete
        """
        await self._http.delete(routes.workspace(workspace_id))

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
    ) -> list[MessageResponse]:
        """
        Search for messages in the current workspace.

        Args:
            query: The search query to use
            filters: Filters to scope the search.
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of MessageResponse objects.
        """
        data = await self._http.post(
            routes.workspace_search(self.workspace_id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [MessageResponse.model_validate(item) for item in data]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def get_queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
    ) -> QueueStatusResponse:
        """
        Get the queue processing status.

        Args:
            observer: Optional observer (ID string or Peer object)
            sender: Optional sender (ID string or Peer object)
            session: Optional session (ID string or Session object)
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

        query: dict[str, Any] = {}
        if resolved_observer_id:
            query["observer_id"] = resolved_observer_id
        if resolved_sender_id:
            query["sender_id"] = resolved_sender_id
        if resolved_session_id:
            query["session_id"] = resolved_session_id

        data = await self._http.get(
            routes.workspace_queue_status(self.workspace_id),
            query=query if query else None,
        )
        return QueueStatusResponse.model_validate(data)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def poll_queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
        timeout: float = Field(
            300.0,
            gt=0,
            description="Maximum time to poll in seconds. Defaults to 5 minutes.",
        ),
    ) -> QueueStatusResponse:
        """
        Poll get_queue_status until pending_work_units and in_progress_work_units are both 0.

        Args:
            observer: Optional observer (ID string or Peer object)
            sender: Optional sender (ID string or Peer object)
            session: Optional session (ID string or Session object)
            timeout: Maximum time to poll in seconds.

        Returns:
            QueueStatusResponse when all work units are complete

        Raises:
            TimeoutError: If timeout is exceeded before work units complete
        """
        start_time = time.time()

        while True:
            try:
                status = await self.get_queue_status(observer, sender, session)
            except Exception as e:
                logger.warning(f"Failed to get queue status: {e}")
                await asyncio.sleep(1)

                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise TimeoutError(
                        f"Polling timeout exceeded after {timeout}s. Error: {e}"
                    ) from e
                continue

            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return status

            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. Status: {status.pending_work_units} pending, {status.in_progress_work_units} in progress."
                )

            total_work_units = status.pending_work_units + status.in_progress_work_units
            sleep_time = max(1, total_work_units)
            remaining_time = timeout - elapsed_time
            sleep_time = min(sleep_time, remaining_time)

            if sleep_time <= 0:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. Status: {status.pending_work_units} pending, {status.in_progress_work_units} in progress."
                )

            await asyncio.sleep(sleep_time)

    @validate_call
    async def list_conclusions(
        self,
        filters: dict[str, object] | None = Field(
            None, description="Filters to scope the conclusions"
        ),
        reverse: bool = Field(
            False, description="Whether to reverse the order of results"
        ),
    ):
        """
        List all conclusions in the current workspace with optional filtering.

        Args:
            filters: Optional filter criteria for conclusions.
            reverse: Whether to reverse the order of results.

        Returns:
            A paginated list of Conclusion objects.
        """
        from ..api_types import ConclusionResponse

        data = await self._http.post(
            routes.conclusions_list(self.workspace_id),
            body={"filters": filters, "reverse": reverse},
        )
        return AsyncPage(data, ConclusionResponse)

    @validate_call
    async def query_conclusions(
        self,
        query: str = Field(..., min_length=1, description="Semantic search query"),
        observer: str = Field(
            ..., min_length=1, description="Observer peer ID (required)"
        ),
        observed: str = Field(
            ..., min_length=1, description="Observed peer ID (required)"
        ),
        top_k: int = Field(
            default=10, ge=1, le=100, description="Number of results to return"
        ),
        distance: float | None = Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Maximum cosine distance threshold",
        ),
        filters: dict[str, object] | None = Field(
            None, description="Additional filters to apply"
        ),
    ):
        """
        Query conclusions using semantic search.

        Args:
            query: The semantic search query
            observer: The observer peer ID (required)
            observed: The observed peer ID (required)
            top_k: Number of results to return
            distance: Maximum cosine distance threshold
            filters: Optional filters to scope the query

        Returns:
            A list of Conclusion objects.
        """
        from ..api_types import ConclusionResponse

        query_filters: dict[str, object | str] = {
            **(filters or {}),
            "observer": observer,
            "observed": observed,
        }

        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "filters": query_filters,
        }
        if distance is not None:
            body["distance"] = distance

        data = await self._http.post(
            routes.conclusions_query(self.workspace_id),
            body=body,
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    @validate_call
    async def delete_conclusion(
        self,
        conclusion_id: str = Field(
            ..., min_length=1, description="ID of the conclusion to delete"
        ),
    ) -> None:
        """
        Delete a specific conclusion by ID.

        Args:
            conclusion_id: The ID of the conclusion to delete
        """
        await self._http.delete(routes.conclusion(self.workspace_id, conclusion_id))

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def update_message(
        self,
        message: MessageResponse | str = Field(
            ..., description="The Message object or message ID to update"
        ),
        metadata: dict[str, object] = Field(
            ..., description="The metadata to update for the message"
        ),
        session: str | SessionBase | None = Field(
            None,
            description="The session - required if message is a string ID",
        ),
    ) -> MessageResponse:
        """
        Update the metadata of a message.

        Args:
            message: Either a Message object or a message ID string
            metadata: The metadata to update
            session: The session - required if message is a string ID

        Returns:
            The updated Message object

        Raises:
            ValueError: If message is a string ID but session is not provided
        """
        if isinstance(message, MessageResponse):
            message_id = message.id
            resolved_session_id = message.session_id
        else:
            message_id = message
            if not session:
                raise ValueError("session is required when message is a string ID")
            resolved_session_id = session if isinstance(session, str) else session.id

        data = await self._http.put(
            routes.message(self.workspace_id, resolved_session_id, message_id),
            body={"metadata": metadata},
        )
        return MessageResponse.model_validate(data)

    def __repr__(self) -> str:
        return f"AsyncHoncho(workspace_id='{self.workspace_id}', base_url='{self._base_url}')"

    def __str__(self) -> str:
        return f"AsyncHoncho Client (workspace: {self.workspace_id})"
