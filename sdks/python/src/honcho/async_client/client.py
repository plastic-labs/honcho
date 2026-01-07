import asyncio
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from ..api_types import (
    DeriverStatus,
    Message,
    Workspace,
)
from ..base import PeerBase, SessionBase
from ..http import AsyncHttpClient, AsyncPage
from .peer import AsyncPeer
from .session import AsyncSession

logger = logging.getLogger(__name__)


class _PeersWithStreamingResponseProxy:
    """Proxy object for patching streaming peer endpoints in tests."""

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder chat method intended to be patched in tests."""
        del args, kwargs
        raise RuntimeError("Streaming peer chat proxy should be patched in tests.")


class _PeersProxy:
    """Proxy object for peer endpoints in the low-level client surface."""

    def __init__(self) -> None:
        self.with_streaming_response: _PeersWithStreamingResponseProxy = (
            _PeersWithStreamingResponseProxy()
        )


class _WorkspacesProxy:
    """Proxy object for workspace endpoints in the low-level client surface."""

    def __init__(self) -> None:
        self.peers: _PeersProxy = _PeersProxy()

    async def schedule_dream(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder dream scheduling method intended to be patched in tests."""
        del args, kwargs
        raise RuntimeError("Dream scheduling proxy should be patched in tests.")


class _CoreProxy:
    """Compatibility proxy that mimics the old `.core` client surface for tests."""

    def __init__(self) -> None:
        self.workspaces: _WorkspacesProxy = _WorkspacesProxy()


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

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    workspace_id: str = Field(
        ...,
        min_length=1,
        description="Workspace ID for scoping operations",
    )
    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: dict[str, object] | None = PrivateAttr(default=None)
    _http: AsyncHttpClient = PrivateAttr()
    _core: Any = PrivateAttr(default=None)

    @property
    def core(self) -> Any:
        """Low-level client surface retained for backwards compatibility."""
        return self._core

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
        return self._http.base_url

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
        **kwargs: Any,  # Accept additional kwargs for backwards compatibility
    ) -> None:
        """
        Initialize the AsyncHoncho client.

        Args:
            api_key:
                API key for authentication. If not provided, will attempt to
                read from HONCHO_API_KEY environment variable
            environment:
                Environment to use (local, demo, or production)
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
        """
        # Resolve workspace_id before calling super().__init__
        resolved_workspace_id = workspace_id or os.getenv(
            "HONCHO_WORKSPACE_ID", "default"
        )

        super().__init__(workspace_id=resolved_workspace_id)

        # Resolve API key
        resolved_api_key = api_key or os.getenv("HONCHO_API_KEY")

        # Resolve base URL
        resolved_base_url = base_url or os.getenv("HONCHO_URL")
        if not resolved_base_url:
            if environment == "local":
                resolved_base_url = "http://localhost:8000"
            elif environment == "demo":
                resolved_base_url = "https://demo.honcho.dev"
            else:
                resolved_base_url = "https://api.honcho.dev"

        self._http = AsyncHttpClient(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            timeout=timeout or 60.0,
            max_retries=max_retries or 2,
            default_headers=dict(default_headers) if default_headers else None,
        )
        self._core = _CoreProxy()

        # Note: We can't call async workspace creation in __init__
        # The workspace will be created on first use

    async def _ensure_workspace(self) -> None:
        """Ensure the workspace exists."""
        await self._http.request(
            "POST",
            "/v2/workspaces",
            json={"id": self.workspace_id},
        )

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
            stable identifier that can be used consistently across sessions.
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
        # AsyncPeer constructor handles API call and caching when metadata/config provided
        return await AsyncPeer.create(
            id, self.workspace_id, self._http, config=config, metadata=metadata
        )

    async def get_peers(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[dict[str, Any], AsyncPeer]:
        """
        Get all peers in the current workspace.

        Makes an async API call to retrieve all peers that have been created or used
        within the current workspace.

        Returns:
            An AsyncPage of AsyncPeer objects representing all peers in the workspace
        """

        async def fetch_page(
            page: int = 1, size: int = 50
        ) -> AsyncPage[dict[str, Any], AsyncPeer]:
            response = await self._http.request(
                "POST",
                f"/v2/workspaces/{self.workspace_id}/peers/list",
                json={"filters": filters, "page": page, "size": size},
            )
            return AsyncPage(
                items=response.get("items", []),
                total=response.get("total"),
                page=response.get("page", page),
                size=response.get("size", size),
                pages=response.get("pages"),
                transform_func=lambda p: AsyncPeer(
                    p["id"],
                    self.workspace_id,
                    self._http,
                    metadata=p.get("metadata"),
                    config=p.get("configuration"),
                ),
                fetch_next=lambda: fetch_page(page + 1, size),
            )

        return await fetch_page()

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
        multiple peers.

        Args:
            id: Unique identifier for the session within the workspace.
            metadata: Optional metadata dictionary to associate with this session.
            config: Optional configuration to set for this session.

        Returns:
            An AsyncSession object that can be used to add peers, send messages, and
            manage conversation context
        """
        return await AsyncSession.create(
            id, self.workspace_id, self._http, config=config, metadata=metadata
        )

    async def get_sessions(
        self, filters: dict[str, object] | None = None
    ) -> AsyncPage[dict[str, Any], AsyncSession]:
        """
        Get all sessions in the current workspace.

        Makes an async API call to retrieve all sessions that have been created within
        the current workspace.

        Returns:
            An AsyncPage of AsyncSession objects representing all sessions in the workspace.
        """

        async def fetch_page(
            page: int = 1, size: int = 50
        ) -> AsyncPage[dict[str, Any], AsyncSession]:
            response = await self._http.request(
                "POST",
                f"/v2/workspaces/{self.workspace_id}/sessions/list",
                json={"filters": filters, "page": page, "size": size},
            )
            return AsyncPage(
                items=response.get("items", []),
                total=response.get("total"),
                page=response.get("page", page),
                size=response.get("size", size),
                pages=response.get("pages"),
                transform_func=lambda s: AsyncSession(
                    s["id"],
                    self.workspace_id,
                    self._http,
                    metadata=s.get("metadata"),
                    config=s.get("configuration"),
                ),
                fetch_next=lambda: fetch_page(page + 1, size),
            )

        return await fetch_page()

    async def get_metadata(self) -> dict[str, object]:
        """Get metadata for the current workspace."""
        response = await self._http.request(
            "POST",
            "/v2/workspaces",
            json={"id": self.workspace_id},
        )
        workspace = Workspace.model_validate(response)
        self._metadata = workspace.metadata or {}
        return self._metadata

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(..., description="Metadata dictionary"),
    ) -> None:
        """Set metadata for the current workspace."""
        await self._http.request(
            "PUT",
            f"/v2/workspaces/{self.workspace_id}",
            json={"metadata": metadata},
        )
        self._metadata = metadata

    async def get_config(self) -> dict[str, object]:
        """Get configuration for the current workspace."""
        response = await self._http.request(
            "POST",
            "/v2/workspaces",
            json={"id": self.workspace_id},
        )
        workspace = Workspace.model_validate(response)
        self._configuration = workspace.configuration or {}
        return self._configuration

    @validate_call
    async def set_config(
        self,
        configuration: dict[str, object] = Field(
            ..., description="Configuration dictionary"
        ),
    ) -> None:
        """Set configuration for the current workspace."""
        await self._http.request(
            "PUT",
            f"/v2/workspaces/{self.workspace_id}",
            json={"configuration": configuration},
        )
        self._configuration = configuration

    async def refresh(self) -> None:
        """Refresh cached metadata and configuration for the current workspace."""
        response = await self._http.request(
            "POST",
            "/v2/workspaces",
            json={"id": self.workspace_id},
        )
        workspace = Workspace.model_validate(response)
        self._metadata = workspace.metadata or {}
        self._configuration = workspace.configuration or {}

    async def get_workspaces(
        self, filters: dict[str, object] | None = None
    ) -> list[str]:
        """Get all workspace IDs from the Honcho instance."""
        response = await self._http.request(
            "POST",
            "/v2/workspaces/list",
            json={"filters": filters},
        )
        response_data = cast(Mapping[str, Any], response or {})
        items_raw = response_data.get("items", [])
        items = (
            cast(list[Mapping[str, Any]], items_raw)
            if isinstance(items_raw, list)
            else []
        )
        workspace_ids: list[str] = []
        for workspace in items:
            workspace_id = workspace.get("id")
            if isinstance(workspace_id, str):
                workspace_ids.append(workspace_id)
        return workspace_ids

    @validate_call
    async def delete_workspace(
        self,
        workspace_id: str = Field(
            ..., min_length=1, description="ID of the workspace to delete"
        ),
    ) -> Workspace:
        """Delete a workspace."""
        response = await self._http.request(
            "DELETE",
            f"/v2/workspaces/{workspace_id}",
        )
        return Workspace.model_validate(response)

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
        """Search for messages in the current workspace."""
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/search",
            json={"query": query, "filters": filters, "limit": limit},
        )
        messages_raw = cast(list[Any], response or [])
        return [Message.model_validate(m) for m in messages_raw]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def get_deriver_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
    ) -> DeriverStatus:
        """Get the deriver processing status."""
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

        params: dict[str, Any] = {}
        if resolved_observer_id:
            params["observer_id"] = resolved_observer_id
        if resolved_sender_id:
            params["sender_id"] = resolved_sender_id
        if resolved_session_id:
            params["session_id"] = resolved_session_id

        response = await self._http.request(
            "GET",
            f"/v2/workspaces/{self.workspace_id}/queue/status",
            params=params,
        )
        return DeriverStatus.model_validate(response)

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
        """Poll get_deriver_status until work is complete."""
        start_time = time.time()

        while True:
            try:
                status = await self.get_deriver_status(observer, sender, session)
            except Exception as e:
                logger.warning(f"Failed to get deriver status: {e}")
                await asyncio.sleep(1)
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise TimeoutError(
                        f"Polling timeout exceeded after {timeout}s. "
                        + f"Error during status check: {e}"
                    ) from e
                continue

            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return status

            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. "
                    + f"Current status: {status.pending_work_units} pending, "
                    + f"{status.in_progress_work_units} in progress work units."
                )

            total_work_units = status.pending_work_units + status.in_progress_work_units
            sleep_time = max(1, total_work_units)
            remaining_time = timeout - elapsed_time
            sleep_time = min(sleep_time, remaining_time)
            if sleep_time <= 0:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. "
                    + f"Current status: {status.pending_work_units} pending, "
                    + f"{status.in_progress_work_units} in progress work units."
                )

            await asyncio.sleep(sleep_time)

    @validate_call
    async def list_observations(
        self,
        filters: dict[str, object] | None = Field(
            None, description="Filters to scope the observations"
        ),
        reverse: bool = Field(
            False, description="Whether to reverse the order of results"
        ),
    ) -> list[dict[str, Any]]:
        """List all observations in the current workspace with optional filtering."""
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/observations/list",
            json={"filters": filters, "reverse": reverse},
        )
        response_data = cast(Mapping[str, Any], response or {})
        items_raw = response_data.get("items", [])
        if not isinstance(items_raw, list):
            return []
        items = cast(list[object], items_raw)
        observations: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                observations.append(cast(dict[str, Any], item))
        return observations

    @validate_call
    async def query_observations(
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
            description="Maximum cosine distance threshold for results",
        ),
        filters: dict[str, object] | None = Field(
            None, description="Additional filters to apply"
        ),
    ) -> list[dict[str, Any]]:
        """Query observations using semantic search."""
        query_filters: dict[str, object | str] = {
            **(filters or {}),
            "observer": observer,
            "observed": observed,
        }

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/observations/query",
            json={
                "query": query,
                "top_k": top_k,
                "distance": distance,
                "filters": query_filters,
            },
        )
        if not isinstance(response, list):
            return []
        results = cast(list[object], response)
        observations: list[dict[str, Any]] = []
        for item in results:
            if isinstance(item, dict):
                observations.append(cast(dict[str, Any], item))
        return observations

    @validate_call
    async def delete_observation(
        self,
        observation_id: str = Field(
            ..., min_length=1, description="ID of the observation to delete"
        ),
    ) -> None:
        """Delete a specific observation by ID."""
        await self._http.request(
            "DELETE",
            f"/v2/workspaces/{self.workspace_id}/observations/{observation_id}",
        )

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
        """Update the metadata of a message."""
        if isinstance(message, Message):
            message_id = message.id
            resolved_session_id = message.session_id
        else:
            message_id = message
            if not session:
                raise ValueError("session is required when message is a string ID")
            resolved_session_id = session if isinstance(session, str) else session.id

        response = await self._http.request(
            "PUT",
            f"/v2/workspaces/{self.workspace_id}/sessions/{resolved_session_id}/messages/{message_id}",
            json={"metadata": metadata},
        )
        return Message.model_validate(response)

    def __repr__(self) -> str:
        return f"AsyncHoncho(workspace_id='{self.workspace_id}', base_url='{self._http.base_url}')"

    def __str__(self) -> str:
        return f"AsyncHoncho Client (workspace: {self.workspace_id})"
