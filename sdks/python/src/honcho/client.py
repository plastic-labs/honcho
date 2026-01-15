"""Sync Honcho client."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from .api_types import (
    MessageResponse,
    PeerResponse,
    QueueStatusResponse,
    SessionResponse,
    WorkspaceResponse,
)
from .base import PeerBase, SessionBase
from .http import HonchoHTTPClient, routes
from .pagination import SyncPage
from .peer import Peer
from .session import Session

logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENTS = {
    "local": "http://localhost:8000",
    "production": "https://api.honcho.dev",
    "demo": "https://demo.honcho.dev",
}


class Honcho(BaseModel):
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
    _http: HonchoHTTPClient = PrivateAttr()
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
        if http_client is not None:
            http_kwargs["http_client"] = http_client

        self._http = HonchoHTTPClient(**http_kwargs)

        # Get or create the workspace
        self._http.post(routes.workspaces(), body={"id": self.workspace_id})

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
        config: dict[str, object] | None = Field(
            None,
            description="Optional configuration to set for this peer. If set, will get/create peer immediately with flags.",
        ),
    ) -> Peer:
        """
        Get or create a peer with the given ID.

        Creates a Peer object that can be used to interact with the specified peer.
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
            A Peer object that can be used to send messages, join sessions, and
            query the peer's knowledge representations

        Raises:
            ValidationError: If the peer ID is empty or invalid
        """
        # Peer constructor handles API call and caching when metadata/config provided
        return Peer(id, self.workspace_id, self._http, config=config, metadata=metadata)

    def get_peers(
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
        data = self._http.post(
            routes.peers_list(self.workspace_id),
            body={"filters": filters} if filters else None,
        )

        def transform(peer: PeerResponse) -> Peer:
            return Peer(
                peer.id,
                self.workspace_id,
                self._http,
                metadata=peer.metadata,
                config=peer.configuration,
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
        config: dict[str, object] | None = Field(
            None,
            description="Optional configuration to set for this session. If set, will get/create session immediately with flags.",
        ),
    ) -> Session:
        """
        Get or create a session with the given ID.

        Creates a Session object that can be used to manage conversations between
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
            A Session object that can be used to add peers, send messages, and
            manage conversation context

        Raises:
            ValidationError: If the session ID is empty or invalid
        """
        return Session(
            id, self.workspace_id, self._http, config=config, metadata=metadata
        )

    def get_sessions(
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
        data = self._http.post(
            routes.sessions_list(self.workspace_id),
            body={"filters": filters} if filters else None,
        )

        def transform(session: SessionResponse) -> Session:
            return Session(
                session.id,
                self.workspace_id,
                self._http,
                metadata=session.metadata,
                config=session.configuration,
            )

        def fetch_next(page: int) -> SyncPage[SessionResponse, Session]:
            next_data = self._http.post(
                routes.sessions_list(self.workspace_id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return SyncPage(next_data, SessionResponse, transform, fetch_next)

        return SyncPage(data, SessionResponse, transform, fetch_next)

    def get_metadata(self) -> dict[str, object]:
        """
        Get metadata for the current workspace.

        Makes an API call to retrieve metadata associated with the current workspace.
        Workspace metadata can include settings, configuration, or any other
        key-value data associated with the workspace. This method also updates the
        cached metadata attribute.

        Returns:
            A dictionary containing the workspace's metadata. Returns an empty
            dictionary if no metadata is set
        """
        data = self._http.post(routes.workspaces(), body={"id": self.workspace_id})
        workspace = WorkspaceResponse.model_validate(data)
        self._metadata = workspace.metadata or {}
        return self._metadata

    @validate_call
    def set_metadata(
        self,
        metadata: dict[str, object] = Field(..., description="Metadata dictionary"),
    ) -> None:
        """
        Set metadata for the current workspace.

        Makes an API call to update the metadata associated with the current workspace.
        This will overwrite any existing metadata with the provided values.
        This method also updates the cached metadata attribute.

        Args:
            metadata: A dictionary of metadata to associate with the workspace.
                      Keys must be strings, values can be any JSON-serializable type
        """
        self._http.put(
            routes.workspace(self.workspace_id),
            body={"metadata": metadata},
        )
        self._metadata = metadata

    def get_config(self) -> dict[str, object]:
        """
        Get configuration for the current workspace.

        Makes an API call to retrieve configuration associated with the current workspace.
        Configuration includes settings that control workspace behavior.
        This method also updates the cached configuration attribute.

        Returns:
            A dictionary containing the workspace's configuration. Returns an empty
            dictionary if no configuration is set
        """
        data = self._http.post(routes.workspaces(), body={"id": self.workspace_id})
        workspace = WorkspaceResponse.model_validate(data)
        self._configuration = workspace.configuration or {}
        return self._configuration

    @validate_call
    def set_config(
        self,
        configuration: dict[str, object] = Field(
            ..., description="Configuration dictionary"
        ),
    ) -> None:
        """
        Set configuration for the current workspace.

        Makes an API call to update the configuration associated with the current workspace.
        This will overwrite any existing configuration with the provided values.
        This method also updates the cached configuration attribute.

        Args:
            configuration: A dictionary of configuration to associate with the workspace.
                          Keys must be strings, values can be any JSON-serializable type
        """
        self._http.put(
            routes.workspace(self.workspace_id),
            body={"configuration": configuration},
        )
        self._configuration = configuration

    def refresh(self) -> None:
        """
        Refresh cached metadata and configuration for the current workspace.

        Makes a single API call to retrieve the latest metadata and configuration
        associated with the current workspace and updates the cached attributes.
        """
        data = self._http.post(routes.workspaces(), body={"id": self.workspace_id})
        workspace = WorkspaceResponse.model_validate(data)
        self._metadata = workspace.metadata or {}
        self._configuration = workspace.configuration or {}

    def get_workspaces(self, filters: dict[str, object] | None = None) -> list[str]:
        """
        Get all workspace IDs from the Honcho instance.

        Makes an API call to retrieve all workspace IDs that the authenticated
        user has access to.

        Returns:
            A list of workspace ID strings. Returns an empty list if no workspaces
            are accessible or none exist
        """
        data = self._http.post(
            routes.workspaces_list(),
            body={"filters": filters} if filters else None,
        )
        workspace_ids: list[str] = []
        for item in data.get("items", []):
            workspace = WorkspaceResponse.model_validate(item)
            workspace_ids.append(workspace.id)
        return workspace_ids

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
    ) -> list[MessageResponse]:
        """
        Search for messages in the current workspace.

        Makes an API call to search for messages in the current workspace.

        Args:
            query: The search query to use
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v3/guides/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        data = self._http.post(
            routes.workspace_search(self.workspace_id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [MessageResponse.model_validate(item) for item in data]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_queue_status(
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

        data = self._http.get(
            routes.workspace_queue_status(self.workspace_id),
            query=query if query else None,
        )
        return QueueStatusResponse.model_validate(data)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def poll_queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
        timeout: float = Field(
            300.0,
            gt=0,
            description="Maximum time to poll in seconds. Defaults to 5 minutes (300 seconds).",
        ),
    ) -> QueueStatusResponse:
        """
        Poll get_queue_status until pending_work_units and in_progress_work_units are both 0.
        This allows you to guarantee that all messages have been processed by the queue for
        use with the dialectic endpoint.

        The polling estimates sleep time by assuming each work unit takes 1 second.

        Args:
            observer: Optional observer (ID string or Peer object) to scope the status check
            sender: Optional sender (ID string or Peer object) to scope the status check
            session: Optional session (ID string or Session object) to scope the status check
            timeout: Maximum time to poll in seconds. Defaults to 5 minutes (300 seconds).

        Returns:
            QueueStatusResponse when all work units are complete

        Raises:
            TimeoutError: If timeout is exceeded before work units complete
            Exception: If get_queue_status fails repeatedly
        """
        start_time = time.time()

        while True:
            try:
                status = self.get_queue_status(observer, sender, session)
            except Exception as e:
                logger.warning(f"Failed to get queue status: {e}")
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

    @validate_call
    def list_conclusions(
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

        Makes an API call to retrieve conclusions that match the specified filters.
        conclusions can be filtered by session_id, observer_id, and observed_id.

        Args:
            filters: Optional filter criteria for conclusions. Supported filters include:
                    - session_id: Filter conclusions by session
                    - observer_id: Filter conclusions by observer peer
                    - observed_id: Filter conclusions by observed peer
            reverse: Whether to reverse the order of results (default: False)

        Returns:
            A paginated list of conclusion objects matching the specified criteria

        Example:
            >>> conclusions = client.list_conclusions(
            ...     filters={"observer_id": "user123", "observed_id": "assistant"}
            ... )
        """
        from .api_types import ConclusionResponse

        data = self._http.post(
            routes.conclusions_list(self.workspace_id),
            body={"filters": filters, "reverse": reverse},
        )
        return SyncPage(data, ConclusionResponse)

    @validate_call
    def query_conclusions(
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
    ):
        """
        Query conclusions using semantic search.

        Performs vector similarity search on conclusions to find semantically relevant results.
        Observer and observed peer IDs are required for semantic search.

        Args:
            query: The semantic search query
            observer: The observer peer ID (required)
            observed: The observed peer ID (required)
            top_k: Number of results to return (1-100, default: 10)
            distance: Maximum cosine distance threshold for results (0.0-1.0)
            filters: Optional filters to scope the query

        Returns:
            A list of conclusion objects matching the query

        Example:
            >>> conclusions = client.query_conclusions(
            ...     query="user preferences about music",
            ...     observer="user123",
            ...     observed="assistant",
            ...     top_k=5,
            ...     distance=0.8
            ... )
        """
        from .api_types import ConclusionResponse

        # Merge observer/observed into filters without mutating the input
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

        data = self._http.post(
            routes.conclusions_query(self.workspace_id),
            body=body,
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    @validate_call
    def delete_conclusion(
        self,
        conclusion_id: str = Field(
            ..., min_length=1, description="ID of the conclusion to delete"
        ),
    ) -> None:
        """
        Delete a specific conclusion by ID.

        This permanently deletes the conclusion (document) from the theory-of-mind system.
        This action cannot be undone.

        Args:
            conclusion_id: The ID of the conclusion to delete

        Example:
            >>> client.delete_conclusion('obs_123abc')
        """
        self._http.delete(routes.conclusion(self.workspace_id, conclusion_id))

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def update_message(
        self,
        message: MessageResponse | str = Field(
            ..., description="The Message object or message ID to update"
        ),
        metadata: dict[str, object] = Field(
            ..., description="The metadata to update for the message"
        ),
        session: str | SessionBase | None = Field(
            None,
            description="The session (ID string or Session object) - required if message is a string ID",
        ),
    ) -> MessageResponse:
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
        if isinstance(message, MessageResponse):
            message_id = message.id
            resolved_session_id = message.session_id
        else:
            message_id = message
            if not session:
                raise ValueError("session is required when message is a string ID")
            resolved_session_id = session if isinstance(session, str) else session.id

        data = self._http.put(
            routes.message(self.workspace_id, resolved_session_id, message_id),
            body={"metadata": metadata},
        )
        return MessageResponse.model_validate(data)

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
