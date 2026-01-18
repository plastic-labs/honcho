import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Literal

import httpx
from honcho_core import Honcho as HonchoCore
from honcho_core.types.workspaces import QueueStatusResponse
from honcho_core.types.workspaces.peer import Peer as PeerCore
from honcho_core.types.workspaces.session import Session as SessionCore
from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from .base import PeerBase, SessionBase
from .pagination import SyncPage
from .peer import Peer
from .session import Session

logger = logging.getLogger(__name__)


class Honcho(BaseModel):
    """
    Main client for the Honcho SDK.

    Provides access to peers, sessions, and workspace operations with configuration
    from environment variables or explicit parameters. This is the primary entry
    point for interacting with the Honcho conversational memory platform.

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
    _client: HonchoCore = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this workspace. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> dict[str, object] | None:
        """Cached configuration for this workspace. May be stale. Use get_config() for fresh data."""
        return self._configuration

    @property
    def core(self) -> HonchoCore:
        """
        Access the underlying honcho_core client. The honcho_core client is the raw Stainless-generated client,
        allowing users to access functionality that is not exposed through this SDK.

        Returns:
            The underlying HonchoCore client instance

        Example:
            ```python
            from honcho import Honcho

            client = Honcho()

            workspace = client.core.workspaces.get_or_create(id="custom-workspace-id")
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

        # Build client kwargs, excluding None values that HonchoCore doesn't handle well
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
        if http_client is not None:
            client_kwargs["http_client"] = http_client

        self._client = HonchoCore(**client_kwargs)

        # Get or create the workspace
        self._client.workspaces.get_or_create(id=self.workspace_id)

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
        return Peer(
            id, self.workspace_id, self._client, config=config, metadata=metadata
        )

    def get_peers(
        self, filters: dict[str, object] | None = None
    ) -> SyncPage[PeerCore, Peer]:
        """
        Get all peers in the current workspace.

        Makes an API call to retrieve all peers that have been created or used
        within the current workspace. Returns a paginated result that transforms
        inner client Peer objects to SDK Peer objects as they are consumed.

        Returns:
            A SyncPage of Peer objects representing all peers in the workspace
        """
        peers_page = self._client.workspaces.peers.list(
            workspace_id=self.workspace_id, filters=filters
        )
        return SyncPage(
            peers_page,
            lambda peer: Peer(
                peer.id,
                self.workspace_id,
                self._client,
                metadata=peer.metadata,
                config=peer.configuration,
            ),
        )

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
            id, self.workspace_id, self._client, config=config, metadata=metadata
        )

    def get_sessions(
        self, filters: dict[str, object] | None = None
    ) -> SyncPage[SessionCore, Session]:
        """
        Get all sessions in the current workspace.

        Makes an API call to retrieve all sessions that have been created within
        the current workspace.

        Returns:
            A SyncPage of Session objects representing all sessions in the workspace.
            Returns an empty page if no sessions exist
        """
        sessions_page = self._client.workspaces.sessions.list(
            workspace_id=self.workspace_id, filters=filters
        )
        return SyncPage(
            sessions_page,
            lambda session: Session(
                session.id,
                self.workspace_id,
                self._client,
                metadata=session.metadata,
                config=session.configuration,
            ),
        )

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
        workspace = self._client.workspaces.get_or_create(id=self.workspace_id)
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
        self._client.workspaces.update(self.workspace_id, metadata=metadata)
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
        workspace = self._client.workspaces.get_or_create(id=self.workspace_id)
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
        self._client.workspaces.update(self.workspace_id, configuration=configuration)
        self._configuration = configuration

    def refresh(self) -> None:
        """
        Refresh cached metadata and configuration for the current workspace.

        Makes a single API call to retrieve the latest metadata and configuration
        associated with the current workspace and updates the cached attributes.
        """
        workspace = self._client.workspaces.get_or_create(id=self.workspace_id)
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
        workspaces = self._client.workspaces.list(filters=filters)
        return [workspace.id for workspace in workspaces]

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
        self._client.workspaces.delete(workspace_id)

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
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        return self._client.workspaces.search(
            self.workspace_id, query=query, filters=filters, limit=limit
        )

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

        return self._client.workspaces.queue.status(
            workspace_id=self.workspace_id,
            observer_id=resolved_observer_id,
            sender_id=resolved_sender_id,
            session_id=resolved_session_id,
        )

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
        return self._client.workspaces.conclusions.list(
            workspace_id=self.workspace_id,
            filters=filters,
            reverse=reverse,
        )

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
        # Merge observer/observed into filters without mutating the input
        query_filters: dict[str, object | str] = {
            **(filters or {}),
            "observer": observer,
            "observed": observed,
        }

        return self._client.workspaces.conclusions.query(
            workspace_id=self.workspace_id,
            query=query,
            top_k=top_k,
            distance=distance,
            filters=query_filters,
        )

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
        self._client.workspaces.conclusions.delete(
            workspace_id=self.workspace_id,
            conclusion_id=conclusion_id,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def update_message(
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

        return self._client.workspaces.sessions.messages.update(
            message_id=message_id,
            workspace_id=self.workspace_id,
            session_id=resolved_session_id,
            metadata=metadata,
        )

    # Reasoning Artifacts (Read-Only)
    # These methods provide access to reasoning artifacts generated during dreams.
    # Hypotheses, predictions, traces, and inductions are created exclusively by
    # reasoning agents during dream processing and cannot be created via the API.

    def get_hypotheses(
        self,
        *,
        observer: str | None = None,
        observed: str | None = None,
        status: Literal["active", "superseded", "falsified"] | None = None,
        tier: int | None = None,
    ) -> Any:
        """
        Get all hypotheses in the current workspace with optional filters.

        Hypotheses are explanatory theories generated by the Abducer agent during
        reasoning dreams. They represent the system's understanding of observed patterns.

        Args:
            observer: Filter by observer peer name
            observed: Filter by observed peer name
            status: Filter by hypothesis status (active, superseded, falsified)
            tier: Filter by confidence tier (1=high, 2=medium, 3=low)

        Returns:
            Paginated list of hypotheses

        Example:
            ```python
            # Get all active hypotheses for a peer
            hypotheses = client.get_hypotheses(
                observer="user_123",
                observed="user_123",
                status="active"
            )
            ```
        """
        params = {}
        if observer:
            params["observer"] = observer
        if observed:
            params["observed"] = observed
        if status:
            params["status"] = status
        if tier is not None:
            params["tier"] = tier

        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/hypotheses",
            params=params,
        )

    def get_hypothesis(self, hypothesis_id: str) -> Any:
        """
        Get a specific hypothesis by ID.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            Hypothesis details

        Example:
            ```python
            hypothesis = client.get_hypothesis("hyp_abc123")
            print(hypothesis.content)
            print(hypothesis.confidence)
            ```
        """
        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/hypotheses/{hypothesis_id}"
        )

    def get_hypothesis_predictions(self, hypothesis_id: str, *, status: str | None = None) -> Any:
        """
        Get all predictions derived from a specific hypothesis.

        Args:
            hypothesis_id: The hypothesis ID
            status: Optional filter by prediction status

        Returns:
            Paginated list of predictions

        Example:
            ```python
            predictions = client.get_hypothesis_predictions(
                "hyp_abc123",
                status="unfalsified"
            )
            ```
        """
        params = {}
        if status:
            params["status"] = status

        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/hypotheses/{hypothesis_id}/predictions",
            params=params,
        )

    def get_hypothesis_genealogy(self, hypothesis_id: str) -> Any:
        """
        Get the evolution tree for a hypothesis.

        Shows parent hypotheses (what this superseded) and child hypotheses
        (what superseded this one) to track how understanding evolved.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            Genealogy tree with parents, children, and reasoning metadata

        Example:
            ```python
            genealogy = client.get_hypothesis_genealogy("hyp_abc123")
            print(f"Parents: {len(genealogy['parents'])}")
            print(f"Children: {len(genealogy['children'])}")
            ```
        """
        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/hypotheses/{hypothesis_id}/genealogy"
        )

    def get_predictions(
        self,
        *,
        hypothesis_id: str | None = None,
        status: Literal["untested", "unfalsified", "falsified"] | None = None,
        is_blind: bool | None = None,
    ) -> Any:
        """
        Get all predictions in the current workspace with optional filters.

        Predictions are testable claims generated by the Predictor agent from
        hypotheses. They are tested by the Falsifier agent.

        Args:
            hypothesis_id: Filter by source hypothesis
            status: Filter by prediction status
            is_blind: Filter by whether prediction was made blindly

        Returns:
            Paginated list of predictions

        Example:
            ```python
            predictions = client.get_predictions(
                hypothesis_id="hyp_abc123",
                status="unfalsified"
            )
            ```
        """
        params = {}
        if hypothesis_id:
            params["hypothesis_id"] = hypothesis_id
        if status:
            params["status"] = status
        if is_blind is not None:
            params["is_blind"] = is_blind

        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/predictions",
            params=params,
        )

    def get_prediction(self, prediction_id: str) -> Any:
        """
        Get a specific prediction by ID.

        Args:
            prediction_id: The prediction ID

        Returns:
            Prediction details

        Example:
            ```python
            prediction = client.get_prediction("pred_xyz789")
            print(prediction.content)
            print(prediction.status)
            ```
        """
        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/predictions/{prediction_id}"
        )

    def search_predictions(
        self, query: str, *, hypothesis_id: str | None = None
    ) -> Any:
        """
        Semantic search for similar predictions.

        Performs vector similarity search to find predictions semantically similar
        to the query text. Useful for finding related predictions or checking if
        a prediction has already been made.

        Args:
            query: The text to search for semantically similar predictions
            hypothesis_id: Optional filter by source hypothesis

        Returns:
            Paginated list of similar predictions

        Example:
            ```python
            similar = client.search_predictions(
                "prefers dark mode over light mode",
                hypothesis_id="hyp_abc123"
            )
            ```
        """
        body = {"query": query}
        if hypothesis_id:
            body["hypothesis_id"] = hypothesis_id

        return self._client.post(
            f"/v2/workspaces/{self.workspace_id}/predictions/search",
            json=body,
        )

    def get_prediction_traces(self, prediction_id: str) -> Any:
        """
        Get all falsification traces for a specific prediction.

        Args:
            prediction_id: The prediction ID

        Returns:
            Paginated list of falsification traces

        Example:
            ```python
            traces = client.get_prediction_traces("pred_xyz789")
            for trace in traces:
                print(trace.final_status)
            ```
        """
        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/predictions/{prediction_id}/traces"
        )

    def get_traces(
        self,
        *,
        prediction_id: str | None = None,
        final_status: Literal["falsified", "unfalsified"] | None = None,
    ) -> Any:
        """
        Get all falsification traces in the current workspace with optional filters.

        Falsification traces are immutable records of the Falsifier agent's attempts
        to find contradictory evidence for predictions.

        Args:
            prediction_id: Filter by the prediction being tested
            final_status: Filter by final determination (falsified/unfalsified)

        Returns:
            Paginated list of traces

        Example:
            ```python
            traces = client.get_traces(final_status="unfalsified")
            ```
        """
        params = {}
        if prediction_id:
            params["prediction_id"] = prediction_id
        if final_status:
            params["final_status"] = final_status

        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/traces",
            params=params,
        )

    def get_trace(self, trace_id: str) -> Any:
        """
        Get a specific falsification trace by ID.

        Args:
            trace_id: The trace ID

        Returns:
            Trace details including search queries, contradicting premises,
            and reasoning chain

        Example:
            ```python
            trace = client.get_trace("trace_123")
            print(trace.search_queries)
            print(trace.final_status)
            ```
        """
        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/traces/{trace_id}"
        )

    def get_inductions(
        self,
        *,
        observer: str | None = None,
        observed: str | None = None,
        pattern_type: str | None = None,
        confidence: Literal["high", "medium", "low"] | None = None,
    ) -> Any:
        """
        Get all inductions in the current workspace with optional filters.

        Inductions are general patterns extracted by the Inductor agent from
        clusters of unfalsified predictions. They represent stable patterns discovered
        through the reasoning process.

        Args:
            observer: Filter by observer peer name
            observed: Filter by observed peer name
            pattern_type: Filter by pattern type (preference, behavior, personality, etc.)
            confidence: Filter by confidence level (high, medium, low)

        Returns:
            Paginated list of inductions

        Example:
            ```python
            inductions = client.get_inductions(
                observer="user_123",
                observed="user_123",
                confidence="high"
            )
            ```
        """
        params = {}
        if observer:
            params["observer"] = observer
        if observed:
            params["observed"] = observed
        if pattern_type:
            params["pattern_type"] = pattern_type
        if confidence:
            params["confidence"] = confidence

        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/inductions",
            params=params,
        )

    def get_induction(self, induction_id: str) -> Any:
        """
        Get a specific induction by ID.

        Args:
            induction_id: The induction ID

        Returns:
            Induction details including pattern description, type, and confidence

        Example:
            ```python
            induction = client.get_induction("ind_abc123")
            print(induction.content)
            print(induction.pattern_type)
            ```
        """
        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/inductions/{induction_id}"
        )

    def get_induction_sources(self, induction_id: str) -> Any:
        """
        Get the source predictions and premises that formed an induction.

        Provides full transparency into how the pattern was discovered by showing
        the predictions and original observations that led to it.

        Args:
            induction_id: The induction ID

        Returns:
            Dictionary with induction, source_predictions, and source_premises

        Example:
            ```python
            sources = client.get_induction_sources("ind_abc123")
            print(f"Based on {len(sources['source_predictions'])} predictions")
            print(f"From {len(sources['source_premises'])} observations")
            ```
        """
        return self._client.get(
            f"/v2/workspaces/{self.workspace_id}/inductions/{induction_id}/sources"
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the Honcho client.

        Returns:
            A string representation suitable for debugging
        """
        return f"Honcho(workspace_id='{self.workspace_id}', base_url='{self._client.base_url}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Honcho client.

        Returns:
            A string showing the workspace ID
        """
        return f"Honcho Client (workspace: {self.workspace_id})"
