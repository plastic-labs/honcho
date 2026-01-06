"""Observation types and scoped access for the Honcho SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .base import SessionBase
from .http import AsyncHttpClient, HttpClient

if TYPE_CHECKING:
    from .types import ObservationCreateParam, Representation


def _convert_observation(item: Any) -> dict[str, Any]:
    """Convert a core SDK Observations model to a dict for our Observation class."""
    if hasattr(item, "model_dump"):
        # Pydantic model - use model_dump()
        return item.model_dump()  # type: ignore[no-any-return]
    elif isinstance(item, dict):
        return cast(dict[str, Any], item)
    else:
        # Fallback: access as object attributes
        return {
            "id": getattr(item, "id", ""),
            "content": getattr(item, "content", ""),
            "observer_id": getattr(item, "observer_id", ""),
            "observed_id": getattr(item, "observed_id", ""),
            "session_id": getattr(item, "session_id", ""),
            "created_at": str(getattr(item, "created_at", "")),
        }


class Observation:
    """
    An observation from the theory-of-mind system.

    Observations are facts derived from messages that help build a representation
    of a peer.

    Attributes:
        id: Unique identifier for this observation
        content: The observation content/text
        observer_id: The peer who made the observation
        observed_id: The peer being observed
        session_id: The session where this observation was made
        created_at: When the observation was created
    """

    id: str
    content: str
    observer_id: str
    observed_id: str
    session_id: str
    created_at: str

    def __init__(
        self,
        id: str,
        content: str,
        observer_id: str,
        observed_id: str,
        session_id: str,
        created_at: str,
    ):
        self.id = id
        self.content = content
        self.observer_id = observer_id
        self.observed_id = observed_id
        self.session_id = session_id
        self.created_at = created_at

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "Observation":
        """Create an Observation from an API response dict."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            observer_id=data.get("observer_id", ""),
            observed_id=data.get("observed_id", ""),
            session_id=data.get("session_id", ""),
            created_at=str(data.get("created_at", "")),
        )

    def __repr__(self) -> str:
        truncated = (
            f"{self.content[:50]}..." if len(self.content) > 50 else self.content
        )
        return f"Observation(id={self.id!r}, content={truncated!r})"


class ObservationScope:
    """
    Scoped access to observations for a specific observer/observed relationship.

    This class provides convenient methods to list, query, create, and delete observations
    that are automatically scoped to a specific observer/observed pair.

    Typically accessed via `peer.observations` (for self-observations) or
    `peer.observations_of(target)` (for observations about another peer).

    Example:
        ```python
        # Get self-observations
        observations = peer.observations
        obs_list = observations.list()
        search_results = observations.query("preferences")

        # Get observations about another peer
        bob_observations = peer.observations_of("bob")
        bob_list = bob_observations.list()
        ```
    """

    _http: HttpClient
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        http: HttpClient,
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize an ObservationScope.

        Args:
            http: The HTTP client instance
            workspace_id: The workspace ID
            observer: The observer peer ID
            observed: The observed peer ID
        """
        self._http = http
        self.workspace_id = workspace_id
        self.observer = observer
        self.observed = observed

    def list(
        self,
        page: int = 1,
        size: int = 50,
        session: str | SessionBase | None = None,
    ) -> list[Observation]:
        """
        List observations in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session: Optional session (ID string or Session object) to filter by

        Returns:
            List of Observation objects
        """
        resolved_session_id = (
            None
            if session is None
            else (session if isinstance(session, str) else session.id)
        )
        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }
        if resolved_session_id:
            filters["session_id"] = resolved_session_id

        response = self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/observations/list",
            json={"filters": filters, "page": page, "size": size},
        )

        items = response.get("items", []) if response else []
        return [
            Observation.from_api_response(_convert_observation(item)) for item in items
        ]

    def query(
        self,
        query: str,
        top_k: int = 10,
        distance: float | None = None,
    ) -> list[Observation]:
        """
        Semantic search for observations in this scope.

        Args:
            query: The search query string
            top_k: Maximum number of results to return
            distance: Maximum cosine distance threshold (0.0-1.0)

        Returns:
            List of matching Observation objects
        """
        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }

        response = self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/observations/query",
            json={
                "query": query,
                "top_k": top_k,
                "distance": distance,
                "filters": filters,
            },
        )

        return [
            Observation.from_api_response(_convert_observation(item))
            for item in (response or [])
        ]

    def delete(self, observation_id: str) -> None:
        """
        Delete an observation by ID.

        Args:
            observation_id: The ID of the observation to delete
        """
        self._http.request(
            "DELETE",
            f"/v2/workspaces/{self.workspace_id}/observations/{observation_id}",
        )

    def create(
        self,
        observations: "ObservationCreateParam | list[ObservationCreateParam]",
    ) -> list[Observation]:
        """
        Create observations in this scope.

        Args:
            observations: Single observation or list of observations to create.
                Each observation must have 'content' and 'session_id' keys.

        Returns:
            List of created Observation objects

        Example:
            ```python
            # Create a single observation
            observations = peer.observations.create(
                {"content": "User prefers dark mode", "session_id": "session1"}
            )

            # Create multiple observations
            observations = peer.observations.create([
                {"content": "User prefers dark mode", "session_id": "session1"},
                {"content": "User is interested in AI", "session_id": "session1"},
            ])
            ```
        """
        # Normalize to list
        if not isinstance(observations, list):
            observations = [observations]

        # Build the request body with observer/observed from scope
        request_observations = [
            {
                "content": obs["content"],
                "session_id": obs["session_id"]
                if isinstance(obs["session_id"], str)
                else obs["session_id"].id,
                "observer_id": self.observer,
                "observed_id": self.observed,
            }
            for obs in observations
        ]

        response = self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/observations",
            json={"observations": request_observations},
        )

        return [
            Observation.from_api_response(_convert_observation(item))
            for item in (response or [])
        ]

    def get_representation(
        self,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_derived: bool | None = None,
        max_observations: int | None = None,
    ) -> "Representation":
        """
        Get the computed representation for this scope.

        This returns the working representation (narrative) built from the
        observations in this scope.

        Args:
            search_query: Optional semantic search query to curate the representation
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_derived: Whether to include the most derived observations
            max_observations: Maximum number of observations to include

        Returns:
            A Representation object containing explicit and deductive observations
        """
        from .types import Representation

        body: dict[str, Any] = {"target": self.observed}
        if search_query is not None:
            body["search_query"] = search_query
        if search_top_k is not None:
            body["search_top_k"] = search_top_k
        if search_max_distance is not None:
            body["search_max_distance"] = search_max_distance
        if include_most_derived is not None:
            body["include_most_derived"] = include_most_derived
        if max_observations is not None:
            body["max_observations"] = max_observations

        response = self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.observer}/representation",
            json=body,
        )

        representation = response.get("representation") if response else None
        if representation is not None:
            return Representation.from_dict(cast(dict[str, Any], representation))
        else:
            return Representation.from_dict(response or {})

    def __repr__(self) -> str:
        return (
            f"ObservationScope(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )


class AsyncObservationScope:
    """
    Async scoped access to observations for a specific observer/observed relationship.

    This class provides convenient async methods to list, query, create, and delete observations
    that are automatically scoped to a specific observer/observed pair.

    Typically accessed via `peer.observations` (for self-observations) or
    `peer.observations_of(target)` (for observations about another peer).

    Example:
        ```python
        # Get self-observations
        observations = peer.observations
        obs_list = await observations.list()
        search_results = await observations.query("preferences")

        # Get observations about another peer
        bob_observations = peer.observations_of("bob")
        bob_list = await bob_observations.list()
        ```
    """

    _http: AsyncHttpClient
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        http: AsyncHttpClient,
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize an AsyncObservationScope.

        Args:
            http: The async HTTP client instance
            workspace_id: The workspace ID
            observer: The observer peer ID
            observed: The observed peer ID
        """
        self._http = http
        self.workspace_id = workspace_id
        self.observer = observer
        self.observed = observed

    async def list(
        self,
        page: int = 1,
        size: int = 50,
        session: str | SessionBase | None = None,
    ) -> list[Observation]:
        """
        List observations in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session: Optional session (ID string or AsyncSession object) to filter by

        Returns:
            List of Observation objects
        """
        resolved_session_id = (
            None
            if session is None
            else (session if isinstance(session, str) else session.id)
        )
        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }
        if resolved_session_id:
            filters["session_id"] = resolved_session_id

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/observations/list",
            json={"filters": filters, "page": page, "size": size},
        )

        items = response.get("items", []) if response else []
        return [
            Observation.from_api_response(_convert_observation(item)) for item in items
        ]

    async def query(
        self,
        query: str,
        top_k: int = 10,
        distance: float | None = None,
    ) -> list[Observation]:
        """
        Semantic search for observations in this scope.

        Args:
            query: The search query string
            top_k: Maximum number of results to return
            distance: Maximum cosine distance threshold (0.0-1.0)

        Returns:
            List of matching Observation objects
        """
        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/observations/query",
            json={
                "query": query,
                "top_k": top_k,
                "distance": distance,
                "filters": filters,
            },
        )

        return [
            Observation.from_api_response(_convert_observation(item))
            for item in (response or [])
        ]

    async def delete(self, observation_id: str) -> None:
        """
        Delete an observation by ID.

        Args:
            observation_id: The ID of the observation to delete
        """
        await self._http.request(
            "DELETE",
            f"/v2/workspaces/{self.workspace_id}/observations/{observation_id}",
        )

    async def create(
        self,
        observations: "ObservationCreateParam | list[ObservationCreateParam]",
    ) -> list[Observation]:
        """
        Create observations in this scope.

        Args:
            observations: Single observation or list of observations to create.
                Each observation must have 'content' and 'session_id' keys.

        Returns:
            List of created Observation objects

        Example:
            ```python
            # Create a single observation
            observations = await peer.observations.create(
                {"content": "User prefers dark mode", "session_id": "session1"}
            )

            # Create multiple observations
            observations = await peer.observations.create([
                {"content": "User prefers dark mode", "session_id": "session1"},
                {"content": "User is interested in AI", "session_id": "session1"},
            ])
            ```
        """
        # Normalize to list
        if not isinstance(observations, list):
            observations = [observations]

        # Build the request body with observer/observed from scope
        request_observations = [
            {
                "content": obs["content"],
                "session_id": obs["session_id"]
                if isinstance(obs["session_id"], str)
                else obs["session_id"].id,
                "observer_id": self.observer,
                "observed_id": self.observed,
            }
            for obs in observations
        ]

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/observations",
            json={"observations": request_observations},
        )

        return [
            Observation.from_api_response(_convert_observation(item))
            for item in (response or [])
        ]

    async def get_representation(
        self,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_derived: bool | None = None,
        max_observations: int | None = None,
    ) -> "Representation":
        """
        Get the computed representation for this scope.

        This returns the working representation (narrative) built from the
        observations in this scope.

        Args:
            search_query: Optional semantic search query to curate the representation
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_derived: Whether to include the most derived observations
            max_observations: Maximum number of observations to include

        Returns:
            A Representation object containing explicit and deductive observations
        """
        from .types import Representation

        body: dict[str, Any] = {"target": self.observed}
        if search_query is not None:
            body["search_query"] = search_query
        if search_top_k is not None:
            body["search_top_k"] = search_top_k
        if search_max_distance is not None:
            body["search_max_distance"] = search_max_distance
        if include_most_derived is not None:
            body["include_most_derived"] = include_most_derived
        if max_observations is not None:
            body["max_observations"] = max_observations

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers/{self.observer}/representation",
            json=body,
        )

        representation = response.get("representation") if response else None
        if representation is not None:
            return Representation.from_dict(cast(dict[str, Any], representation))
        else:
            return Representation.from_dict(response or {})

    def __repr__(self) -> str:
        return (
            f"AsyncObservationScope(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )
