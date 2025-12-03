"""Observation types and scoped access for the Honcho SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .types import Representation


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
            created_at=data.get("created_at", ""),
        )

    def __repr__(self) -> str:
        truncated = (
            f"{self.content[:50]}..." if len(self.content) > 50 else self.content
        )
        return f"Observation(id={self.id!r}, content={truncated!r})"


class ObservationScope:
    """
    Scoped access to observations for a specific observer/observed relationship.

    This class provides convenient methods to list, query, and delete observations
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

    Note:
        This class requires the core Honcho SDK to support observation endpoints.
        The observation endpoints are:
        - POST /workspaces/{workspace_id}/observations/list
        - POST /workspaces/{workspace_id}/observations/query
        - DELETE /workspaces/{workspace_id}/observations/{observation_id}
    """

    _client: Any
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        client: Any,
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize an ObservationScope.

        Args:
            client: The Honcho client instance
            workspace_id: The workspace ID
            observer: The observer peer ID
            observed: The observed peer ID
        """
        self._client = client
        self.workspace_id = workspace_id
        self.observer = observer
        self.observed = observed

    def list(
        self,
        page: int = 1,
        size: int = 50,
        session_id: str | None = None,
    ) -> list[Observation]:
        """
        List observations in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session_id: Optional session ID to filter by

        Returns:
            List of Observation objects
        """
        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }
        if session_id:
            filters["session_id"] = session_id

        # Note: This requires the core SDK to support observations.list()
        response = self._client.workspaces.observations.list(
            workspace_id=self.workspace_id,
            filters=filters,
            page=page,
            size=size,
        )

        return [Observation.from_api_response(item) for item in response.items]

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

        # Note: This requires the core SDK to support observations.query()
        response = self._client.workspaces.observations.query(
            workspace_id=self.workspace_id,
            query=query,
            top_k=top_k,
            distance=distance,
            filters=filters,
        )

        return [Observation.from_api_response(item) for item in response]

    def delete(self, observation_id: str) -> None:
        """
        Delete an observation by ID.

        Args:
            observation_id: The ID of the observation to delete
        """
        # Note: This requires the core SDK to support observations.delete()
        self._client.workspaces.observations.delete(
            workspace_id=self.workspace_id,
            observation_id=observation_id,
        )

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
        from honcho_core._types import omit

        from .types import Representation

        response = self._client.workspaces.peers.working_representation(
            peer_id=self.observer,
            workspace_id=self.workspace_id,
            target=self.observed,
            search_query=search_query if search_query is not None else omit,
            search_top_k=search_top_k if search_top_k is not None else omit,
            search_max_distance=search_max_distance
            if search_max_distance is not None
            else omit,
            include_most_derived=include_most_derived
            if include_most_derived is not None
            else omit,
            max_observations=max_observations if max_observations is not None else omit,
        )

        representation = response.get("representation")
        if representation is not None:
            return Representation.from_dict(cast(dict[str, Any], representation))
        else:
            return Representation.from_dict(response)

    def __repr__(self) -> str:
        return (
            f"ObservationScope(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )


class AsyncObservationScope:
    """
    Async scoped access to observations for a specific observer/observed relationship.

    This class provides convenient async methods to list, query, and delete observations
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

    Note:
        This class requires the core Honcho SDK to support observation endpoints.
        The observation endpoints are:
        - POST /workspaces/{workspace_id}/observations/list
        - POST /workspaces/{workspace_id}/observations/query
        - DELETE /workspaces/{workspace_id}/observations/{observation_id}
    """

    _client: Any
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        client: Any,
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize an AsyncObservationScope.

        Args:
            client: The AsyncHoncho client instance
            workspace_id: The workspace ID
            observer: The observer peer ID
            observed: The observed peer ID
        """
        self._client = client
        self.workspace_id = workspace_id
        self.observer = observer
        self.observed = observed

    async def list(
        self,
        page: int = 1,
        size: int = 50,
        session_id: str | None = None,
    ) -> list[Observation]:
        """
        List observations in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session_id: Optional session ID to filter by

        Returns:
            List of Observation objects
        """
        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }
        if session_id:
            filters["session_id"] = session_id

        # Note: This requires the core SDK to support observations.list()
        response = await self._client.workspaces.observations.list(
            workspace_id=self.workspace_id,
            filters=filters,
            page=page,
            size=size,
        )

        return [Observation.from_api_response(item) for item in response.items]

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

        # Note: This requires the core SDK to support observations.query()
        response = await self._client.workspaces.observations.query(
            workspace_id=self.workspace_id,
            query=query,
            top_k=top_k,
            distance=distance,
            filters=filters,
        )

        return [Observation.from_api_response(item) for item in response]

    async def delete(self, observation_id: str) -> None:
        """
        Delete an observation by ID.

        Args:
            observation_id: The ID of the observation to delete
        """
        # Note: This requires the core SDK to support observations.delete()
        await self._client.workspaces.observations.delete(
            workspace_id=self.workspace_id,
            observation_id=observation_id,
        )

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
        from honcho_core._types import omit

        from .types import Representation

        response = await self._client.workspaces.peers.working_representation(
            peer_id=self.observer,
            workspace_id=self.workspace_id,
            target=self.observed,
            search_query=search_query if search_query is not None else omit,
            search_top_k=search_top_k if search_top_k is not None else omit,
            search_max_distance=search_max_distance
            if search_max_distance is not None
            else omit,
            include_most_derived=include_most_derived
            if include_most_derived is not None
            else omit,
            max_observations=max_observations if max_observations is not None else omit,
        )

        representation = response.get("representation")
        if representation is not None:
            return Representation.from_dict(cast(dict[str, Any], representation))
        else:
            return Representation.from_dict(response)

    def __repr__(self) -> str:
        return (
            f"AsyncObservationScope(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )
