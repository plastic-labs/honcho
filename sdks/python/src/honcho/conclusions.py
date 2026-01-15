"""Conclusion types and scoped access for the Honcho SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .api_types import ConclusionResponse, RepresentationResponse
from .base import SessionBase

if TYPE_CHECKING:
    from .http import AsyncHonchoHTTPClient, HonchoHTTPClient

__all__ = [
    "ConclusionResponse",
    "ConclusionCreateResponse",
    "ConclusionScope",
    "ConclusionCreateParams",
    "AsyncConclusionScope",
]

ConclusionCreateResponse = list[ConclusionResponse]


class ConclusionCreateParams(BaseModel):
    content: str
    session_id: str


class ConclusionScope:
    """
    Scoped access to conclusions for a specific observer/observed relationship.

    This class provides convenient methods to list, query, create, and delete conclusions
    that are automatically scoped to a specific observer/observed pair.

    Typically accessed via `peer.conclusions` (for self-conclusions) or
    `peer.conclusions_of(target)` (for conclusions about another peer).

    Example:
        ```python
        # Get self-conclusions
        conclusions = peer.conclusions
        obs_list = conclusions.list()
        search_results = conclusions.query("preferences")

        # Get conclusions about another peer
        bob_conclusions = peer.conclusions_of("bob")
        bob_list = bob_conclusions.list()
        ```
    """

    _http: "HonchoHTTPClient"
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        http: "HonchoHTTPClient",
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize a ConclusionScope.

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
    ) -> "SyncPage[ConclusionResponse, ConclusionResponse]":
        """
        List conclusions in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session: Optional session (ID string or Session object) to filter by

        Returns:
            Paginated response containing ConclusionResponse objects
        """
        from .http import routes
        from .pagination import SyncPage

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

        data = self._http.post(
            routes.conclusions_list(self.workspace_id),
            body={"filters": filters, "page": page, "size": size},
        )

        def fetch_next(
            next_page: int,
        ) -> "SyncPage[ConclusionResponse, ConclusionResponse]":
            next_data = self._http.post(
                routes.conclusions_list(self.workspace_id),
                body={"filters": filters, "page": next_page, "size": size},
            )
            return SyncPage(next_data, ConclusionResponse, None, fetch_next)

        return SyncPage(data, ConclusionResponse, None, fetch_next)

    def query(
        self,
        query: str,
        top_k: int = 10,
        distance: float | None = None,
    ) -> list[ConclusionResponse]:
        """
        Semantic search for conclusions in this scope.

        Args:
            query: The search query string
            top_k: Maximum number of results to return
            distance: Maximum cosine distance threshold (0.0-1.0)

        Returns:
            List of matching ConclusionResponse objects
        """
        from .http import routes

        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }

        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
        }
        if distance is not None:
            body["distance"] = distance

        data = self._http.post(
            routes.conclusions_query(self.workspace_id),
            body=body,
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    def delete(self, conclusion_id: str) -> None:
        """
        Delete a conclusion by ID.

        Args:
            conclusion_id: The ID of the conclusion to delete
        """
        from .http import routes

        self._http.delete(routes.conclusion(self.workspace_id, conclusion_id))

    def create(
        self,
        conclusions: list[ConclusionCreateParams | dict[str, Any]],
    ) -> list[ConclusionResponse]:
        """
        Create conclusions in this scope.

        Args:
            conclusions: List of conclusions to create.
                Each conclusion can be a ConclusionCreateParams object or a dictionary with 'content' and 'session_id' keys.

        Returns:
            List of created ConclusionResponse objects

        Example:
            ```python
            conclusions = peer.conclusions.create([
                {"content": "User prefers dark mode", "session_id": "session1"},
                {"content": "User is interested in AI", "session_id": "session1"},
            ])
            ```
        """
        from .http import routes

        conclusion_params = [
            {
                "content": c.content
                if isinstance(c, ConclusionCreateParams)
                else c["content"],
                "session_id": c.session_id
                if isinstance(c, ConclusionCreateParams)
                else c["session_id"],
                "observer_id": self.observer,
                "observed_id": self.observed,
            }
            for c in conclusions
        ]

        data = self._http.post(
            routes.conclusions(self.workspace_id),
            body={"conclusions": conclusion_params},
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    def get_representation(
        self,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> str:
        """
        Get the computed representation for this scope.

        This returns the working representation (narrative) built from the
        conclusions in this scope.

        Args:
            search_query: Optional semantic search query to curate the representation
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_frequent: Whether to include the most frequent conclusions
            max_conclusions: Maximum number of conclusions to include

        Returns:
            A Representation string
        """
        from .http import routes

        query: dict[str, Any] = {"target": self.observed}
        if search_query is not None:
            query["search_query"] = search_query
        if search_top_k is not None:
            query["search_top_k"] = search_top_k
        if search_max_distance is not None:
            query["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            query["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            query["max_conclusions"] = max_conclusions

        data = self._http.get(
            routes.peer_representation(self.workspace_id, self.observer),
            query=query,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    def __repr__(self) -> str:
        return (
            f"ConclusionScope(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )


class AsyncConclusionScope:
    """
    Async scoped access to conclusions for a specific observer/observed relationship.

    This class provides convenient async methods to list, query, create, and delete conclusions
    that are automatically scoped to a specific observer/observed pair.

    Typically accessed via `peer.conclusions` (for self-conclusions) or
    `peer.conclusions_of(target)` (for conclusions about another peer).

    Example:
        ```python
        # Get self-conclusions
        conclusions = peer.conclusions
        obs_list = await conclusions.list()
        search_results = await conclusions.query("preferences")

        # Get conclusions about another peer
        bob_conclusions = peer.conclusions_of("bob")
        bob_list = await bob_conclusions.list()
        ```
    """

    _http: "AsyncHonchoHTTPClient"
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        http: "AsyncHonchoHTTPClient",
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize an AsyncConclusionScope.

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
    ) -> "AsyncPage[ConclusionResponse, ConclusionResponse]":
        """
        List conclusions in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session: Optional session (ID string or AsyncSession object) to filter by

        Returns:
            Paginated response containing ConclusionResponse objects
        """
        from .async_client.pagination import AsyncPage
        from .http import routes

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

        data = await self._http.post(
            routes.conclusions_list(self.workspace_id),
            body={"filters": filters, "page": page, "size": size},
        )

        async def fetch_next(
            next_page: int,
        ) -> "AsyncPage[ConclusionResponse, ConclusionResponse]":
            next_data = await self._http.post(
                routes.conclusions_list(self.workspace_id),
                body={"filters": filters, "page": next_page, "size": size},
            )
            return AsyncPage(next_data, ConclusionResponse, None, fetch_next)

        return AsyncPage(data, ConclusionResponse, None, fetch_next)

    async def query(
        self,
        query: str,
        top_k: int = 10,
        distance: float | None = None,
    ) -> list[ConclusionResponse]:
        """
        Semantic search for conclusions in this scope.

        Args:
            query: The search query string
            top_k: Maximum number of results to return
            distance: Maximum cosine distance threshold (0.0-1.0)

        Returns:
            List of matching ConclusionResponse objects
        """
        from .http import routes

        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }

        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
        }
        if distance is not None:
            body["distance"] = distance

        data = await self._http.post(
            routes.conclusions_query(self.workspace_id),
            body=body,
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    async def delete(self, conclusion_id: str) -> None:
        """
        Delete a conclusion by ID.

        Args:
            conclusion_id: The ID of the conclusion to delete
        """
        from .http import routes

        await self._http.delete(routes.conclusion(self.workspace_id, conclusion_id))

    async def create(
        self,
        conclusions: list[ConclusionCreateParams | dict[str, Any]],
    ) -> list[ConclusionResponse]:
        """
        Create conclusions in this scope.

        Args:
            conclusions: List of conclusions to create.
                Each conclusion can be a ConclusionCreateParams object or a dictionary with 'content' and 'session_id' keys.

        Returns:
            List of created ConclusionResponse objects

        Example:
            ```python
            conclusions = await peer.conclusions.create([
                {"content": "User prefers dark mode", "session_id": "session1"},
                {"content": "User is interested in AI", "session_id": "session1"},
            ])
            ```
        """
        from .http import routes

        conclusion_params = [
            {
                "content": c.content
                if isinstance(c, ConclusionCreateParams)
                else c["content"],
                "session_id": c.session_id
                if isinstance(c, ConclusionCreateParams)
                else c["session_id"],
                "observer_id": self.observer,
                "observed_id": self.observed,
            }
            for c in conclusions
        ]

        data = await self._http.post(
            routes.conclusions(self.workspace_id),
            body={"conclusions": conclusion_params},
        )
        return [ConclusionResponse.model_validate(item) for item in data]

    async def get_representation(
        self,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> str:
        """
        Get the computed representation for this scope.

        This returns the working representation (narrative) built from the
        conclusions in this scope.

        Args:
            search_query: Optional semantic search query to curate the representation
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_frequent: Whether to include the most frequent conclusions
            max_conclusions: Maximum number of conclusions to include

        Returns:
            A Representation string
        """
        from .http import routes

        query: dict[str, Any] = {"target": self.observed}
        if search_query is not None:
            query["search_query"] = search_query
        if search_top_k is not None:
            query["search_top_k"] = search_top_k
        if search_max_distance is not None:
            query["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            query["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            query["max_conclusions"] = max_conclusions

        data = await self._http.get(
            routes.peer_representation(self.workspace_id, self.observer),
            query=query,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    def __repr__(self) -> str:
        return (
            f"AsyncConclusionScope(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )


# Import for type hints
from .async_client.pagination import AsyncPage  # noqa: E402
from .pagination import SyncPage  # noqa: E402
