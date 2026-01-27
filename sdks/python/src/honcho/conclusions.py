# pyright: reportPrivateUsage=false
"""Conclusion types and scoped access for the Honcho SDK."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .api_types import ConclusionResponse, RepresentationResponse
from .base import SessionBase
from .http import routes
from .pagination import SyncPage
from .utils import resolve_id

if TYPE_CHECKING:
    from .aio import ConclusionScopeAio
    from .client import Honcho

__all__ = [
    "Conclusion",
    "ConclusionScope",
    "ConclusionCreateParams",
]


class ConclusionCreateParams(BaseModel):
    content: str
    session_id: str | None = None


class Conclusion:
    """
    A conclusion from Honcho's reasoning system.

    Conclusions are facts derived from messages that help build a representation
    of a peer.

    Attributes:
        id: Unique identifier for this conclusion
        content: The conclusion content/text
        observer_id: The peer ID who made this conclusion
        observed_id: The peer ID this conclusion is about
        session_id: The session this conclusion relates to
        created_at: Timestamp for when the conclusion was created
    """

    id: str
    content: str
    observer_id: str
    observed_id: str
    session_id: str | None = None
    created_at: datetime.datetime

    def __init__(
        self,
        id: str,
        content: str,
        observer_id: str,
        observed_id: str,
        session_id: str | None,
        created_at: datetime.datetime,
    ) -> None:
        self.id = id
        self.content = content
        self.observer_id = observer_id
        self.observed_id = observed_id
        self.session_id = session_id
        self.created_at = created_at

    @classmethod
    def from_api_response(cls, data: ConclusionResponse) -> "Conclusion":
        """Create a Conclusion from an API response."""
        return cls(
            id=data.id,
            content=data.content,
            observer_id=data.observer_id,
            observed_id=data.observed_id,
            session_id=data.session_id,
            created_at=data.created_at,
        )

    def __repr__(self) -> str:
        truncated = (
            f"{self.content[:50]}..." if len(self.content) > 50 else self.content
        )
        return f"Conclusion(id='{self.id}', content='{truncated}')"

    def __str__(self) -> str:
        return self.content


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

        # Async operations via .aio accessor
        obs_list = await peer.conclusions.aio.list()
        ```
    """

    _honcho: "Honcho"
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        honcho: "Honcho",
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize a ConclusionScope.

        Args:
            honcho: The Honcho client instance
            workspace_id: The workspace ID
            observer: The observer peer ID
            observed: The observed peer ID
        """
        self._honcho = honcho
        self.workspace_id = workspace_id
        self.observer = observer
        self.observed = observed

    @property
    def aio(self) -> "ConclusionScopeAio":
        """
        Access async versions of all ConclusionScope methods.

        Returns a ConclusionScopeAio view that provides async versions of all methods
        while sharing state with this ConclusionScope instance.

        Example:
            ```python
            # Async operations
            obs_list = await scope.aio.list()
            results = await scope.aio.query("preferences")
            ```
        """
        # Import here to avoid circular import (aio.py imports from this module)
        from .aio import ConclusionScopeAio

        return ConclusionScopeAio(self)

    def list(
        self,
        page: int = 1,
        size: int = 50,
        session: str | SessionBase | None = None,
    ) -> SyncPage[ConclusionResponse, Conclusion]:
        """
        List conclusions in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session: Optional session (ID string or Session object) to filter by

        Returns:
            Paginated response containing Conclusion objects
        """
        self._honcho._ensure_workspace()
        resolved_session_id = resolve_id(session)
        filters: dict[str, Any] = {
            "observer_id": self.observer,
            "observed_id": self.observed,
        }
        if resolved_session_id:
            filters["session_id"] = resolved_session_id

        data = self._honcho._http.post(
            routes.conclusions_list(self.workspace_id),
            body={"filters": filters},
            query={"page": page, "size": size},
        )

        def transform(response: ConclusionResponse) -> Conclusion:
            return Conclusion.from_api_response(response)

        def fetch_next(
            page: int,
        ) -> SyncPage[ConclusionResponse, Conclusion]:
            next_data = self._honcho._http.post(
                routes.conclusions_list(self.workspace_id),
                body={"filters": filters},
                query={"page": page, "size": size},
            )
            return SyncPage(next_data, ConclusionResponse, transform, fetch_next)

        return SyncPage(data, ConclusionResponse, transform, fetch_next)

    def query(
        self,
        query: str,
        top_k: int = 10,
        distance: float | None = None,
    ) -> list[Conclusion]:
        """
        Semantic search for conclusions in this scope.

        Args:
            query: The search query string
            top_k: Maximum number of results to return
            distance: Maximum cosine distance threshold (0.0-1.0)

        Returns:
            List of matching Conclusion objects
        """
        self._honcho._ensure_workspace()
        filters: dict[str, Any] = {
            "observer_id": self.observer,
            "observed_id": self.observed,
        }

        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
        }
        if distance is not None:
            body["distance"] = distance

        data = self._honcho._http.post(
            routes.conclusions_query(self.workspace_id),
            body=body,
        )
        return [
            Conclusion.from_api_response(ConclusionResponse.model_validate(item))
            for item in data
        ]

    def delete(self, conclusion_id: str) -> None:
        """
        Delete a conclusion by ID.

        Args:
            conclusion_id: The ID of the conclusion to delete
        """
        self._honcho._ensure_workspace()
        self._honcho._http.delete(routes.conclusion(self.workspace_id, conclusion_id))

    def create(
        self,
        conclusions: list[ConclusionCreateParams | dict[str, Any]],
    ) -> list[Conclusion]:
        """
        Create conclusions in this scope.

        Args:
            conclusions: List of conclusions to create.
                Each conclusion can be a ConclusionCreateParams object or a dictionary
                with a required 'content' key and an optional 'session_id' key.

        Returns:
            List of created Conclusion objects

        Example:
            ```python
            conclusions = peer.conclusions.create([
                {"content": "User prefers dark mode", "session_id": "session1"},
                {"content": "User is interested in AI"},
            ])
            ```
        """
        self._honcho._ensure_workspace()

        def build_conclusion_payload(
            item: ConclusionCreateParams | dict[str, Any],
        ) -> dict[str, Any]:
            """
            Build a single conclusion create payload.

            This normalizes both `ConclusionCreateParams` instances and plain dictionaries
            into the wire format expected by the Honcho API.

            Notes:
                - `content` is required.
                - `session_id` is optional; when not provided it is omitted from the payload.
                - `observer_id` and `observed_id` are always injected from this scope.

            Args:
                item: A `ConclusionCreateParams` instance or a dict with a required
                    `content` key and an optional `session_id` key.

            Returns:
                A dictionary suitable for inclusion in the `conclusions` array for the
                create conclusions endpoint.
            """
            payload: dict[str, Any] = {
                "observer_id": self.observer,
                "observed_id": self.observed,
            }
            if isinstance(item, ConclusionCreateParams):
                payload["content"] = item.content
                if item.session_id is not None:
                    payload["session_id"] = item.session_id
                return payload

            payload["content"] = item["content"]
            session_id = item.get("session_id")
            if session_id is not None:
                payload["session_id"] = session_id
            return payload

        conclusion_params = [build_conclusion_payload(c) for c in conclusions]

        data = self._honcho._http.post(
            routes.conclusions(self.workspace_id),
            body={"conclusions": conclusion_params},
        )
        return [
            Conclusion.from_api_response(ConclusionResponse.model_validate(item))
            for item in data
        ]

    def representation(
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
        self._honcho._ensure_workspace()
        body: dict[str, Any] = {"target": self.observed}
        if search_query is not None:
            body["search_query"] = search_query
        if search_top_k is not None:
            body["search_top_k"] = search_top_k
        if search_max_distance is not None:
            body["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            body["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            body["max_conclusions"] = max_conclusions

        data = self._honcho._http.post(
            routes.peer_representation(self.workspace_id, self.observer),
            body=body,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    def __repr__(self) -> str:
        return (
            f"ConclusionScope(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )
