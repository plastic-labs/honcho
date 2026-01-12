"""Conclusion types and scoped access for the Honcho SDK."""

from __future__ import annotations

from typing import Any

from honcho_core import AsyncHoncho as AsyncHonchoCore
from honcho_core import Honcho as HonchoCore
from honcho_core.pagination import AsyncPage, SyncPage
from honcho_core.types.workspaces import conclusion_create_params
from honcho_core.types.workspaces.conclusion import Conclusion
from pydantic import BaseModel, PrivateAttr
from typing_extensions import TypeAlias

from .base import SessionBase

__all__ = [
    "Conclusion",
    "ConclusionCreateResponse",
    "ConclusionScope",
    "ConclusionCreateParams",
    "AsyncConclusionScope",
]

ConclusionCreateResponse: TypeAlias = list[Conclusion]


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

    _client: HonchoCore = PrivateAttr()
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        client: HonchoCore,
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize a ConclusionScope.

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
        session: str | SessionBase | None = None,
    ) -> SyncPage[Conclusion]:
        """
        List conclusions in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session: Optional session (ID string or Session object) to filter by

        Returns:
            List of Conclusion objects
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

        return self._client.workspaces.conclusions.list(
            workspace_id=self.workspace_id,
            filters=filters,
            page=page,
            size=size,
        )

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
        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }

        return self._client.workspaces.conclusions.query(
            workspace_id=self.workspace_id,
            query=query,
            top_k=top_k,
            distance=distance,
            filters=filters,
        )

    def delete(self, conclusion_id: str) -> None:
        """
        Delete a conclusion by ID.

        Args:
            conclusion_id: The ID of the conclusion to delete
        """
        self._client.workspaces.conclusions.delete(
            workspace_id=self.workspace_id,
            conclusion_id=conclusion_id,
        )

    def create(
        self,
        conclusions: list[ConclusionCreateParams | dict[str, Any]],
    ) -> list[Conclusion]:
        """
        Create conclusions in this scope.

        Args:
            conclusions: List of conclusions to create.
                Each conclusion can be a ConclusionCreateParams object or a dictionary with 'content' and 'session_id' keys.

        Returns:
            List of created Conclusion objects

        Example:
            ```python
            conclusions = peer.conclusions.create([
                {"content": "User prefers dark mode", "session_id": "session1"},
                {"content": "User is interested in AI", "session_id": "session1"},
            ])
            ```
        """

        return self._client.workspaces.conclusions.create(
            workspace_id=self.workspace_id,
            conclusions=[
                conclusion_create_params.Conclusion(
                    content=conclusion.content
                    if isinstance(conclusion, ConclusionCreateParams)
                    else conclusion["content"],
                    session_id=conclusion.session_id
                    if isinstance(conclusion, ConclusionCreateParams)
                    else conclusion["session_id"],
                    observer_id=self.observer,
                    observed_id=self.observed,
                )
                for conclusion in conclusions
            ],
        )

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
        from honcho_core._types import omit

        response = self._client.workspaces.peers.representation(
            peer_id=self.observer,
            workspace_id=self.workspace_id,
            target=self.observed,
            search_query=search_query if search_query is not None else omit,
            search_top_k=search_top_k if search_top_k is not None else omit,
            search_max_distance=search_max_distance
            if search_max_distance is not None
            else omit,
            include_most_frequent=include_most_frequent
            if include_most_frequent is not None
            else omit,
            max_conclusions=max_conclusions if max_conclusions is not None else omit,
        )

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

    _client: AsyncHonchoCore = PrivateAttr()
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        client: AsyncHonchoCore,
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize an AsyncConclusionScope.

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
        session: str | SessionBase | None = None,
    ) -> AsyncPage[Conclusion]:
        """
        List conclusions in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session: Optional session (ID string or AsyncSession object) to filter by

        Returns:
            List of Conclusion objects
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

        return await self._client.workspaces.conclusions.list(
            workspace_id=self.workspace_id,
            filters=filters,
            page=page,
            size=size,
        )

    async def query(
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
        filters: dict[str, Any] = {
            "observer": self.observer,
            "observed": self.observed,
        }

        return await self._client.workspaces.conclusions.query(
            workspace_id=self.workspace_id,
            query=query,
            top_k=top_k,
            distance=distance,
            filters=filters,
        )

    async def delete(self, conclusion_id: str) -> None:
        """
        Delete a conclusion by ID.

        Args:
            conclusion_id: The ID of the conclusion to delete
        """
        await self._client.workspaces.conclusions.delete(
            workspace_id=self.workspace_id,
            conclusion_id=conclusion_id,
        )

    async def create(
        self,
        conclusions: list[ConclusionCreateParams | dict[str, Any]],
    ) -> list[Conclusion]:
        """
        Create conclusions in this scope.

        Args:
            conclusions: List of conclusions to create.
                Each conclusion can be a ConclusionCreateParams object or a dictionary with 'content' and 'session_id' keys.

        Returns:
            List of created Conclusion objects

        Example:
            ```python
            conclusions = await peer.conclusions.create([
                {"content": "User prefers dark mode", "session_id": "session1"},
                {"content": "User is interested in AI", "session_id": "session1"},
            ])
            ```
        """
        return await self._client.workspaces.conclusions.create(
            workspace_id=self.workspace_id,
            conclusions=[
                conclusion_create_params.Conclusion(
                    content=conclusion.content
                    if isinstance(conclusion, ConclusionCreateParams)
                    else conclusion["content"],
                    session_id=conclusion.session_id
                    if isinstance(conclusion, ConclusionCreateParams)
                    else conclusion["session_id"],
                    observer_id=self.observer,
                    observed_id=self.observed,
                )
                for conclusion in conclusions
            ],
        )

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
        from honcho_core._types import omit

        response = await self._client.workspaces.peers.representation(
            peer_id=self.observer,
            workspace_id=self.workspace_id,
            target=self.observed,
            search_query=search_query if search_query is not None else omit,
            search_top_k=search_top_k if search_top_k is not None else omit,
            search_max_distance=search_max_distance
            if search_max_distance is not None
            else omit,
            include_most_frequent=include_most_frequent
            if include_most_frequent is not None
            else omit,
            max_conclusions=max_conclusions if max_conclusions is not None else omit,
        )

        return response.representation

    def __repr__(self) -> str:
        return (
            f"AsyncConclusionScope(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )
