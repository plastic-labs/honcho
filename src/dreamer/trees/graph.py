"""
Graph-theoretic surprisal using k-NN graph and random walk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .base import SurprisalTree

if TYPE_CHECKING:
    from sklearn.neighbors import NearestNeighbors


def _knn_indices(
    points: NDArray[np.floating[Any]], n_neighbors: int
) -> NDArray[np.intp]:
    """Get k-nearest neighbor indices for each point."""
    from sklearn.neighbors import NearestNeighbors

    knn: NearestNeighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    knn.fit(points)  # pyright: ignore[reportUnknownMemberType]
    _distances, indices = knn.kneighbors(points)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return indices  # pyright: ignore[reportUnknownVariableType]


def _nearest_index(points: NDArray[np.floating[Any]], query: np.ndarray) -> int:
    """Find index of nearest point to query."""
    from sklearn.neighbors import NearestNeighbors

    knn: NearestNeighbors = NearestNeighbors(n_neighbors=1)
    knn.fit(points)  # pyright: ignore[reportUnknownMemberType]
    _distances, indices = knn.kneighbors([query])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return int(indices[0, 0])  # pyright: ignore[reportUnknownArgumentType]


class GraphSurprisal(SurprisalTree):
    """
    Graph-theoretic surprisal using k-NN graph and random walk.
    Surprisal based on stationary distribution of random walk.
    """

    k: int
    max_iter: int
    points: list[NDArray[np.floating[Any]]]
    stationary_dist: NDArray[np.floating[Any]] | None
    graph_built: bool
    total_points: int

    def __init__(
        self, k: int = 5, max_iter: int = 100, max_leaf_size: int = 10
    ) -> None:
        super().__init__(max_leaf_size)
        self.k = k
        self.max_iter = max_iter
        self.points = []
        self.stationary_dist = None
        self.graph_built = False

    def insert(self, point: np.ndarray) -> None:
        self.points.append(point)
        self.total_points += 1
        self.graph_built = False

    def batch_insert(self, points: np.ndarray) -> None:
        """More efficient batch insertion."""
        self.points.extend(points)
        self.total_points += len(points)
        self.graph_built = False

    def _build_graph_and_compute_stationary(self) -> None:
        """Build k-NN graph and compute stationary distribution."""
        if len(self.points) < 2:
            return

        points_array: NDArray[np.floating[Any]] = np.array(self.points)
        k_actual = min(self.k, len(self.points) - 1)

        indices = _knn_indices(points_array, k_actual + 1)

        n = len(self.points)
        transition = np.zeros((n, n))

        for i in range(n):
            neighbors: NDArray[np.intp] = indices[i, 1:]
            for j_idx in neighbors:
                j: int = int(j_idx)
                transition[i, j] = 1.0

        row_sums = transition.sum(axis=1, keepdims=True)
        # Add self-loops for isolated nodes to maintain stochasticity
        for i in range(n):
            if row_sums[i, 0] == 0:
                transition[i, i] = 1.0
                row_sums[i, 0] = 1.0
        transition = transition / row_sums

        stationary: NDArray[np.floating[Any]] = np.ones(n) / n
        for _ in range(self.max_iter):
            new_stationary: NDArray[np.floating[Any]] = transition.T @ stationary
            if np.allclose(new_stationary, stationary, atol=1e-6):
                break
            stationary = new_stationary

        self.stationary_dist = stationary / stationary.sum()
        self.graph_built = True

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal based on stationary distribution.
        Well-connected central facts = high probability = low surprisal
        Peripheral isolated facts = low probability = high surprisal
        """
        if not self.graph_built:
            self._build_graph_and_compute_stationary()

        if self.stationary_dist is None or len(self.points) == 0:
            return float("inf")

        points_array: NDArray[np.floating[Any]] = np.array(self.points)
        nearest_idx = _nearest_index(points_array, point)

        prob = self.stationary_dist[nearest_idx]

        return float(-np.log(prob + 1e-10))
