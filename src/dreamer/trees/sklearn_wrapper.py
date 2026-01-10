"""
Wrapper for sklearn's KDTree and BallTree.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import (
    BallTree,  # pyright: ignore[reportUnknownVariableType]
    KDTree,  # pyright: ignore[reportUnknownVariableType]
)

from .base import SurprisalTree


class SklearnTreeWrapper(SurprisalTree):
    """
    Wrapper for sklearn's KDTree and BallTree with surprisal computation.
    Uses density estimation via k-nearest neighbors.
    """

    tree_type: str
    k: int
    points: list[NDArray[np.floating[Any]]]
    tree: KDTree | BallTree | None
    total_points: int

    def __init__(
        self, tree_type: str = "kd", k: int = 5, max_leaf_size: int = 10
    ) -> None:
        super().__init__(max_leaf_size)
        self.tree_type = tree_type
        self.k = k
        self.points = []
        self.tree = None

    def insert(self, point: np.ndarray) -> None:
        self.points.append(point)
        self.total_points += 1
        self._rebuild_tree()

    def batch_insert(self, points: np.ndarray) -> None:
        """More efficient batch insertion."""
        self.points.extend(points)
        self.total_points += len(points)
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        if len(self.points) == 0:
            return

        points_array: NDArray[np.floating[Any]] = np.array(self.points)
        if self.tree_type == "kd":
            self.tree = KDTree(points_array)
        elif self.tree_type == "ball":
            self.tree = BallTree(points_array)
        else:
            raise ValueError(f"Unknown tree type: {self.tree_type}")

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal using k-NN density estimation.
        S(e) ≈ log(V_k(e)) where V_k is the volume of k-ball
        """
        if self.tree is None or len(self.points) < self.k:  # pyright: ignore[reportUnknownMemberType]
            return float("inf")

        k_actual = min(self.k, len(self.points))
        distances, _indices = self.tree.query([point], k=k_actual)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        avg_distance: float = float(np.mean(distances[0]))  # pyright: ignore[reportUnknownArgumentType]

        dim = point.shape[0]
        surprisal_value: float = dim * np.log(avg_distance + 1e-10)

        return surprisal_value
