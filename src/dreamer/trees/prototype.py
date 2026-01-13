"""
Prototype-based surprisal using clustering.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import SurprisalTree


class PrototypeSurprisal(SurprisalTree):
    """
    Prototype-based surprisal using clustering.
    Surprisal proportional to distance from nearest prototype centroid.
    """

    n_clusters: int
    points: list[NDArray[np.floating[Any]]]
    prototypes: NDArray[np.floating[Any]] | None
    clusters_built: bool
    total_points: int
    surprisal_scale: float

    def __init__(
        self,
        n_clusters: int = 10,
        max_leaf_size: int = 10,
        surprisal_scale: float = 10.0,
    ) -> None:
        """
        Initialize the prototype-based surprisal tree.

        Args:
            n_clusters: Number of prototype clusters to form.
            max_leaf_size: Maximum size for leaf nodes (passed to base class).
            surprisal_scale: Multiplier applied to the minimum distance from
                prototypes. Default is 10.0 to normalize raw embedding distances
                (typically in [0, 1]) to a more interpretable surprisal range.
        """
        super().__init__(max_leaf_size)
        self.n_clusters = n_clusters
        self.points = []
        self.prototypes = None
        self.clusters_built = False
        self.surprisal_scale = surprisal_scale

    def insert(self, point: np.ndarray) -> None:
        self.points.append(point)
        self.total_points += 1
        self.clusters_built = False

    def batch_insert(self, points: np.ndarray) -> None:
        """More efficient batch insertion."""
        self.points.extend(points)
        self.total_points += len(points)
        self.clusters_built = False

    def _build_clusters(self) -> None:
        """Build clusters and identify prototypes."""
        if len(self.points) < self.n_clusters:
            self.prototypes = np.array(self.points)
            self.clusters_built = True
            return

        from sklearn.cluster import KMeans

        points_array: NDArray[np.floating[Any]] = np.array(self.points)
        n_clusters_actual = min(self.n_clusters, len(self.points))

        kmeans: KMeans = KMeans(
            n_clusters=n_clusters_actual, random_state=42, n_init="auto"
        )
        kmeans.fit(points_array)  # pyright: ignore[reportUnknownMemberType]

        self.prototypes = kmeans.cluster_centers_  # pyright: ignore[reportUnknownMemberType]
        self.clusters_built = True

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal as distance to nearest prototype.
        Close to prototype = expected = low surprisal
        Far from prototype = surprising = high surprisal
        """
        if not self.clusters_built:
            self._build_clusters()

        if self.prototypes is None or len(self.prototypes) == 0:
            return float("inf")

        distances: NDArray[np.floating[Any]] = np.linalg.norm(
            self.prototypes - point, axis=1
        )
        min_distance: float = float(np.min(distances))

        return min_distance * self.surprisal_scale
