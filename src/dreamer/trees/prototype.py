"""
Prototype-based surprisal using clustering.
"""

import numpy as np
from .base import SurprisalTree


class PrototypeSurprisal(SurprisalTree):
    """
    Prototype-based surprisal using clustering.
    Surprisal proportional to distance from nearest prototype centroid.
    """

    def __init__(self, n_clusters: int = 10):
        super().__init__()
        self.n_clusters = n_clusters
        self.points = []
        self.prototypes = None
        self.clusters_built = False

    def insert(self, point: np.ndarray):
        self.points.append(point)
        self.total_points += 1
        self.clusters_built = False

    def batch_insert(self, points: np.ndarray):
        """More efficient batch insertion."""
        self.points.extend(points)
        self.total_points += len(points)
        self.clusters_built = False

    def _build_clusters(self):
        """Build clusters and identify prototypes."""
        if len(self.points) < self.n_clusters:
            # Not enough points, use all points as prototypes
            self.prototypes = np.array(self.points)
            self.clusters_built = True
            return

        from sklearn.cluster import KMeans

        points_array = np.array(self.points)
        n_clusters_actual = min(self.n_clusters, len(self.points))

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
        kmeans.fit(points_array)

        # Store cluster centers as prototypes
        self.prototypes = kmeans.cluster_centers_
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
            return float('inf')

        # Find distance to nearest prototype
        distances = np.linalg.norm(self.prototypes - point, axis=1)
        min_distance = np.min(distances)

        # Surprisal proportional to distance
        # Add scaling factor to make values comparable
        return min_distance * 10.0
