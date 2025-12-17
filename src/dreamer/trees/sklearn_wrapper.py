"""
Wrapper for sklearn's KDTree and BallTree.
"""

import numpy as np
from .base import SurprisalTree


class SklearnTreeWrapper(SurprisalTree):
    """
    Wrapper for sklearn's KDTree and BallTree with surprisal computation.
    Uses density estimation via k-nearest neighbors.
    """

    def __init__(self, tree_type='kd', k=5):
        super().__init__()
        self.tree_type = tree_type
        self.k = k
        self.points = []
        self.tree = None

    def insert(self, point: np.ndarray):
        self.points.append(point)
        self.total_points += 1
        # Rebuild tree (expensive, but sklearn trees don't support incremental building)
        self._rebuild_tree()

    def batch_insert(self, points: np.ndarray):
        """More efficient batch insertion."""
        self.points.extend(points)
        self.total_points += len(points)
        self._rebuild_tree()

    def _rebuild_tree(self):
        if len(self.points) == 0:
            return

        from sklearn.neighbors import KDTree, BallTree

        points_array = np.array(self.points)
        if self.tree_type == 'kd':
            self.tree = KDTree(points_array)
        elif self.tree_type == 'ball':
            self.tree = BallTree(points_array)
        else:
            raise ValueError(f"Unknown tree type: {self.tree_type}")

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal using k-NN density estimation.
        S(e) ≈ log(V_k(e)) where V_k is the volume of k-ball
        """
        if self.tree is None or len(self.points) < self.k:
            return float('inf')

        # Find k nearest neighbors
        k_actual = min(self.k, len(self.points))
        distances, _ = self.tree.query([point], k=k_actual)

        # Surprisal proportional to log of average distance to k-NN
        # (approximates log of volume)
        avg_distance = np.mean(distances[0])

        # Add dimension factor for volume scaling
        dim = point.shape[0]
        surprisal_value = dim * np.log(avg_distance + 1e-10)

        return surprisal_value
