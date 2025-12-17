"""
Graph-theoretic surprisal using k-NN graph and random walk.
"""

import numpy as np
from .base import SurprisalTree


class GraphSurprisal(SurprisalTree):
    """
    Graph-theoretic surprisal using k-NN graph and random walk.
    Surprisal based on stationary distribution of random walk.
    """

    def __init__(self, k: int = 5, max_iter: int = 100):
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.points = []
        self.stationary_dist = None
        self.graph_built = False

    def insert(self, point: np.ndarray):
        self.points.append(point)
        self.total_points += 1
        self.graph_built = False  # Need to rebuild graph

    def batch_insert(self, points: np.ndarray):
        """More efficient batch insertion."""
        self.points.extend(points)
        self.total_points += len(points)
        self.graph_built = False

    def _build_graph_and_compute_stationary(self):
        """Build k-NN graph and compute stationary distribution."""
        if len(self.points) < 2:
            return

        from sklearn.neighbors import NearestNeighbors

        points_array = np.array(self.points)
        k_actual = min(self.k, len(self.points) - 1)

        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k_actual + 1, algorithm='auto').fit(points_array)
        distances, indices = nbrs.kneighbors(points_array)

        # Build transition matrix for random walk
        n = len(self.points)
        transition = np.zeros((n, n))

        for i in range(n):
            # Connect to k nearest neighbors (excluding self)
            neighbors = indices[i, 1:]  # Skip first (self)
            for j in neighbors:
                transition[i, j] = 1.0

        # Normalize rows to get probability distribution
        row_sums = transition.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition = transition / row_sums

        # Compute stationary distribution via power iteration
        stationary = np.ones(n) / n
        for _ in range(self.max_iter):
            new_stationary = transition.T @ stationary
            if np.allclose(new_stationary, stationary, atol=1e-6):
                break
            stationary = new_stationary

        # Normalize
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
            return float('inf')

        # Find nearest point in graph
        from sklearn.neighbors import NearestNeighbors
        points_array = np.array(self.points)
        nbrs = NearestNeighbors(n_neighbors=1).fit(points_array)
        _, indices = nbrs.kneighbors([point])
        nearest_idx = indices[0][0]

        # Get stationary probability for nearest point
        prob = self.stationary_dist[nearest_idx]

        return -np.log(prob + 1e-10)
