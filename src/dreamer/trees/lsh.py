"""
Locality-Sensitive Hashing based surprisal estimation.
"""

import numpy as np
from .base import SurprisalTree


class LSHSurprisal(SurprisalTree):
    """
    Locality-Sensitive Hashing based surprisal estimation.
    O(1) operations using hash collision frequency as density proxy.
    """

    def __init__(self, num_tables: int = 10, num_bits: int = 8):
        super().__init__()
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.tables = [dict() for _ in range(num_tables)]
        self.hash_directions = []
        self.initialized = False

    def _initialize_hash_functions(self, dim: int):
        """Initialize random projection directions for LSH."""
        if not self.initialized:
            for _ in range(self.num_tables):
                # Random projection directions
                directions = np.random.randn(self.num_bits, dim)
                directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
                self.hash_directions.append(directions)
            self.initialized = True

    def _hash_vector(self, point: np.ndarray, table_idx: int) -> int:
        """Hash a vector using random projections."""
        projections = self.hash_directions[table_idx] @ point
        # Convert to binary hash
        binary = (projections > 0).astype(int)
        # Convert binary to integer
        hash_val = int(''.join(map(str, binary)), 2)
        return hash_val

    def insert(self, point: np.ndarray):
        # Initialize on first insert
        if not self.initialized:
            self._initialize_hash_functions(len(point))

        for i, table in enumerate(self.tables):
            bucket = self._hash_vector(point, i)
            table[bucket] = table.get(bucket, 0) + 1

        self.total_points += 1

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal using hash collision frequency.
        High collision = low surprisal (common pattern)
        Low collision = high surprisal (rare pattern)
        """
        if self.total_points == 0 or not self.initialized:
            return float('inf')

        counts = []
        for i, table in enumerate(self.tables):
            bucket = self._hash_vector(point, i)
            count = table.get(bucket, 0)
            counts.append(count)

        # Average density estimate across tables
        avg_count = np.mean(counts)
        avg_density = avg_count / self.total_points

        # Surprisal = -log(density)
        return -np.log(avg_density + 1e-10)
