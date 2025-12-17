"""
Random Projection Tree implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Union

from .base import SurprisalTree, InternalNode, LeafNode


class RPTree(SurprisalTree):
    """
    Random Projection Tree with surprisal computation.
    Uses random projection directions at each split.
    """

    @dataclass
    class RPInternalNode(InternalNode):
        direction: np.ndarray = None
        threshold: float = 0.0

    def insert(self, point: np.ndarray):
        if self.root is None:
            self.root = LeafNode(points=[point], count=1)
        else:
            self.root = self._insert(self.root, point)
        self.total_points += 1

    def _insert(self, node: Union[LeafNode, 'RPTree.RPInternalNode'], point: np.ndarray):
        node.count += 1

        if isinstance(node, LeafNode):
            node.points.append(point)
            if len(node.points) > self.max_leaf_size:
                return self._split_leaf(node)
            return node
        else:
            # Internal node: route to appropriate child
            if self._go_left(node, point):
                node.left = self._insert(node.left, point)
            else:
                node.right = self._insert(node.right, point)
            return node

    def _split_leaf(self, leaf: LeafNode) -> Union[LeafNode, 'RPTree.RPInternalNode']:
        """
        Split a leaf using random projection.
        Tries multiple random directions to find a good split.
        """
        points = np.array(leaf.points)

        # Handle degenerate case: single unique point repeated
        if len(points) > 1:
            variance = np.var(points, axis=0).sum()
            if variance < 1e-10:
                # All points are essentially identical
                return leaf

        # Try multiple random directions if needed
        max_attempts = 5
        for attempt in range(max_attempts):
            # Random projection direction
            direction = np.random.randn(points.shape[1])
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue  # Skip degenerate direction
            direction /= norm

            # Project and find median
            projections = points @ direction

            # Check if there's any variance in projections
            if np.std(projections) < 1e-10:
                continue  # All points project to same location, try another direction

            threshold = np.median(projections)

            # Use strict inequality for one side to ensure proper split
            left_mask = projections < threshold
            right_mask = ~left_mask

            # If median creates empty split, try mean of min and max
            if not left_mask.any() or not right_mask.any():
                proj_min, proj_max = projections.min(), projections.max()
                if proj_max - proj_min < 1e-10:
                    continue  # No spread, try another direction
                threshold = (proj_min + proj_max) / 2
                left_mask = projections < threshold
                right_mask = ~left_mask

            left_points = [p for p, m in zip(leaf.points, left_mask) if m]
            right_points = [p for p, m in zip(leaf.points, right_mask) if m]

            # If we successfully split, create internal node
            if left_points and right_points:
                return self.RPInternalNode(
                    direction=direction,
                    threshold=threshold,
                    left=LeafNode(points=left_points, count=len(left_points)),
                    right=LeafNode(points=right_points, count=len(right_points)),
                    count=leaf.count
                )

        # If all attempts fail (all points are identical or degenerate), return leaf as-is
        # This is acceptable - leaf will just be larger than max_leaf_size
        return leaf

    def _go_left(self, node: 'RPTree.RPInternalNode', point: np.ndarray) -> bool:
        """Determine if point should go left. Must match split criterion."""
        return (point @ node.direction) < node.threshold

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal as cumulative log-probability along path.
        S(x) = -log P(path to x) = Σ -log(n_child / n_parent)
        """
        if self.root is None:
            return float('inf')

        surprisal_value = 0.0
        node = self.root

        while isinstance(node, self.RPInternalNode):
            parent_count = node.count

            if self._go_left(node, point):
                child_count = node.left.count
                node = node.left
            else:
                child_count = node.right.count
                node = node.right

            # Probability of taking this branch
            p_branch = child_count / parent_count
            surprisal_value += -np.log(p_branch + 1e-10)

        return surprisal_value
