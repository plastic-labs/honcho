"""
Random Projection Tree implementation.
"""

from dataclasses import dataclass

import numpy as np

from .base import InternalNode, LeafNode, SurprisalTree


@dataclass
class RPInternalNode(InternalNode):
    """Internal node for Random Projection Tree with direction and threshold."""

    direction: np.ndarray | None = None
    threshold: float = 0.0


class RPTree(SurprisalTree):
    """
    Random Projection Tree with surprisal computation.
    Uses random projection directions at each split.
    """

    root: LeafNode | RPInternalNode | None
    total_points: int

    def __init__(self, max_leaf_size: int = 10) -> None:
        super().__init__(max_leaf_size)
        self.root = None

    def insert(self, point: np.ndarray) -> None:
        if self.root is None:
            self.root = LeafNode(points=[point], count=1)
        else:
            self.root = self._insert(self.root, point)
        self.total_points += 1

    def _insert(
        self, node: LeafNode | RPInternalNode, point: np.ndarray
    ) -> LeafNode | RPInternalNode:
        node.count += 1

        if isinstance(node, LeafNode):
            node.points.append(point)
            if len(node.points) > self.max_leaf_size:
                return self._split_leaf(node)
            return node
        else:
            if self._go_left(node, point):
                if node.left is not None:
                    node.left = self._insert_child(node.left, point)
            else:
                if node.right is not None:
                    node.right = self._insert_child(node.right, point)
            return node

    def _insert_child(
        self, child: InternalNode | LeafNode, point: np.ndarray
    ) -> LeafNode | RPInternalNode:
        """Insert into a child node, handling the type narrowing."""
        if isinstance(child, LeafNode | RPInternalNode):
            return self._insert(child, point)
        raise TypeError(f"Unexpected child type: {type(child)}")

    def _split_leaf(self, leaf: LeafNode) -> LeafNode | RPInternalNode:
        """
        Split a leaf using random projection.
        Tries multiple random directions to find a good split.
        """
        points = np.array(leaf.points)

        if len(points) > 1:
            variance = np.var(points, axis=0).sum()
            if variance < 1e-10:
                return leaf

        max_attempts = 5
        for _attempt in range(max_attempts):
            direction = np.random.randn(points.shape[1])
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm

            projections = points @ direction

            if np.std(projections) < 1e-10:
                continue

            threshold = np.median(projections)

            left_mask = projections < threshold
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                proj_min, proj_max = projections.min(), projections.max()
                if proj_max - proj_min < 1e-10:
                    continue
                threshold = (proj_min + proj_max) / 2
                left_mask = projections < threshold
                right_mask = ~left_mask

            left_points = [p for p, m in zip(leaf.points, left_mask, strict=False) if m]
            right_points = [
                p for p, m in zip(leaf.points, right_mask, strict=False) if m
            ]

            if left_points and right_points:
                return RPInternalNode(
                    direction=direction,
                    threshold=float(threshold),
                    left=LeafNode(points=left_points, count=len(left_points)),
                    right=LeafNode(points=right_points, count=len(right_points)),
                    count=leaf.count,
                )

        return leaf

    def _go_left(self, node: RPInternalNode, point: np.ndarray) -> bool:
        """Determine if point should go left. Must match split criterion."""
        return bool((point @ node.direction) < node.threshold)

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal as cumulative log-probability along path.
        S(x) = -log P(path to x) = Σ -log(n_child / n_parent)
        """
        if self.root is None:
            return float("inf")

        surprisal_value = 0.0
        node: LeafNode | RPInternalNode = self.root

        while isinstance(node, RPInternalNode):
            parent_count = node.count

            if self._go_left(node, point) and node.left is not None:
                child_count = node.left.count
                if isinstance(node.left, LeafNode | RPInternalNode):
                    node = node.left
                else:
                    break
            elif node.right is not None:
                child_count = node.right.count
                if isinstance(node.right, LeafNode | RPInternalNode):
                    node = node.right
                else:
                    break
            else:
                break

            p_branch = child_count / parent_count
            surprisal_value += -np.log(p_branch + 1e-10)

        return surprisal_value
