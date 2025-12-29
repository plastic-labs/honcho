"""
Cover Tree implementation.
"""

from dataclasses import dataclass, field

import numpy as np

from .base import SurprisalTree, TreeNode


@dataclass
class CoverNode(TreeNode):
    """Node for Cover Tree with point, scale, and children."""

    point: np.ndarray | None = None
    scale: float = 0.0
    children: list["CoverNode"] = field(default_factory=list)


class CoverTree(SurprisalTree):
    """
    Cover Tree implementation with surprisal computation.
    Organizes points hierarchically by scale.
    """

    base: float
    root: CoverNode | None
    total_points: int

    def __init__(self, base: float = 2.0, max_leaf_size: int = 10) -> None:
        super().__init__(max_leaf_size)
        self.base = base
        self.root = None

    def insert(self, point: np.ndarray) -> None:
        if self.root is None:
            self.root = CoverNode(point=point, scale=0.0, count=1)
        else:
            self._insert_recursive(self.root, point, self.root.scale)
        self.total_points += 1

    def _insert_recursive(
        self, node: CoverNode, point: np.ndarray, scale: float
    ) -> None:
        """
        Insert point into cover tree recursively.
        Fixed to ensure proper tree structure and varied paths.
        """
        node.count += 1
        dist = float(np.linalg.norm(node.point - point))

        cover_radius = self.base**scale
        if dist <= cover_radius:
            for child in node.children:
                child_dist = float(np.linalg.norm(child.point - point))
                child_radius = self.base ** (scale - 1)
                if child_dist <= child_radius:
                    self._insert_recursive(child, point, scale - 1)
                    return

            new_child = CoverNode(point=point, scale=scale - 1, count=1)
            node.children.append(new_child)
        else:
            new_scale = scale + 1

            new_root = CoverNode(
                point=node.point, scale=new_scale, count=node.count + 1
            )
            new_root.children = [node]

            new_sibling = CoverNode(point=point, scale=scale, count=1)
            new_root.children.append(new_sibling)

            self.root = new_root

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal based on path through cover tree.
        Uses combination of branch probabilities and distance to final node.
        """
        if self.root is None:
            return float("inf")

        surprisal_value = 0.0
        node: CoverNode = self.root
        scale = self.root.scale
        depth = 0

        while node.children:
            best_child: CoverNode | None = None
            best_dist = float("inf")

            for child in node.children:
                dist = float(np.linalg.norm(child.point - point))
                if dist < best_dist:
                    best_dist = dist
                    best_child = child

            if best_child is None:
                break

            parent_count = node.count
            child_count = best_child.count
            p_branch = child_count / parent_count

            surprisal_value += -np.log(p_branch + 1e-10)

            node = best_child
            scale -= 1
            depth += 1

        dist_to_rep = float(np.linalg.norm(node.point - point))

        dim = len(point)
        distance_surprisal = dim * np.log(dist_to_rep + 0.01)

        return float(surprisal_value + distance_surprisal)
