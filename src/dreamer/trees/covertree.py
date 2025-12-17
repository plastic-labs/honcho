"""
Cover Tree implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from .base import SurprisalTree, TreeNode


class CoverTree(SurprisalTree):
    """
    Cover Tree implementation with surprisal computation.
    Organizes points hierarchically by scale.
    """

    @dataclass
    class CoverNode(TreeNode):
        point: np.ndarray = None
        scale: float = 0.0
        children: List['CoverTree.CoverNode'] = None

        def __post_init__(self):
            if self.children is None:
                self.children = []

    def __init__(self, base: float = 2.0):
        super().__init__()
        self.base = base  # Expansion constant
        self.root = None

    def insert(self, point: np.ndarray):
        if self.root is None:
            self.root = self.CoverNode(point=point, scale=0.0, count=1)
        else:
            self._insert_recursive(self.root, point, self.root.scale)
        self.total_points += 1

    def _insert_recursive(self, node: 'CoverTree.CoverNode', point: np.ndarray, scale: float):
        """
        Insert point into cover tree recursively.
        Fixed to ensure proper tree structure and varied paths.
        """
        node.count += 1
        dist = np.linalg.norm(node.point - point)

        # Check if point is covered at this scale
        cover_radius = self.base ** scale
        if dist <= cover_radius:
            # Try to insert into an existing child that covers this point
            for child in node.children:
                child_dist = np.linalg.norm(child.point - point)
                child_radius = self.base ** (scale - 1)
                if child_dist <= child_radius:
                    self._insert_recursive(child, point, scale - 1)
                    return

            # No existing child covers this point, create new child
            new_child = self.CoverNode(point=point, scale=scale - 1, count=1)
            node.children.append(new_child)
        else:
            # Point not covered, need to expand root scale
            # This is a critical path - we need to create a new root
            new_scale = scale + 1

            # Create new root with old root as child
            new_root = self.CoverNode(point=node.point, scale=new_scale, count=node.count + 1)
            new_root.children = [node]

            # Create new sibling for the incoming point
            new_sibling = self.CoverNode(point=point, scale=scale, count=1)
            new_root.children.append(new_sibling)

            self.root = new_root

    def surprisal(self, point: np.ndarray) -> float:
        """
        Compute surprisal based on path through cover tree.
        Uses combination of branch probabilities and distance to final node.
        """
        if self.root is None:
            return float('inf')

        surprisal_value = 0.0
        node = self.root
        scale = self.root.scale
        depth = 0

        # Traverse down the tree following closest children
        while node.children:
            # Find closest child
            best_child = None
            best_dist = float('inf')

            for child in node.children:
                dist = np.linalg.norm(child.point - point)
                if dist < best_dist:
                    best_dist = dist
                    best_child = child

            if best_child is None:
                break

            # Accumulate surprisal based on branch probability
            parent_count = node.count
            child_count = best_child.count
            p_branch = child_count / parent_count

            # Add probability-based surprisal
            surprisal_value += -np.log(p_branch + 1e-10)

            node = best_child
            scale -= 1
            depth += 1

        # Add distance-based component to distinguish points at same leaf
        # Distance to the closest representative point in the leaf
        dist_to_rep = np.linalg.norm(node.point - point)

        # Scale distance by dimension to get volume-like measure
        dim = len(point)
        distance_surprisal = dim * np.log(dist_to_rep + 0.01)

        # Combine path-based and distance-based surprisal
        return surprisal_value + distance_surprisal
