"""
Base classes for tree-based surprisal estimation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union
from abc import ABC, abstractmethod


@dataclass
class TreeNode:
    """Base node for tree structures."""
    count: int = 0


@dataclass
class LeafNode(TreeNode):
    """Leaf node containing actual points."""
    points: List[np.ndarray] = None

    def __post_init__(self):
        if self.points is None:
            self.points = []
        if self.count == 0:
            self.count = len(self.points)


@dataclass
class InternalNode(TreeNode):
    """Internal node with splitting criterion."""
    left: Union['InternalNode', LeafNode] = None
    right: Union['InternalNode', LeafNode] = None


class SurprisalTree(ABC):
    """Base class for tree-based surprisal estimation."""

    def __init__(self, max_leaf_size: int = 10):
        self.max_leaf_size = max_leaf_size
        self.root = None
        self.total_points = 0

    @abstractmethod
    def insert(self, point: np.ndarray) -> None:
        """Insert a point into the tree."""
        pass

    @abstractmethod
    def surprisal(self, point: np.ndarray) -> float:
        """Compute surprisal for a point."""
        pass

    def batch_insert(self, points: np.ndarray) -> None:
        """Insert multiple points."""
        for point in points:
            self.insert(point)
