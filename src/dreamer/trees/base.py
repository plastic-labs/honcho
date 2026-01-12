"""
Base classes for tree-based surprisal estimation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TreeNode:
    """Base node for tree structures."""

    count: int = 0


@dataclass
class LeafNode(TreeNode):
    """Leaf node containing actual points."""

    points: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.count == 0:
            self.count: int = len(self.points)


@dataclass
class InternalNode(TreeNode):
    """Internal node with splitting criterion."""

    left: "InternalNode | LeafNode | None" = None
    right: "InternalNode | LeafNode | None" = None


class SurprisalTree(ABC):
    """
    Abstract base class for tree-based surprisal estimation.

    Subclasses implement different spatial indexing strategies.
    Not all implementations use a traditional tree structure.
    """

    max_leaf_size: int
    total_points: int

    def __init__(self, max_leaf_size: int = 10) -> None:
        self.max_leaf_size = max_leaf_size
        self.total_points = 0

    @abstractmethod
    def insert(self, point: np.ndarray) -> None:
        """Insert a point into the structure."""

    @abstractmethod
    def surprisal(self, point: np.ndarray) -> float:
        """Compute surprisal for a point."""

    def batch_insert(self, points: np.ndarray) -> None:
        """Insert multiple points."""
        for point in points:
            self.insert(point)
