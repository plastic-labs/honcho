"""
Base classes for tree-based surprisal estimation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def sklearn_import_error(tree_types: str, exc: ImportError) -> RuntimeError:
    """Build an actionable error for a tree type that needs scikit-learn.

    scikit-learn lives behind Honcho's optional ``surprisal`` extra: it drags in
    scipy (~199MB installed on Linux) to serve a feature that is disabled by
    default (``DREAM.SURPRISAL.ENABLED``).
    """
    return RuntimeError(
        f"DREAM.SURPRISAL.TREE_TYPE is set to {tree_types}, which requires "
        + "scikit-learn, but the package could not be imported. Install Honcho's "
        + "'surprisal' extra (for example, `uv sync --extra surprisal`), or pick a "
        + "TREE_TYPE with no scikit-learn dependency ('rptree', 'covertree', 'lsh'). "
        + f"Original import error: {exc}"
    )


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
