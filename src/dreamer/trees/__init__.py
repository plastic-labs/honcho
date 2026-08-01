"""
Tree-based structures for computing surprisal from embeddings.
Each tree computes surprisal based on the path to a point in the tree.
"""

from typing import Any

from .base import InternalNode, LeafNode, SurprisalTree, TreeNode
from .covertree import CoverNode, CoverTree
from .graph import GraphSurprisal
from .lsh import LSHSurprisal
from .prototype import PrototypeSurprisal
from .rptree import RPInternalNode, RPTree
from .sklearn_wrapper import SklearnTreeWrapper


def create_tree(tree_type: str, **kwargs: Any) -> SurprisalTree:
    """
    Factory function to create different tree types.

    Args:
        tree_type: Type of tree to create ('rptree', 'kdtree', 'balltree',
                   'covertree', 'lsh', 'graph', 'prototype')
        **kwargs: Additional arguments passed to tree constructor

    Returns:
        An instance of the specified tree type

    Raises:
        ValueError: If tree_type is not recognized
    """
    # `surprisal.py` calls this factory with a uniform `k=settings.DREAM.SURPRISAL.TREE_K` kwarg for every tree type,
    # but `k` is only meaningful for the kNN-based trees (kdtree, balltree, graph).
    # The other 4 use different tunables and raise TypeError if `k` is passed.
    # Drop it here so the factory accepts a uniform kwargs dict.

    trees_without_k = {"rptree", "covertree", "lsh", "prototype"}
    if tree_type in trees_without_k:
        kwargs.pop("k", None)

    if tree_type == "rptree":
        return RPTree(**kwargs)
    elif tree_type == "kdtree":
        return SklearnTreeWrapper(tree_type="kd", **kwargs)
    elif tree_type == "balltree":
        return SklearnTreeWrapper(tree_type="ball", **kwargs)
    elif tree_type == "covertree":
        return CoverTree(**kwargs)
    elif tree_type == "lsh":
        return LSHSurprisal(**kwargs)
    elif tree_type == "graph":
        return GraphSurprisal(**kwargs)
    elif tree_type == "prototype":
        return PrototypeSurprisal(**kwargs)
    else:
        raise ValueError(f"Unknown tree type: {tree_type}")


__all__ = [
    "create_tree",
    "CoverNode",
    "CoverTree",
    "GraphSurprisal",
    "InternalNode",
    "LeafNode",
    "LSHSurprisal",
    "PrototypeSurprisal",
    "RPInternalNode",
    "RPTree",
    "SklearnTreeWrapper",
    "SurprisalTree",
    "TreeNode",
]
