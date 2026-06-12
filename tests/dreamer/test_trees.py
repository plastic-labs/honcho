"""Tests for the create_tree factory function."""

import numpy as np
import pytest

from src.dreamer.trees import create_tree

ALL_TREE_TYPES = ["rptree", "kdtree", "balltree", "covertree", "lsh", "graph", "prototype"]


@pytest.mark.parametrize("tree_type", ALL_TREE_TYPES)
def test_create_tree_all_types_with_default_k(tree_type: str) -> None:
    """create_tree() should not crash for any tree type when k is passed conditionally."""
    # KNN-based types accept k; non-KNN types should not receive it.
    if tree_type in ("kdtree", "balltree", "graph"):
        tree = create_tree(tree_type=tree_type, k=5)
    else:
        tree = create_tree(tree_type=tree_type)

    # Every tree must expose batch_insert and surprisal.
    embeddings = np.random.randn(20, 8).astype(np.float32)
    tree.batch_insert(embeddings)
    score = tree.surprisal(embeddings[0])
    assert isinstance(score, float)
    assert np.isfinite(score)
