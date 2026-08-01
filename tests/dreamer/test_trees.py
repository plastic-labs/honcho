import pytest

from src.dreamer.trees import (
    CoverTree,
    GraphSurprisal,
    LSHSurprisal,
    PrototypeSurprisal,
    RPTree,
    SklearnTreeWrapper,
    SurprisalTree,
    create_tree,
)

ALL_TREE_TYPES = [
    "kdtree",
    "balltree",
    "rptree",
    "covertree",
    "lsh",
    "graph",
    "prototype",
]

EXPECTED_CLASS = {
    "kdtree": SklearnTreeWrapper,
    "balltree": SklearnTreeWrapper,
    "rptree": RPTree,
    "covertree": CoverTree,
    "lsh": LSHSurprisal,
    "graph": GraphSurprisal,
    "prototype": PrototypeSurprisal,
}


@pytest.mark.parametrize("tree_type", ALL_TREE_TYPES)
def test_create_tree_accepts_uniform_k_kwarg(tree_type: str):
    tree = create_tree(tree_type=tree_type, k=5)
    assert isinstance(tree, SurprisalTree)
    assert isinstance(tree, EXPECTED_CLASS[tree_type])


@pytest.mark.parametrize("tree_type", ALL_TREE_TYPES)
def test_create_tree_without_k(tree_type: str):
    """The factory should also work when no ``k`` is supplied."""
    tree = create_tree(tree_type=tree_type)
    assert isinstance(tree, EXPECTED_CLASS[tree_type])


def test_create_tree_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown tree type"):
        create_tree(tree_type="not_a_tree")
