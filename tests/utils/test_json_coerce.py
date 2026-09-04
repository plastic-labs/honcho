"""Unit tests for the shared JSON narrowing helpers."""

from typing import Any

import pytest

from src.utils.json_coerce import as_dict, as_int, as_list, as_str

MISMATCHED: list[Any] = [None, "text", 7, 1.5, True, [], {}, object()]


@pytest.mark.parametrize(
    ("coerce", "value"),
    [
        (as_dict, {"a": 1}),
        (as_dict, {}),
        (as_list, [1, "two"]),
        (as_list, []),
        (as_str, "text"),
        (as_str, ""),
        (as_int, 7),
        (as_int, 0),
        (as_int, -1),
    ],
)
def test_matching_type_passes_the_value_through(coerce: Any, value: Any) -> None:
    """A match returns the original object, not a copy, so callers can mutate it."""
    assert coerce(value) is value


@pytest.mark.parametrize("coerce", [as_dict, as_list, as_str, as_int])
def test_mismatched_type_returns_none(coerce: Any) -> None:
    matches = {as_dict: dict, as_list: list, as_str: str, as_int: int}[coerce]
    for value in MISMATCHED:
        if isinstance(value, matches):
            continue
        assert coerce(value) is None, f"{coerce.__name__} accepted {value!r}"


@pytest.mark.parametrize("value", [True, False])
def test_as_int_rejects_bool(value: bool) -> None:
    """bool is an int subclass, so a JSON `true` would otherwise become 1.

    Regression: the mock provider's `dimensions` field accepted `true` and
    generated a one-element vector instead of rejecting the request.
    """
    assert as_int(value) is None
