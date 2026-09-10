"""Unit tests for the shared JSON narrowing helpers."""

from collections.abc import Callable
from typing import Any

import pytest

from src.utils.json_coerce import as_dict, as_int, as_list, as_str

Coercer = Callable[[object], Any]

# One representative value per JSON type, reused to build the mismatch matrix.
SAMPLES: list[Any] = [None, "text", 7, 1.5, True, ["a"], {"a": 1}]

COERCERS: list[tuple[Coercer, type]] = [
    (as_dict, dict),
    (as_list, list),
    (as_str, str),
    (as_int, int),
]


def _name(arg: object) -> str:
    return getattr(arg, "__name__", repr(arg))


@pytest.mark.parametrize(
    ("coerce", "value"),
    [
        (as_dict, {"a": 1}),
        (as_dict, {}),
        (as_list, ["a"]),
        (as_list, []),
        (as_str, "text"),
        (as_str, ""),
        (as_int, 7),
        (as_int, 0),
        (as_int, -1),
    ],
    ids=_name,
)
def test_matching_type_passes_the_value_through(coerce: Coercer, value: Any) -> None:
    """A match returns the original object, so callers can mutate it in place.

    `deep_update` relies on this: it recurses into the dict `as_dict` hands back
    and expects the caller's dict to see the writes.

    The empty and zero cases pin narrowing on type rather than truthiness — a
    `return value if value else None` implementation would pass every other row.
    """
    assert coerce(value) is value


@pytest.mark.parametrize(
    ("coerce", "value"),
    [
        (coerce, value)
        for coerce, accepted in COERCERS
        for value in SAMPLES
        if not isinstance(value, accepted)
    ],
    ids=_name,
)
def test_mismatched_type_returns_none(coerce: Coercer, value: Any) -> None:
    assert coerce(value) is None


@pytest.mark.parametrize("value", [True, False])
def test_as_int_rejects_bool(value: bool) -> None:
    """bool is an int subclass, so a JSON `true` would otherwise become 1.

    `True` is filtered out of the mismatch matrix above precisely because it
    passes `isinstance(True, int)`; this is the case that pins the exclusion.
    """
    assert as_int(value) is None
