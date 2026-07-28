"""Unit tests for filter condition building."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, cast

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB

from src.exceptions import FilterError
from src.models import Document, Message, Peer, Session
from src.utils.filter import apply_filter


def test_unknown_operator_dict_on_scalar_column_raises():
    """An unrecognized operator dict must 422, not reach the driver as a 500.

    Regression: {"session_id": {"operator": "null"}} compiled to
    `session_name = %(param)s` with a dict bind, which psycopg rejected with
    "cannot adapt type 'dict'" -> unhandled 500.
    """
    with pytest.raises(FilterError):
        apply_filter(select(Document), Document, {"session_id": {"operator": "null"}})


def test_known_operator_dict_on_scalar_column_still_works():
    stmt = apply_filter(
        select(Document), Document, {"session_id": {"in": ["s1", "s2"]}}
    )
    assert "session_name IN" in str(stmt).replace("documents.", "")


def test_dict_on_jsonb_column_still_works():
    stmt = apply_filter(select(Document), Document, {"metadata": {"kind": "note"}})
    assert "internal_metadata" in str(stmt)


def test_ne_none_on_scalar_column_is_not_null():
    """Regression: float(None) raised TypeError, which the ValueError handler
    missed -> unhandled 500."""
    stmt = apply_filter(select(Document), Document, {"session_id": {"ne": None}})
    assert "session_name IS NOT NULL" in str(stmt)


def test_ne_string_on_text_column_compares_as_string():
    """Regression: numeric operators float()-cast on every column type, so a
    string inequality on a text column was rejected as a bad number."""
    stmt = apply_filter(select(Document), Document, {"session_id": {"ne": "abc"}})
    assert "session_name !=" in str(stmt)


def test_numeric_operator_still_validates_on_numeric_column():
    with pytest.raises(FilterError):
        apply_filter(select(Message), Message, {"token_count": {"gt": "nope"}})


def test_ne_none_on_numeric_column_is_not_null():
    stmt = apply_filter(select(Message), Message, {"token_count": {"ne": None}})
    assert "token_count IS NOT NULL" in str(stmt)


def test_null_operand_on_non_ne_operator_raises():
    with pytest.raises(FilterError):
        apply_filter(select(Message), Message, {"token_count": {"gt": None}})


# --- Invariants over the whole DSL -------------------------------------------
#
# The filter body is arbitrary client JSON. Enumerating bad shapes one at a time
# is endless, so these two tests assert the properties that make any unhandled
# shape a 422 instead of a 500, and fail on the next shape nobody thought of.

_OPERANDS: list[Any] = [
    None,
    True,
    False,
    0,
    -1,
    1.5,
    "",
    "abc",
    "*",
    [],
    [None],
    [[1]],
    [{"a": 1}],
    {},
    {"operator": "null"},
    {"ne": None},
    {"ne": {"a": 1}},
    {"ne": [1]},
    {"in": None},
    {"in": "abc"},
    {"in": [{"a": 1}]},
    {"in": [[1]]},
    {"gt": {}},
    {"gt": []},
    {"gt": True},
    {"contains": None},
    {"contains": {"a": 1}},
    {"lt": [1, 2]},
]

_COLUMNS: dict[Any, list[str]] = {
    Document: ["session_id", "workspace_id", "metadata", "level", "source_ids", "id"],
    Message: ["session_id", "peer_id", "token_count", "created_at", "metadata"],
    Session: ["id", "is_active", "created_at", "configuration"],
    Peer: ["id", "created_at", "metadata"],
}

_MALFORMED: list[dict[str, Any]] = [
    {"AND": "notalist"},
    {"AND": [None]},
    {"AND": [[]]},
    {"AND": [1]},
    {"OR": [None]},
    {"OR": [1]},
    {"NOT": None},
    {"NOT": [None]},
    {"unknown_column": 1},
    {"metadata": None},
]


def _filter_shapes() -> list[tuple[Any, dict[str, Any]]]:
    shapes: list[tuple[Any, dict[str, Any]]] = []
    for model, columns in _COLUMNS.items():
        for column in columns:
            for operand in _OPERANDS:
                leaf = {column: operand}
                shapes.append((model, leaf))
                shapes.append((model, {"AND": [leaf]}))
                shapes.append((model, {"NOT": [leaf]}))
        shapes.extend((model, bad) for bad in _MALFORMED)
    return shapes


def test_every_filter_shape_either_compiles_or_raises_filter_error():
    """No filter body may escape as anything other than a compiled statement or
    a FilterError. Anything else reaches the client as an unhandled 500."""
    escaped: list[tuple[str, dict[str, Any], str]] = []
    for model, filters in _filter_shapes():
        try:
            str(apply_filter(select(model), model, filters))
        except FilterError:
            pass
        except Exception as exc:  # pragma: no cover - failure path
            escaped.append((model.__name__, filters, type(exc).__name__))
    assert not escaped, f"non-FilterError escapes: {escaped[:10]}"


def test_no_non_scalar_value_is_bound_to_a_scalar_column():
    """A dict or list bound to a non-JSONB parameter compiles cleanly and then
    fails in psycopg at execute time — the original 500. Nothing may reach that
    state, including non-scalars nested inside an `in` list."""
    offenders: list[tuple[str, dict[str, Any], str]] = []
    for model, filters in _filter_shapes():
        try:
            stmt = apply_filter(select(model), model, filters)
        except FilterError:
            continue
        compiled = stmt.compile(dialect=postgresql.dialect())
        for bind in compiled.binds.values():
            if isinstance(bind.type, JSONB):
                continue
            value: Any = bind.value
            # An expanding IN bind holds the list itself; check its elements.
            elements = cast(
                "Sequence[Any]", value if isinstance(value, list | tuple) else [value]
            )
            for element in elements:
                if element is not None and not isinstance(
                    element, str | bool | int | float | Decimal | datetime
                ):
                    offenders.append((model.__name__, filters, repr(element)[:40]))
    assert not offenders, f"non-scalar bound to scalar column: {offenders[:10]}"
