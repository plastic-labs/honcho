"""Unit tests for filter condition building."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, cast, get_args

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import psycopg as psycopg_dialect

from src.exceptions import FilterError
from src.models import Document, Message, Peer, Session
from src.utils.filter import apply_filter
from src.utils.types import DocumentLevel


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
    # Assert on the WHERE clause specifically: internal_metadata is in the
    # SELECT projection either way, so checking the whole statement passes even
    # when no condition was applied at all.
    assert stmt.whereclause is not None
    assert "internal_metadata" in str(stmt.whereclause)
    compiled = stmt.compile(dialect=postgresql.dialect())
    assert {"kind": "note"} in [bind.value for bind in compiled.binds.values()]


def test_numeric_operand_keeps_integer_precision():
    """float() rounds anything past 2**53, silently shifting the comparison."""
    big = 2**53 + 1
    stmt = apply_filter(select(Message), Message, {"token_count": {"gt": big}})
    compiled = stmt.compile(dialect=postgresql.dialect())
    assert big in [bind.value for bind in compiled.binds.values()]


def test_fractional_operand_on_integer_column_is_not_truncated():
    """Coercing to the column's int type would turn `lt 5.5` into `lt 5`."""
    stmt = apply_filter(select(Message), Message, {"token_count": {"lt": 5.5}})
    compiled = stmt.compile(dialect=postgresql.dialect())
    assert 5.5 in [bind.value for bind in compiled.binds.values()]


def test_integer_operand_binds_without_an_integer_cast():
    """A plain int bind renders `::INTEGER`, so any value past int4 fails at
    execute time with "integer out of range" even when the comparison is
    meaningful. Decimal renders no cast, which is what float() used to do."""
    stmt = apply_filter(select(Message), Message, {"token_count": {"gt": 2**31}})
    assert "::INTEGER" not in str(stmt.compile(dialect=psycopg_dialect.dialect()))


def test_in_list_on_numeric_column_handles_out_of_range_values():
    stmt = apply_filter(select(Message), Message, {"token_count": {"in": [1, 2**31]}})
    assert "::INTEGER" not in str(stmt.compile(dialect=psycopg_dialect.dialect()))


def test_in_list_on_numeric_column_rejects_garbage():
    with pytest.raises(FilterError):
        apply_filter(select(Message), Message, {"token_count": {"in": [1, "nope"]}})


@pytest.mark.parametrize(
    "filters",
    [
        {"is_active": True},
        {"is_active": False},
        {"is_active": {"ne": True}},
        {"is_active": {"in": [True, False]}},
    ],
)
def test_bool_column_accepts_native_booleans(filters: dict[str, Any]):
    """bool subclasses int, so a boolean column would otherwise be coerced to 1
    and rejected by Postgres as `boolean <> integer`."""
    stmt = apply_filter(select(Session), Session, filters)
    assert stmt.whereclause is not None
    assert "is_active" in str(stmt.whereclause)


@pytest.mark.parametrize(
    "filters",
    [
        {"is_active": "true"},
        {"is_active": "false"},
        {"is_active": 1},
        {"is_active": {"ne": "true"}},
    ],
)
def test_bool_column_rejects_non_boolean_operands(filters: dict[str, Any]):
    """These bind as VARCHAR/INTEGER against a boolean column, which Postgres
    rejects at execute time — a 422 is the honest answer, not a 500."""
    with pytest.raises(FilterError):
        apply_filter(select(Session), Session, filters)


def test_numeric_string_operand_stays_exact():
    stmt = apply_filter(select(Message), Message, {"token_count": {"gt": "5"}})
    compiled = stmt.compile(dialect=postgresql.dialect())
    assert 5 in [bind.value for bind in compiled.binds.values()]


def test_ne_none_on_scalar_column_is_not_null():
    """Regression: float(None) raised TypeError, which the ValueError handler
    missed -> unhandled 500."""
    stmt = apply_filter(select(Document), Document, {"session_id": {"ne": None}})
    assert "session_name IS NOT NULL" in str(stmt)


def test_ne_string_on_text_column_compares_as_string():
    """Regression: numeric operators float()-cast on every column type, so a
    string inequality on a text column was rejected as a bad number."""
    stmt = apply_filter(select(Document), Document, {"session_id": {"ne": "abc"}})
    assert "session_name IS DISTINCT FROM" in str(stmt)


def test_numeric_operator_still_validates_on_numeric_column():
    with pytest.raises(FilterError):
        apply_filter(select(Message), Message, {"token_count": {"gt": "nope"}})


def test_ne_none_on_numeric_column_is_not_null():
    stmt = apply_filter(select(Message), Message, {"token_count": {"ne": None}})
    assert "token_count IS NOT NULL" in str(stmt)


def test_null_operand_on_non_ne_operator_raises():
    with pytest.raises(FilterError):
        apply_filter(select(Message), Message, {"token_count": {"gt": None}})


def test_enum_column_rejects_an_unknown_value():
    """An invalid level silently matched nothing, which reads as "no results"
    rather than "you sent a value that cannot exist"."""
    with pytest.raises(FilterError):
        apply_filter(select(Document), Document, {"level": "banana"})


@pytest.mark.parametrize("level", get_args(DocumentLevel))
def test_enum_column_accepts_every_declared_level(level: str):
    """Derived from the Literal, so a new level (e.g. "abduction") is covered
    here the moment it is declared — no second list to keep in sync."""
    stmt = apply_filter(select(Document), Document, {"level": level})
    assert stmt.whereclause is not None


def test_empty_in_list_matches_nothing_rather_than_everything():
    """Dropping an empty IN would widen the query to every row. Session scoping
    relies on an empty allowlist failing closed."""
    stmt = apply_filter(select(Document), Document, {"session_id": {"in": []}})
    assert stmt.whereclause is not None
    assert "IN" in str(stmt.whereclause)


@pytest.mark.parametrize(
    "filters",
    [
        {"created_at": "2026-01-01"},
        {"created_at": {"gte": "2026-01-01"}},
        {"token_count": "5"},
        {"token_count": {"gt": "5"}},
    ],
)
def test_equality_and_comparison_paths_coerce_alike(filters: dict[str, Any]):
    """The two paths had different rules: comparison operators parsed datetimes
    and coerced numbers, bare equality bound the raw string and 500'd on
    `timestamp with time zone = character varying`."""
    stmt = apply_filter(select(Message), Message, filters)
    assert stmt.whereclause is not None


@pytest.mark.parametrize(
    ("model", "filters"),
    [
        (Message, {"token_count": {"contains": 5}}),  # integer ~~* text
        (Message, {"created_at": {"contains": "x"}}),  # timestamptz ~~* text
        (Document, {"metadata": {"gte": 5}}),  # jsonb >= integer
        (Document, {"metadata": {"contains": "x"}}),  # jsonb ~~* text
        (Document, {"source_ids": "abc"}),  # jsonb = character varying
        (Document, {"session_id": 5}),  # text = integer
        (Document, {"session_id": True}),  # text = boolean
        (Message, {"created_at": 5}),  # timestamptz = integer
        (Document, {"embedding": 5}),  # no python_type at all
    ],
)
def test_incompatible_operand_types_are_rejected(model: Any, filters: dict[str, Any]):
    """Each of these compiled cleanly and failed in Postgres as
    `operator does not exist: <coltype> <op> <operandtype>`."""
    with pytest.raises(FilterError):
        apply_filter(select(model), model, filters)


def _where(model: Any, filters: dict[str, Any]) -> str:
    stmt = apply_filter(select(model), model, filters)
    # Without this, a dropped condition makes split("WHERE") return the whole
    # statement, so a filter that silently widened could still pass the asserts.
    assert stmt.whereclause is not None
    return str(stmt.compile(dialect=psycopg_dialect.dialect())).split("WHERE")[-1]


def test_not_is_null_safe():
    """`NOT (col = v)` is NULL when col is NULL, so plain negation drops rows
    whose column is unset — even though an unset column is not `v`."""
    where = _where(Document, {"NOT": [{"session_id": "abc"}]})
    assert "IS NOT true" in where


def test_ne_is_null_safe():
    where = _where(Document, {"session_id": {"ne": "abc"}})
    assert "IS DISTINCT FROM" in where


def test_nested_metadata_ne_string_is_null_safe():
    """`ne` on a JSONB metadata key went through the operator map as plain <>,
    unlike the scalar path, so a row missing that key was silently dropped."""
    where = _where(Document, {"metadata": {"priority": {"ne": "high"}}})
    assert "IS DISTINCT FROM" in where
    assert "!=" not in where


def test_nested_metadata_ne_numeric_is_null_safe():
    where = _where(Document, {"metadata": {"score": {"ne": 5}}})
    assert "IS DISTINCT FROM" in where
    assert "!=" not in where


def test_not_is_null_safe_over_a_compound_condition():
    """Negation has to survive nesting, not just single comparisons."""
    where = _where(
        Document, {"NOT": [{"AND": [{"session_id": "a"}, {"level": "explicit"}]}]}
    )
    assert "IS NOT true" in where
    assert "AND" in where


def test_not_is_null_safe_over_contains():
    where = _where(Document, {"NOT": [{"session_id": {"contains": "x"}}]})
    assert "ILIKE" in where
    assert "IS NOT true" in where


def test_ne_null_still_renders_is_not_null():
    """The null operand is intercepted before the operator dispatch, so this
    path is unchanged by null-safe `ne`."""
    assert "IS NOT NULL" in _where(Document, {"session_id": {"ne": None}})


@pytest.mark.parametrize(
    ("model", "filters"),
    [
        (Document, {"session_id": None}),  # text
        (Message, {"token_count": None}),  # numeric
        (Session, {"is_active": None}),  # boolean
        (Message, {"created_at": None}),  # datetime
        (Document, {"metadata": None}),  # JSONB
        (Document, {"source_ids": None}),  # JSONB via the raw-key fallback
    ],
)
def test_bare_null_is_a_null_check(model: Any, filters: dict[str, Any]):
    """Regression: every operand routes through _coerce_operand, which accepts
    no None on any column type, so bare null raised FilterError instead of
    matching the unset rows it names."""
    assert "IS NULL" in _where(model, filters)


def test_bare_null_agrees_with_negation():
    """`NOT [{col: null}]` and `{col: {ne: null}}` must select the same rows.
    `x IS NULL` never evaluates to NULL, so `(x IS NULL) IS NOT true` is exactly
    `x IS NOT NULL` — the three forms have to stay in agreement."""
    assert "IS NOT NULL" in _where(Document, {"session_id": {"ne": None}})
    negated = _where(Document, {"NOT": [{"session_id": None}]})
    assert "IS NULL" in negated
    assert "IS NOT true" in negated


def test_dict_on_raw_key_jsonb_column_is_equality():
    """Document falls back to raw column names, so a JSONB column outside
    JSONB_COLUMNS reaches _build_field_condition's dict branch. It compares as
    equality rather than containment — unlike `metadata`."""
    where = _where(Document, {"source_ids": {"kind": "note"}})
    assert "source_ids =" in where


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"session_id": "abc"}, "session_name ="),
        ({"session_id": {"contains": "x"}}, "ILIKE"),
    ],
)
def test_positive_predicates_are_unchanged(filters: dict[str, Any], expected: str):
    """A NULL column does not equal or contain anything, so positive predicates
    correctly exclude those rows and must keep their plain operators."""
    where = _where(Document, filters)
    assert expected in where
    assert "IS NOT true" not in where
    assert "IS DISTINCT FROM" not in where


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
