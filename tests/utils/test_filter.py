"""Unit tests for filter condition building."""

import pytest
from sqlalchemy import select

from src.exceptions import FilterError
from src.models import Document, Message
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
