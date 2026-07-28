"""Unit tests for filter condition building."""

import pytest
from sqlalchemy import select

from src.exceptions import FilterError
from src.models import Document
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
