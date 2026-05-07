"""Unit tests for FTS query routing branches in search.py.

These tests verify that the correct PostgreSQL tsquery function is selected
based on FULLTEXT_USE_WEBSEARCH and query content, without requiring a live DB.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from src import models
from src.exceptions import ValidationException
from src.utils.search import (
    _fulltext_search,
    _fulltext_search_documents,
    search,
)


def _compile_sql(stmt) -> str:
    """Compile a SQLAlchemy statement to a lowercase SQL string for inspection."""
    return str(stmt.compile(dialect=postgresql.dialect())).lower()


def _make_mock_db() -> tuple[AsyncMock, list]:
    """Return (mock_db, captured_stmts). Each db.execute call appends its stmt."""
    captured: list = []
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    db = AsyncMock()

    async def _capture(stmt, *args, **kwargs):
        captured.append(stmt)
        return mock_result

    db.execute.side_effect = _capture
    return db, captured


# ---------------------------------------------------------------------------
# _fulltext_search (messages) — three branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fulltext_search_messages_websearch_uses_websearch_to_tsquery():
    """Branch A: FULLTEXT_USE_WEBSEARCH=True → websearch_to_tsquery in query."""
    db, captured = _make_mock_db()
    stmt = select(models.Message)

    with patch("src.utils.search.settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH", True):
        await _fulltext_search(db, "test query", stmt, limit=10)

    assert len(captured) == 1
    sql = _compile_sql(captured[0])
    assert "websearch_to_tsquery" in sql
    assert "plainto_tsquery" not in sql


@pytest.mark.asyncio
async def test_fulltext_search_messages_unsafe_chars_uses_ilike_only():
    """Branch B: FULLTEXT_USE_WEBSEARCH=False + unsafe chars → ILIKE, no tsquery."""
    db, captured = _make_mock_db()
    stmt = select(models.Message)

    with patch("src.utils.search.settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH", False):
        # "@" triggers _PLAINTO_TSQUERY_UNSAFE_CHARS
        await _fulltext_search(db, "user@domain.com", stmt, limit=10)

    assert len(captured) == 1
    sql = _compile_sql(captured[0])
    assert "plainto_tsquery" not in sql
    assert "websearch_to_tsquery" not in sql
    assert "ilike" in sql
    # Pure ILIKE path omits ts_rank ordering
    assert "ts_rank" not in sql


@pytest.mark.asyncio
async def test_fulltext_search_messages_safe_chars_uses_plainto_tsquery():
    """Branch C: FULLTEXT_USE_WEBSEARCH=False + safe chars → plainto_tsquery."""
    db, captured = _make_mock_db()
    stmt = select(models.Message)

    with patch("src.utils.search.settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH", False):
        await _fulltext_search(db, "simple safe query", stmt, limit=10)

    assert len(captured) == 1
    sql = _compile_sql(captured[0])
    assert "plainto_tsquery" in sql
    assert "websearch_to_tsquery" not in sql


@pytest.mark.asyncio
async def test_fulltext_search_messages_ts_rank_ordered_descending():
    """_build_fts_ranked_query sorts ts_rank DESC (highest relevance first)."""
    db, captured = _make_mock_db()
    stmt = select(models.Message)

    with patch("src.utils.search.settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH", False):
        await _fulltext_search(db, "keyword search", stmt, limit=5)

    sql = _compile_sql(captured[0])
    # DESC must appear after ts_rank in the ORDER BY clause
    ts_rank_pos = sql.find("ts_rank")
    desc_pos = sql.find("desc", ts_rank_pos)
    assert ts_rank_pos != -1
    assert desc_pos != -1 and desc_pos > ts_rank_pos


# ---------------------------------------------------------------------------
# _fulltext_search_documents — three branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fulltext_search_documents_websearch_uses_websearch_to_tsquery():
    """Branch A: FULLTEXT_USE_WEBSEARCH=True → websearch_to_tsquery."""
    db, captured = _make_mock_db()

    with patch("src.utils.search.settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH", True):
        await _fulltext_search_documents(
            db,
            workspace_name="ws1",
            observer="alice",
            observed="alice",
            query="test query",
            filters=None,
            limit=10,
        )

    assert len(captured) == 1
    sql = _compile_sql(captured[0])
    assert "websearch_to_tsquery" in sql
    assert "plainto_tsquery" not in sql


@pytest.mark.asyncio
async def test_fulltext_search_documents_unsafe_chars_uses_ilike_only():
    """Branch B: FULLTEXT_USE_WEBSEARCH=False + unsafe chars → ILIKE, no tsquery."""
    db, captured = _make_mock_db()

    with patch("src.utils.search.settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH", False):
        await _fulltext_search_documents(
            db,
            workspace_name="ws1",
            observer="alice",
            observed="alice",
            query="some-query!",
            filters=None,
            limit=10,
        )

    assert len(captured) == 1
    sql = _compile_sql(captured[0])
    assert "plainto_tsquery" not in sql
    assert "websearch_to_tsquery" not in sql
    assert "ilike" in sql
    assert "ts_rank" not in sql


@pytest.mark.asyncio
async def test_fulltext_search_documents_safe_chars_uses_plainto_tsquery():
    """Branch C: FULLTEXT_USE_WEBSEARCH=False + safe chars → plainto_tsquery."""
    db, captured = _make_mock_db()

    with patch("src.utils.search.settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH", False):
        await _fulltext_search_documents(
            db,
            workspace_name="ws1",
            observer="alice",
            observed="alice",
            query="simple query",
            filters=None,
            limit=10,
        )

    assert len(captured) == 1
    sql = _compile_sql(captured[0])
    assert "plainto_tsquery" in sql
    assert "websearch_to_tsquery" not in sql


# ---------------------------------------------------------------------------
# search() — hybrid pgvector guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_raises_validation_exception_when_hybrid_enabled_without_pgvector():
    """search() raises ValidationException when HYBRID_ENABLED=True but vector store is not pgvector."""
    with (
        patch("src.utils.search.settings.RETRIEVAL.HYBRID_ENABLED", True),
        patch(
            "src.utils.search._uses_pgvector_message_search",
            return_value=False,
        ),
    ):
        with pytest.raises(ValidationException, match="pgvector"):
            await search("test query", filters={"workspace_id": "ws1"})


@pytest.mark.asyncio
async def test_search_does_not_raise_when_hybrid_disabled():
    """search() skips the pgvector guard entirely when HYBRID_ENABLED=False."""
    with (
        patch("src.utils.search.settings.RETRIEVAL.HYBRID_ENABLED", False),
        patch("src.utils.search.settings.EMBED_MESSAGES", False),
        patch(
            "src.utils.search._uses_pgvector_message_search",
            return_value=False,
        ),
        patch("src.utils.search.tracked_db") as mock_tracked_db,
    ):
        mock_db = AsyncMock()
        mock_db.execute.return_value = MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        )
        mock_tracked_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_tracked_db.return_value.__aexit__ = AsyncMock(return_value=False)

        # Should not raise even though non-pgvector
        results = await search("test query", filters={"workspace_id": "ws1"})
        assert results == []
