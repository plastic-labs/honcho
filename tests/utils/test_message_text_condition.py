"""search.message_text_condition, message_text_page and matching_messages."""

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from src import models
from src.config import settings
from src.db import tenant_context
from src.utils.search import (
    TEXT_MATCH_PAGE_SIZE,
    matching_messages,
    message_text_condition,
    message_text_page,
)


def _compile(expr: Any) -> tuple[str, dict[str, Any]]:
    compiled = expr.compile(dialect=postgresql.dialect())
    return str(compiled), dict(compiled.params)


@pytest.mark.parametrize("use_fts", [True, False])
def test_inline_condition_matches_content(use_fts: bool) -> None:
    sql, params = _compile(message_text_condition("hello", use_fts=use_fts))
    assert models.MESSAGE_TEXT_MATCHES not in sql
    assert "messages.content ILIKE" in sql
    assert ("@@" in sql) is use_fts
    assert "%hello%" in params.values()


def test_inline_condition_escapes_the_pattern() -> None:
    sql, params = _compile(message_text_condition("50%_off", use_fts=False))
    assert "ESCAPE '\\\\'" in sql
    assert r"%50\%\_off%" in params.values()


def _with_tenant(fn: Any) -> Any:
    token = tenant_context.set("tenant-a")
    try:
        return fn()
    finally:
        tenant_context.reset(token)


async def _bound(fn: Any) -> Any:
    token = tenant_context.set("tenant-a")
    try:
        return await fn()
    finally:
        tenant_context.reset(token)


@pytest.mark.parametrize("use_fts", [True, False])
def test_page_reads_the_function_with_a_null_cursor(
    monkeypatch: pytest.MonkeyPatch, use_fts: bool
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    sql, params = _with_tenant(
        lambda: _compile(
            message_text_page("w", "hello", use_fts=use_fts, after=None, limit=7)
        )
    )
    assert f"FROM public.{models.MESSAGE_TEXT_MATCHES}(" in sql
    assert "CAST(" in sql and "AS FLOAT)" in sql and "AS BIGINT)" in sql
    assert "ILIKE" not in sql and "@@" not in sql
    assert "CAST(NULL AS FLOAT)" in sql
    assert list(params.values()) == ["tenant-a", "w", "hello", "%hello%", use_fts, 7]


def test_page_passes_the_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    at = datetime(2026, 1, 1, tzinfo=UTC)
    _, params = _with_tenant(
        lambda: _compile(
            message_text_page("w", "hello", use_fts=True, after=(0.5, at, 42), limit=7)
        )
    )
    assert list(params.values())[5:] == [0.5, at, 42, 7]


def test_page_qualifies_the_function_with_the_configured_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings.DB, "SCHEMA", "honcho_tenants")
    sql, _ = _with_tenant(
        lambda: _compile(
            message_text_page("w", "hello", use_fts=True, after=None, limit=7)
        )
    )
    assert f"FROM honcho_tenants.{models.MESSAGE_TEXT_MATCHES}(" in sql


def test_page_without_a_bound_tenant_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    with pytest.raises(ValueError, match="no tenant bound"):
        message_text_page("w", "hello", use_fts=True, after=None, limit=7)


class _Row:
    def __init__(self, id_: int) -> None:
        self.id: int = id_
        self.created_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)
        self.rank: float = 0.0


class _Message:
    def __init__(self, id_: int) -> None:
        self.id: int = id_


class _Pages:
    """A db whose page reads return canned keys and whose row reads keep a subset."""

    def __init__(self, pages: list[list[int]], kept: set[int]) -> None:
        self._pages: list[list[int]] = pages
        self._kept: set[int] = kept
        self.page_reads: int = 0
        self.cursors: list[tuple[float, datetime, int] | None] = []

    async def execute(self, stmt: Any) -> Any:
        sql, params = _compile(stmt)
        if models.MESSAGE_TEXT_MATCHES in sql:
            values = list(params.values())
            self.cursors.append(None if len(values) == 6 else tuple(values[5:8]))
            page = (
                self._pages[self.page_reads]
                if self.page_reads < len(self._pages)
                else []
            )
            self.page_reads += 1
            return _Result(rows=[_Row(i) for i in page])
        ids = params[next(k for k in params if k.startswith("id_1"))]
        wanted = [i for i in ids if i in self._kept]
        return _Result(messages=[_Message(i) for i in reversed(wanted)])


class _Result:
    def __init__(
        self, rows: list[_Row] | None = None, messages: list[_Message] | None = None
    ) -> None:
        self._rows: list[_Row] = rows or []
        self._messages: list[_Message] = messages or []

    def all(self) -> list[_Row]:
        return self._rows

    def scalars(self) -> list[_Message]:
        return self._messages


@pytest.mark.asyncio
async def test_matching_messages_walks_pages_until_its_filters_keep_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    full = list(range(1, TEXT_MATCH_PAGE_SIZE + 1))
    second = list(range(TEXT_MATCH_PAGE_SIZE + 1, TEXT_MATCH_PAGE_SIZE + 4))
    db = _Pages(pages=[full, second], kept={3, 7, 150, 102})
    stmt = select(models.Message)

    found = await _bound(
        lambda: matching_messages(
            db,  # pyright: ignore[reportArgumentType]
            stmt,
            "w",
            "x",
            use_fts=False,
            limit=3,
        )
    )

    assert [m.id for m in found] == [3, 7, 102]
    assert db.page_reads == 2
    assert db.cursors == [None, (0.0, datetime(2026, 1, 1, tzinfo=UTC), 100)]


@pytest.mark.asyncio
async def test_matching_messages_stops_at_a_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    db = _Pages(pages=[[5, 4, 3]], kept={4})

    found = await _bound(
        lambda: matching_messages(
            db,  # pyright: ignore[reportArgumentType]
            select(models.Message),
            "w",
            "x",
            use_fts=True,
            limit=10,
        )
    )

    assert [m.id for m in found] == [4]
    assert db.page_reads == 1


@pytest.mark.asyncio
async def test_matching_messages_keeps_the_page_order_and_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    db = _Pages(pages=[[9, 8, 7, 6]], kept={6, 7, 8, 9})

    found = await _bound(
        lambda: matching_messages(
            db,  # pyright: ignore[reportArgumentType]
            select(models.Message),
            "w",
            "x",
            use_fts=True,
            limit=2,
        )
    )

    assert [m.id for m in found] == [9, 8]
