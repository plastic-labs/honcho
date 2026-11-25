"""Message search runs under ``DB.SEARCH_STATEMENT_TIMEOUT_MS``, against a real database."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.dependencies import tracked_db
from src.exceptions import SearchTimeoutError
from src.utils import search as search_module
from src.utils.search import bound_search_statement, search_db


async def _statement_timeout(db: AsyncSession) -> str:
    return (await db.execute(text("SHOW statement_timeout"))).scalar_one()


def _ms(shown: str) -> int:
    """``SHOW statement_timeout`` output (``"1233ms"``, ``"1s"``, ``"0"``) in ms."""
    if shown.endswith("ms"):
        return int(shown[:-2])
    if shown.endswith("s"):
        return int(shown[:-1]) * 1000
    return int(shown)


@pytest.mark.asyncio
async def test_search_db_sets_the_timeout_for_its_transaction_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DB, "SEARCH_STATEMENT_TIMEOUT_MS", 1234)
    async with tracked_db("test.before") as db:
        default = await _statement_timeout(db)

    async with search_db("test.search") as db:
        # the budget left by the time the statement runs: at most the setting
        assert 0 < _ms(await _statement_timeout(db)) <= 1234

    async with tracked_db("test.after") as db:
        assert await _statement_timeout(db) == default


@pytest.mark.asyncio
async def test_search_db_without_a_timeout_leaves_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with tracked_db("test.before") as db:
        default = await _statement_timeout(db)
    monkeypatch.setattr(settings.DB, "SEARCH_STATEMENT_TIMEOUT_MS", 0)
    async with search_db("test.search") as db:
        assert await _statement_timeout(db) == default


@pytest.mark.asyncio
async def test_a_search_over_the_timeout_raises_search_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DB, "SEARCH_STATEMENT_TIMEOUT_MS", 50)
    with pytest.raises(SearchTimeoutError) as raised:
        async with search_db("test.search") as db:
            await db.execute(text("SELECT pg_sleep(2)"))
    assert raised.value.status_code == 504

    # The cancelled transaction is rolled back and the connection is usable.
    async with tracked_db("test.after") as db:
        assert (await db.execute(text("SELECT 1"))).scalar_one() == 1


@pytest.mark.asyncio
async def test_other_database_errors_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DB, "SEARCH_STATEMENT_TIMEOUT_MS", 50)
    with pytest.raises(ProgrammingError):
        async with search_db("test.search") as db:
            await db.execute(text("SELECT * FROM no_such_table"))


@pytest.mark.asyncio
async def test_reads_the_server_statement_timeout() -> None:
    async with tracked_db("test.server") as db:
        await db.execute(text("SET LOCAL statement_timeout = '1234ms'"))
        assert await search_module._server_statement_timeout_ms(db) == 1234  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_a_stricter_server_timeout_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DB, "SEARCH_STATEMENT_TIMEOUT_MS", 60_000)

    async def _server(_db: AsyncSession) -> int:
        return 500

    monkeypatch.setattr(search_module, "_server_statement_timeout_ms", _server)
    async with search_db("test.search") as db:
        assert 0 < _ms(await _statement_timeout(db)) <= 500


@pytest.mark.asyncio
async def test_the_budget_covers_the_whole_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each statement gets what is left, so several statements that each fit
    the setting still can't add up past it."""
    monkeypatch.setattr(settings.DB, "SEARCH_STATEMENT_TIMEOUT_MS", 600)
    with pytest.raises(SearchTimeoutError):
        async with search_db("test.search") as db:
            await db.execute(text("SELECT pg_sleep(0.35)"))
            await bound_search_statement(db)
            assert _ms(await _statement_timeout(db)) <= 250
            await db.execute(text("SELECT pg_sleep(0.35)"))


@pytest.mark.asyncio
async def test_a_spent_budget_stops_before_the_next_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DB, "SEARCH_STATEMENT_TIMEOUT_MS", 100)
    with pytest.raises(SearchTimeoutError):
        async with search_db("test.search") as db:
            await asyncio.sleep(0.15)
            await bound_search_statement(db)


@pytest.mark.asyncio
async def test_bound_search_statement_is_a_noop_outside_search_db() -> None:
    async with tracked_db("test.plain") as db:
        default = await _statement_timeout(db)
        await bound_search_statement(db)
        assert await _statement_timeout(db) == default
