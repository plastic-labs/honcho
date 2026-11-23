"""Message search runs under ``DB.SEARCH_STATEMENT_TIMEOUT_MS``, against a real database."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.dependencies import tracked_db
from src.exceptions import SearchTimeoutError
from src.utils.search import search_db


async def _statement_timeout(db: AsyncSession) -> str:
    return (await db.execute(text("SHOW statement_timeout"))).scalar_one()


@pytest.mark.asyncio
async def test_search_db_sets_the_timeout_for_its_transaction_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DB, "SEARCH_STATEMENT_TIMEOUT_MS", 1234)
    async with tracked_db("test.before") as db:
        default = await _statement_timeout(db)

    async with search_db("test.search") as db:
        assert await _statement_timeout(db) == "1234ms"

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
