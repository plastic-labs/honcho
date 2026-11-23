"""Single-tenant text search, against a real database.

With MULTI_TENANT off the text match runs as plain SQL
(``message_text_page_inline``) instead of through the SECURITY DEFINER
function, and every search runs under ``DB.SEARCH_STATEMENT_TIMEOUT_MS``.
"""

from collections.abc import Callable
from datetime import datetime

import pytest
from sqlalchemy import Select, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.db import tenant_context
from src.dependencies import tracked_db
from src.exceptions import SearchTimeoutError
from src.utils.search import (
    message_text_page,
    message_text_page_inline,
    search_db,
)

PageQuery = Callable[..., Select[tuple[int, datetime, float]]]

CONTENTS = [
    "apple pie recipe",
    "an apple a day",
    "apples and oranges",
    "50% off apple juice",
    "nothing to see here",
    "apple apple apple",
    "50% off everything",
    "pear tart",
    "apple crumble with apple sauce",
    "the apple of my eye",
    "50% off pears",
    "save 50% off now",
]


async def _seed(db: AsyncSession) -> str:
    workspace = models.Workspace(name="search-ws")
    peer = models.Peer(name="search-peer", workspace_name=workspace.name)
    session = models.Session(name="search-session", workspace_name=workspace.name)
    db.add_all([workspace, peer, session])
    await db.flush()
    # One transaction, so every row shares created_at: the id tail of the sort
    # key is what orders them, which is the tie the cursor has to get right.
    db.add_all(
        models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            seq_in_session=seq,
            content=content,
        )
        for seq, content in enumerate(CONTENTS, start=1)
    )
    await db.flush()
    return workspace.name


async def _walk(
    db: AsyncSession, page_query: PageQuery, workspace: str, query: str, use_fts: bool
) -> list[int]:
    ids: list[int] = []
    after: tuple[float, datetime, int] | None = None
    while True:
        page = (
            await db.execute(
                page_query(workspace, query, use_fts=use_fts, after=after, limit=2)
            )
        ).all()
        ids.extend(row.id for row in page)
        if len(page) < 2:
            return ids
        last = page[-1]
        after = (last.rank, last.created_at, last.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "use_fts"), [("apple", True), ("apple", False), ("50% off", False)]
)
async def test_inline_page_returns_what_the_function_returns(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    use_fts: bool,
) -> None:
    """The two page sources agree row for row, across pages of two."""
    db = db_session
    workspace = await _seed(db)
    tenant = models.instance_tenant_id()

    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    inline = await _walk(db, message_text_page_inline, workspace, query, use_fts)

    for statement in models.message_text_matches_ddl(settings.DB.SCHEMA):
        await db.execute(text(statement))
    await db.execute(select(text("set_config('app.tenant', :t, true)")), {"t": tenant})
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(tenant)
    try:
        function = await _walk(db, message_text_page, workspace, query, use_fts)
    finally:
        tenant_context.reset(token)

    assert len(inline) > 2  # more than one page
    assert inline == function


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
