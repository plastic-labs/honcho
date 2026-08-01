import uuid
from typing import Any

import pytest

import src.dependencies as dependencies_module
from src.config import settings
from src.db import request_context
from src.dependencies import get_db as real_get_db
from src.dependencies import tracked_db as real_tracked_db


class FakeSession:
    def __init__(self, *, in_transaction: bool = False):
        self._in_transaction: bool = in_transaction
        self.execute_calls: list[tuple[Any, ...]] = []
        self.rollback_calls: int = 0
        self.close_calls: int = 0
        self.connection_calls: int = 0

    async def connection(self) -> None:
        # Tracks checkout attempts so tests can assert get_db/tracked_db stay
        # lazy (they should never force a checkout themselves).
        self.connection_calls += 1

    async def execute(self, statement: Any, params: Any = None) -> None:
        self.execute_calls.append((statement, params))

    async def rollback(self) -> None:
        self.rollback_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    def in_transaction(self) -> bool:
        return self._in_transaction


@pytest.mark.asyncio
async def test_get_db_yields_lazily_without_checkout_or_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # get_db must NOT touch the connection or run set_config itself — checkout
    # happens lazily inside the AsyncSession on first DB use, so a handler doing
    # non-DB work before its first query never pins a connection.
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", True)  # still no set_config here

    dep_gen = real_get_db()
    try:
        db = await anext(dep_gen)
        assert db is fake_db
        assert fake_db.connection_calls == 0  # no eager checkout
        assert fake_db.execute_calls == []  # no set_config in get_db
    finally:
        await dep_gen.aclose()

    assert fake_db.rollback_calls == 1  # unconditional rollback in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_get_db_rolls_back_and_closes_when_consumer_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", False)

    dep_gen = real_get_db()
    await anext(dep_gen)

    with pytest.raises(RuntimeError, match="boom"):
        await dep_gen.athrow(RuntimeError("boom"))

    assert fake_db.rollback_calls == 2  # once in except, once in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_creates_and_resets_task_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )

    clear_token = request_context.set(None)
    try:
        async with real_tracked_db("cleanup_job"):
            # tracked_db sets the task context so the lazy session can read it.
            assert request_context.get() == "task:cleanup_job:12345678"
    finally:
        request_context.reset(clear_token)

    assert request_context.get() is None
    assert fake_db.rollback_calls == 1  # unconditional rollback in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_preserves_existing_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    context_token = request_context.set("request:existing")
    try:
        async with real_tracked_db("ignored_op"):
            assert request_context.get() == "request:existing"
    finally:
        request_context.reset(context_token)

    assert fake_db.rollback_calls == 1  # unconditional rollback in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_rolls_back_on_error_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", False)

    with pytest.raises(ValueError, match="failed operation"):
        async with real_tracked_db("operation"):
            raise ValueError("failed operation")

    assert fake_db.rollback_calls == 2  # once in except, once in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_rolls_back_open_transaction_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession(in_transaction=True)
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", False)

    async with real_tracked_db("operation"):
        pass

    assert fake_db.rollback_calls == 1
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_read_only_uses_read_sessionmaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # read_only=True must construct the session from ReadSessionLocal (the
    # AUTOCOMMIT engine) — and never touch SessionLocal — while keeping the
    # same rollback/close teardown.
    read_fake = FakeSession()
    monkeypatch.setattr(dependencies_module, "ReadSessionLocal", lambda: read_fake)
    monkeypatch.setattr(
        dependencies_module,
        "SessionLocal",
        lambda: pytest.fail("read_only window constructed a write session"),
    )

    async with real_tracked_db("read_op", read_only=True) as db:
        assert db is read_fake

    assert read_fake.rollback_calls == 1
    assert read_fake.close_calls == 1


@pytest.mark.asyncio
async def test_get_read_db_rolls_back_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fake = FakeSession()
    monkeypatch.setattr(dependencies_module, "ReadSessionLocal", lambda: read_fake)

    dep_gen = dependencies_module.get_read_db()
    try:
        db = await anext(dep_gen)
        assert db is read_fake
        assert read_fake.connection_calls == 0  # still lazy, no eager checkout
    finally:
        await dep_gen.aclose()

    assert read_fake.rollback_calls == 1
    assert read_fake.close_calls == 1


def test_read_engine_is_autocommit_and_shares_pool() -> None:
    # The read engine must differ from the write engine ONLY by isolation
    # level: AUTOCOMMIT (so reads never autobegin a transaction) on the same
    # underlying pool (no second connection budget).
    from src.db import engine, read_engine

    assert (
        read_engine.sync_engine._execution_options.get(  # pyright: ignore[reportPrivateUsage]
            "isolation_level"
        )
        == "AUTOCOMMIT"
    )
    assert read_engine.sync_engine.pool is engine.sync_engine.pool


@pytest.mark.asyncio
async def test_read_only_session_runs_in_autocommit_on_the_wire() -> None:
    # Wire-level guarantee behind the whole read-path fix: a read_only session's
    # connection has the DBAPI autocommit flag set, so psycopg emits no BEGIN
    # and the backend sits in state 'idle' (not 'idle in transaction') after a
    # statement returns. NOTE: get_isolation_level() can NOT verify this — it
    # reports the server's transaction_isolation GUC (READ COMMITTED), because
    # autocommit is a driver behavior, not a server isolation level.
    from sqlalchemy import text

    from src.db import read_engine

    async with real_tracked_db("read_op", read_only=True) as db:
        pid = (await db.execute(text("SELECT pg_backend_pid()"))).scalar()
        conn = await db.connection()
        raw = (await conn.get_raw_connection()).driver_connection
        assert raw is not None
        assert raw.autocommit is True

        # Definitive check, from a second connection: after the SELECT above,
        # the session's backend must be plain 'idle' — an open transaction
        # would report 'idle in transaction' and be reapable in production.
        async with read_engine.connect() as observer:
            state = (
                await observer.execute(
                    text("SELECT state FROM pg_stat_activity WHERE pid = :p"),
                    {"p": pid},
                )
            ).scalar()
        assert state == "idle"


@pytest.mark.asyncio
async def test_write_session_holds_idle_in_transaction_after_select() -> None:
    # Contrast guard documenting WHY the read engine exists: the default
    # (transactional) session autobegins on the first statement and leaves the
    # backend 'idle in transaction' until rollback/close — the state that
    # Postgres's idle_in_transaction_session_timeout reaps and that pins a
    # transaction-mode pooler backend.
    from sqlalchemy import text

    from src.db import read_engine

    async with real_tracked_db("write_op") as db:
        pid = (await db.execute(text("SELECT pg_backend_pid()"))).scalar()
        async with read_engine.connect() as observer:
            state = (
                await observer.execute(
                    text("SELECT state FROM pg_stat_activity WHERE pid = :p"),
                    {"p": pid},
                )
            ).scalar()
        assert state == "idle in transaction"


@pytest.mark.asyncio
async def test_read_only_session_works_with_tracing_checkout_hook() -> None:
    # Regression: the DB.TRACING checkout hook runs set_config() at pool
    # checkout, BEFORE the dialect applies the read engine's AUTOCOMMIT
    # isolation level. If that statement is allowed to autobegin a transaction,
    # psycopg then refuses to switch the connection into AUTOCOMMIT
    # ("can't change 'autocommit' now: connection in transaction") and every
    # read_only session 500s under TRACING. The hook must run in autocommit so
    # it leaves the connection idle. This combination is otherwise untested
    # because DB.TRACING defaults to false.
    from sqlalchemy import event, text

    from src.db import (
        _set_application_name_on_checkout,  # pyright: ignore[reportPrivateUsage]
        engine,
        read_engine,
    )

    context_token = request_context.set("tracing-regression")
    event.listen(engine.sync_engine, "checkout", _set_application_name_on_checkout)
    try:
        async with real_tracked_db("read_op", read_only=True) as db:
            pid = (await db.execute(text("SELECT pg_backend_pid()"))).scalar()
            app_name = (await db.execute(text("SHOW application_name"))).scalar()
            conn = await db.connection()
            raw = (await conn.get_raw_connection()).driver_connection
            assert raw is not None
            # AUTOCOMMIT was applied despite the checkout hook running first.
            assert raw.autocommit is True
            # The hook still tagged the connection (set_config is session-scoped,
            # so it survives the autocommit boundary).
            assert app_name == "tracing-regression"
            # Backend is idle, not idle-in-transaction: the no-BEGIN guarantee
            # holds even with the hook firing.
            async with read_engine.connect() as observer:
                state = (
                    await observer.execute(
                        text("SELECT state FROM pg_stat_activity WHERE pid = :p"),
                        {"p": pid},
                    )
                ).scalar()
            assert state == "idle"
    finally:
        event.remove(engine.sync_engine, "checkout", _set_application_name_on_checkout)
        request_context.reset(context_token)
