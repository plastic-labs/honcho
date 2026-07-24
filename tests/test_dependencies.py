import asyncio
import uuid
from typing import Any

import pytest
from sqlalchemy.exc import InterfaceError

import src.dependencies as dependencies_module
from src.config import settings
from src.db import request_context
from src.dependencies import get_db as real_get_db
from src.dependencies import tracked_db as real_tracked_db


class FakeSession:
    def __init__(
        self,
        *,
        in_transaction: bool = False,
        rollback_exc: BaseException | None = None,
        close_exc: BaseException | None = None,
    ):
        self._in_transaction: bool = in_transaction
        # Optional failures injected into the cleanup path to exercise
        # cancellation-hardening (DEV-1861): a CancelledError re-delivered on
        # rollback, or a broken-connection error that must trigger invalidate().
        self._rollback_exc: BaseException | None = rollback_exc
        self._close_exc: BaseException | None = close_exc
        self.execute_calls: list[tuple[Any, ...]] = []
        self.rollback_calls: int = 0
        self.close_calls: int = 0
        self.connection_calls: int = 0
        self.invalidate_calls: int = 0

    async def connection(self) -> None:
        # Tracks checkout attempts so tests can assert get_db/tracked_db stay
        # lazy (they should never force a checkout themselves).
        self.connection_calls += 1

    async def execute(self, statement: Any, params: Any = None) -> None:
        self.execute_calls.append((statement, params))

    async def rollback(self) -> None:
        self.rollback_calls += 1
        if self._rollback_exc is not None:
            raise self._rollback_exc

    async def invalidate(self) -> None:
        self.invalidate_calls += 1

    async def close(self) -> None:
        self.close_calls += 1
        if self._close_exc is not None:
            raise self._close_exc

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

    assert fake_db.rollback_calls == 1  # single rollback in shielded finally
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

    assert fake_db.rollback_calls == 1  # single rollback in shielded finally
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


# --- Cancellation-hardened cleanup (DEV-1861) --------------------------------
#
# The regression: on a cancelled request task, cleanup must still ROLLBACK and
# close so the pooled connection is released and no backend is left parked
# 'idle in transaction'. The shielded finally in dependencies.py must survive
# both a CancelledError raised in the body AND one re-delivered during cleanup.


@pytest.mark.asyncio
async def test_tracked_db_rolls_back_and_closes_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Body cancelled mid-transaction (the client-disconnect case): cleanup must
    # still run and the CancelledError must propagate (never be swallowed).
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    with pytest.raises(asyncio.CancelledError):
        async with real_tracked_db("cancelled_op"):
            raise asyncio.CancelledError()

    assert fake_db.rollback_calls == 1
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_closes_when_cancel_redelivered_during_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The core DEV-1861 defense: a cancellation re-delivered on the ROLLBACK
    # await must NOT skip close(). We simulate re-delivery by making rollback()
    # itself raise CancelledError; close() must still run so the connection is
    # released rather than orphaned with an open BEGIN.
    fake_db = FakeSession(rollback_exc=asyncio.CancelledError())
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    with pytest.raises(asyncio.CancelledError):
        async with real_tracked_db("cancelled_cleanup_op"):
            pass

    assert fake_db.rollback_calls == 1
    assert fake_db.close_calls == 1  # close ran despite the interrupted rollback


@pytest.mark.asyncio
async def test_tracked_db_invalidates_connection_on_broken_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mid-protocol break: ROLLBACK fails with a driver InterfaceError. The dead
    # connection must be invalidated (not returned to the pool) and close() must
    # still run. The broken-connection error must not surface to the caller —
    # cleanup is best-effort by this point.
    broken = InterfaceError("ROLLBACK", None, Exception("connection is closed"))
    fake_db = FakeSession(rollback_exc=broken)
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    async with real_tracked_db("broken_conn_op"):
        pass

    assert fake_db.rollback_calls == 1
    assert fake_db.invalidate_calls == 1
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_get_db_rolls_back_and_closes_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same guarantee via the FastAPI DI generator path (Starlette cancels the
    # request task on client disconnect and drives teardown through athrow).
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    dep_gen = real_get_db()
    await anext(dep_gen)

    with pytest.raises(asyncio.CancelledError):
        await dep_gen.athrow(asyncio.CancelledError())

    assert fake_db.rollback_calls == 1
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_survives_real_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end check with a genuine asyncio cancellation (not a hand-raised
    # exception): cancel a task blocked inside the tracked_db body and assert
    # cleanup still completed and the CancelledError still propagates.
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    entered = asyncio.Event()

    async def worker() -> None:
        async with real_tracked_db("blocked_op"):
            entered.set()
            await asyncio.sleep(3600)  # block until cancelled

    task = asyncio.ensure_future(worker())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_db.rollback_calls == 1
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_survives_cancellation_storm_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The strongest DEV-1861 guarantee: repeated cancellations re-delivered
    # WHILE cleanup is running must not stop rollback+close from completing.
    # `_run_to_completion` runs cleanup as a shielded detached task, so even a
    # storm of task.cancel()s cannot orphan the connection. (This is precisely
    # the case `anyio.CancelScope(shield=True)` fails to cover for a native
    # task.cancel(), which is why we do not use it.) rollback() here awaits, so
    # a re-delivered cancel has a real window to land mid-cleanup.
    class SlowRollbackSession(FakeSession):
        async def rollback(self) -> None:
            self.rollback_calls += 1
            await asyncio.sleep(0.2)  # cancellation window during cleanup

    fake_db = SlowRollbackSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    entered = asyncio.Event()

    async def worker() -> None:
        async with real_tracked_db("stormed_op"):
            entered.set()
            await asyncio.sleep(3600)

    task = asyncio.ensure_future(worker())
    await entered.wait()
    for _ in range(5):  # cancel storm straddling the cleanup window
        task.cancel()
        await asyncio.sleep(0.05)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_db.rollback_calls == 1
    assert fake_db.close_calls == 1  # cleanup completed despite the storm


@pytest.mark.asyncio
async def test_tracked_db_resets_request_context_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The task-scoped contextvar must be reset even when cleanup re-raises
    # CancelledError, else a reused long-lived task (e.g. the deriver) carries a
    # stale `task:...` context into later work and corrupts tracing/attribution.
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    clear_token = request_context.set(None)
    try:
        with pytest.raises(asyncio.CancelledError):
            async with real_tracked_db("ctx_cancel_op"):
                assert (request_context.get() or "").startswith("task:ctx_cancel_op:")
                raise asyncio.CancelledError()
        assert request_context.get() is None  # reset despite the cancel
    finally:
        request_context.reset(clear_token)


@pytest.mark.asyncio
async def test_run_to_completion_abandons_wedged_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shielded cleanup is uninterruptible by design; a wedged connection (dead
    # socket, no libpq timeout) must therefore NOT pin the caller forever.
    # _run_to_completion abandons cleanup past _CLEANUP_TIMEOUT_SECONDS.
    monkeypatch.setattr(dependencies_module, "_CLEANUP_TIMEOUT_SECONDS", 0.2)
    started = asyncio.Event()

    async def wedged() -> None:
        started.set()
        await asyncio.sleep(3600)  # never completes

    # Returns (abandons) well under the sleep instead of hanging.
    await asyncio.wait_for(dependencies_module._run_to_completion(wedged()), timeout=5)
    assert started.is_set()


@pytest.mark.asyncio
async def test_tracked_db_no_idle_in_transaction_after_real_cancellation() -> None:
    # Wire-level chaos test (ticket verification): cancel a REAL in-flight write
    # transaction mid-flight and assert, from an independent connection, that the
    # backend is not left parked 'idle in transaction'. Complements the mock
    # tests above with a real pooled connection receiving ROLLBACK under cancel.
    from sqlalchemy import text

    from src.db import read_engine

    entered = asyncio.Event()
    pid_box: dict[str, int] = {}

    async def worker() -> None:
        async with real_tracked_db("chaos_cancel") as db:
            pid = (await db.execute(text("SELECT pg_backend_pid()"))).scalar()
            pid_box["pid"] = int(pid)  # pyright: ignore[reportArgumentType]
            entered.set()
            await asyncio.sleep(3600)  # hold the open BEGIN until cancelled

    async def backend_state(pid: int) -> str | None:
        async with read_engine.connect() as obs:
            return (
                await obs.execute(
                    text("SELECT state FROM pg_stat_activity WHERE pid = :p"),
                    {"p": pid},
                )
            ).scalar()

    task = asyncio.ensure_future(worker())
    await entered.wait()
    pid = pid_box["pid"]
    assert await backend_state(pid) == "idle in transaction"  # precondition

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    state = await backend_state(pid)
    for _ in range(20):  # allow shielded cleanup a moment to land
        if state != "idle in transaction":
            break
        await asyncio.sleep(0.1)
        state = await backend_state(pid)
    assert state != "idle in transaction", f"backend left {state!r} after cancel"
