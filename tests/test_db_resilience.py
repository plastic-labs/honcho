"""Unit tests for DB connection resilience + observability.

These are DB-free: they exercise the retry helper against a fake session, the
deriver polling backoff math, and the in-flight gauge listeners directly.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

import src.db as db_module
from src.config import settings
from src.db import DBQueryInflightTracker, acquire_connection_with_retry
from src.telemetry.prometheus.metrics import (
    db_connection_acquisitions_counter,
    db_queries_in_flight_gauge,
)


def _make_operational_error() -> OperationalError:
    """A stand-in for how a saturated pooler surfaces ('too many clients')."""
    return OperationalError("SELECT 1", {}, Exception("too many clients"))


class _FlakyConnSession:
    """Fake AsyncSession whose connection() fails N times then succeeds."""

    def __init__(self, fail_times: int, *, always_fail: bool = False) -> None:
        self.fail_times: int = fail_times
        self.always_fail: bool = always_fail
        self.calls: int = 0
        self.rollback_calls: int = 0

    async def connection(self) -> None:
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_times:
            raise _make_operational_error()

    async def rollback(self) -> None:
        self.rollback_calls += 1


def _acq_count(outcome: str) -> float:
    child = db_connection_acquisitions_counter.labels(
        instance_type="api", outcome=outcome
    )
    return float(child._value.get())  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]


@pytest.fixture
def metrics_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.METRICS, "ENABLED", True)
    monkeypatch.setattr(settings.METRICS, "NAMESPACE", "test")
    monkeypatch.setattr(db_module, "_db_instance_type", "api")
    # Fast, deterministic backoff so retry tests don't sleep.
    monkeypatch.setattr(settings.DB, "CONNECTION_RETRY_ENABLED", True)
    monkeypatch.setattr(settings.DB, "CONNECTION_RETRY_BACKOFF_INITIAL_SECONDS", 0.001)
    monkeypatch.setattr(settings.DB, "CONNECTION_RETRY_BACKOFF_MAX_SECONDS", 0.01)


@pytest.mark.asyncio
@pytest.mark.usefixtures("metrics_on")
async def test_acquire_succeeds_first_try_records_ok() -> None:
    before = _acq_count("ok")
    session = _FlakyConnSession(fail_times=0)
    await acquire_connection_with_retry(session, "request:test")  # pyright: ignore[reportArgumentType]
    assert session.calls == 1
    assert _acq_count("ok") == before + 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("metrics_on")
async def test_acquire_retries_then_succeeds_records_retried() -> None:
    before = _acq_count("retried")
    session = _FlakyConnSession(fail_times=2)
    await acquire_connection_with_retry(session, "request:test")  # pyright: ignore[reportArgumentType]
    assert session.calls == 3  # two failures then success
    # Session is reset after each failed checkout before the next attempt.
    assert session.rollback_calls == 2
    assert _acq_count("retried") == before + 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("metrics_on")
async def test_acquire_exhausts_budget_reraises_and_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tiny budget so the loop gives up quickly under sustained failure.
    monkeypatch.setattr(settings.DB, "CONNECTION_RETRY_MAX_DELAY_SECONDS", 0.05)
    before = _acq_count("exhausted")
    session = _FlakyConnSession(fail_times=0, always_fail=True)
    with pytest.raises(OperationalError):
        await acquire_connection_with_retry(session, "request:test")  # pyright: ignore[reportArgumentType]
    assert session.calls >= 1
    assert _acq_count("exhausted") == before + 1


@pytest.mark.asyncio
async def test_acquire_disabled_calls_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.DB, "CONNECTION_RETRY_ENABLED", False)
    session = _FlakyConnSession(fail_times=0)
    await acquire_connection_with_retry(session, "request:test")  # pyright: ignore[reportArgumentType]
    assert session.calls == 1


# --- HonchoAsyncSession: lazy checkout with retry on first DB use ------------


@pytest.mark.asyncio
async def test_session_lazy_acquires_on_first_db_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No checkout at construction; acquisition (with retry) happens once on the
    first execute, and the statement itself runs exactly once."""
    monkeypatch.setattr(settings.DB, "TRACING", False)
    acquired: list[str] = []
    executed: list[Any] = []

    async def fake_acquire(_session: Any, context: str) -> None:
        acquired.append(context)

    async def fake_super_execute(_self: Any, *args: Any, **_kw: Any) -> str:
        executed.append(args[0] if args else None)
        return "result"

    monkeypatch.setattr(db_module, "acquire_connection_with_retry", fake_acquire)
    monkeypatch.setattr(AsyncSession, "execute", fake_super_execute)

    session = db_module.SessionLocal()
    assert session._honcho_acquired is False  # pyright: ignore[reportPrivateUsage]
    assert acquired == []  # constructing the session does NOT check out

    result = await session.execute("SELECT 1")
    assert result == "result"
    assert acquired == ["unknown"]  # acquired exactly once, on first use
    assert session._honcho_acquired is True  # pyright: ignore[reportPrivateUsage]
    assert executed == ["SELECT 1"]  # statement ran once (not retried)

    await session.execute("SELECT 2")
    assert acquired == ["unknown"]  # still once — no re-acquire on later use


@pytest.mark.asyncio
async def test_session_tracing_sets_application_name_on_acquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DB, "TRACING", True)
    calls: list[tuple[Any, ...]] = []

    async def fake_acquire(_session: Any, _context: str) -> None:
        return None

    async def fake_super_execute(_self: Any, *args: Any, **_kw: Any) -> None:
        calls.append(args)

    monkeypatch.setattr(db_module, "acquire_connection_with_retry", fake_acquire)
    monkeypatch.setattr(AsyncSession, "execute", fake_super_execute)

    token = db_module.request_context.set("request:trace-ctx")
    try:
        session = db_module.SessionLocal()
        await session.execute("SELECT 1")
    finally:
        db_module.request_context.reset(token)

    # First super().execute is the set_config, then the real statement.
    assert any("set_config" in str(c[0]) for c in calls)
    set_config_call = next(c for c in calls if "set_config" in str(c[0]))
    assert set_config_call[1] == {"name": "request:trace-ctx"}


@pytest.mark.asyncio
async def test_session_commit_and_rollback_reset_acquired_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acquire(_session: Any, _context: str) -> None:
        return None

    async def noop(_self: Any) -> None:
        return None

    monkeypatch.setattr(db_module, "acquire_connection_with_retry", fake_acquire)
    monkeypatch.setattr(AsyncSession, "commit", noop)
    monkeypatch.setattr(AsyncSession, "rollback", noop)

    session = db_module.SessionLocal()
    await session.commit()  # ensures acquired, commits, then resets
    assert session._honcho_acquired is False  # pyright: ignore[reportPrivateUsage]

    session._honcho_acquired = True  # pyright: ignore[reportPrivateUsage]
    await session.rollback()
    assert session._honcho_acquired is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_db_does_not_acquire_at_dependency_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.dependencies import get_db as real_get_db

    acquired: list[str] = []

    async def fake_acquire(_session: Any, context: str) -> None:
        acquired.append(context)

    monkeypatch.setattr(db_module, "acquire_connection_with_retry", fake_acquire)

    dep_gen = real_get_db()
    await anext(dep_gen)
    assert acquired == []  # yielding the session must not check out
    await dep_gen.aclose()
    assert acquired == []  # finally rollback/close must not check out either


def test_polling_backoff_sequence_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_ENABLED", True)
    monkeypatch.setattr(settings.DERIVER, "POLLING_SLEEP_INTERVAL_SECONDS", 1.0)
    monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_MULTIPLIER", 2.0)
    monkeypatch.setattr(settings.DERIVER, "POLLING_SLEEP_MAX_INTERVAL_SECONDS", 30.0)

    from src.deriver.queue_manager import QueueManager

    qm = QueueManager()
    seq = [qm._advance_poll_interval() for _ in range(8)]  # pyright: ignore[reportPrivateUsage]
    assert seq == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]  # caps at max

    qm._reset_poll_interval()  # pyright: ignore[reportPrivateUsage]
    assert qm._advance_poll_interval() == 1.0  # pyright: ignore[reportPrivateUsage]


def test_polling_backoff_disabled_stays_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_ENABLED", False)
    monkeypatch.setattr(settings.DERIVER, "POLLING_SLEEP_INTERVAL_SECONDS", 1.0)

    from src.deriver.queue_manager import QueueManager

    qm = QueueManager()
    assert [qm._advance_poll_interval() for _ in range(3)] == [1.0, 1.0, 1.0]  # pyright: ignore[reportPrivateUsage]


def test_pool_timeout_must_be_under_retry_budget() -> None:
    """A pooled QueuePool checkout that can block past the retry budget is a
    contradiction (no retry ever happens) and must fail config validation."""
    from src.config import DBSettings

    # Contradiction: pooled + retry on, but POOL_TIMEOUT >= budget.
    with pytest.raises(ValueError, match="DB_POOL_TIMEOUT"):
        DBSettings(
            POOL_CLASS="default",
            POOL_TIMEOUT=30,
            CONNECTION_RETRY_ENABLED=True,
            CONNECTION_RETRY_MAX_DELAY_SECONDS=10.0,
        )

    # NullPool has no queue wait, so POOL_TIMEOUT is irrelevant -> allowed.
    DBSettings(
        POOL_CLASS="null",
        POOL_TIMEOUT=30,
        CONNECTION_RETRY_ENABLED=True,
        CONNECTION_RETRY_MAX_DELAY_SECONDS=10.0,
    )


@pytest.mark.asyncio
async def test_polling_loop_idle_sleeps_once_per_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real loop on an empty queue: exactly one (growing, capped)
    sleep per empty poll — no double-sleep from the queue_empty_flag branch."""
    monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_ENABLED", True)
    monkeypatch.setattr(settings.DERIVER, "POLLING_SLEEP_INTERVAL_SECONDS", 1.0)
    monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_MULTIPLIER", 2.0)
    monkeypatch.setattr(settings.DERIVER, "POLLING_SLEEP_MAX_INTERVAL_SECONDS", 8.0)

    import asyncio

    from src.deriver import queue_manager as qm_mod

    qm = qm_mod.QueueManager()
    sleeps: list[float] = []
    polls = {"n": 0}

    async def fake_cleanup() -> None:
        return None

    async def fake_claim() -> dict[str, str]:
        polls["n"] += 1
        if polls["n"] >= 5:
            qm.shutdown_event.set()  # stop after 5 empty polls
        return {}

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(qm, "cleanup_stale_work_units", fake_cleanup)
    monkeypatch.setattr(qm, "get_and_claim_work_units", fake_claim)
    # queue_manager calls asyncio.sleep on the stdlib module; patch it there.
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await qm.polling_loop()

    # One sleep per empty poll, growing 1->2->4->8 then capped at 8 (not doubled).
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_inflight_gauge_no_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.METRICS, "NAMESPACE", "test")
    child: Any = db_queries_in_flight_gauge.labels(instance_type="api")
    tracker = DBQueryInflightTracker(child)
    key = DBQueryInflightTracker.INFLIGHT_KEY

    def value() -> float:
        return float(child._value.get())

    start = value()
    conn = SimpleNamespace(info={})

    # Normal execute: before -> after returns to baseline.
    tracker.on_before(conn)
    assert value() == start + 1
    assert conn.info[key] is True
    tracker.on_after(conn)
    assert value() == start
    assert key not in conn.info

    # Errored execute: before -> on_error decrements (after never fires).
    tracker.on_before(conn)
    assert value() == start + 1
    tracker.on_error(SimpleNamespace(connection=conn))
    assert value() == start

    # on_error without a matching before (e.g. connect error) must not push the
    # gauge negative.
    tracker.on_error(SimpleNamespace(connection=SimpleNamespace(info={})))
    assert value() == start
