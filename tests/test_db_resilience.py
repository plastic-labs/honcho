"""Unit tests for DB connection resilience + observability.

These are DB-free: they exercise the application_name checkout hook against a
fake DBAPI connection, the deriver polling backoff math, and the in-flight gauge
listeners directly.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import src.db as db_module
from src.config import settings
from src.db import DBQueryInflightTracker
from src.telemetry.prometheus.metrics import db_queries_in_flight_gauge


def test_session_local_uses_vanilla_async_session() -> None:
    """Regression guard: no custom session subclass / acquisition logic.

    Connection acquisition is a single lazy checkout owned by AsyncSession; there
    must be no re-introduced eager-checkout or retry hooks on the session.
    """
    session = db_module.SessionLocal()
    assert type(session) is AsyncSession
    assert not hasattr(session, "_ensure_acquired")
    assert not hasattr(session, "_honcho_acquired")


# --- application_name checkout hook ------------------------------------------


class _FakeCursor:
    def __init__(self, recorder: list[Any], raise_exc: Exception | None) -> None:
        self.recorder: list[Any] = recorder
        self.raise_exc: Exception | None = raise_exc
        self.closed: bool = False

    def execute(self, sql: str, params: Any = None) -> None:
        if self.raise_exc is not None:
            raise self.raise_exc
        self.recorder.append((sql, params))

    def close(self) -> None:
        self.closed = True


class _FakeDBAPIConn:
    def __init__(self, recorder: list[Any], raise_exc: Exception | None = None) -> None:
        self._cursor: _FakeCursor = _FakeCursor(recorder, raise_exc)
        # Real pooled connections are checked out in non-autocommit mode; the
        # hook flips this to True for its statement then restores it so it never
        # leaves an open transaction that would block the read engine's
        # AUTOCOMMIT switch.
        self.autocommit: bool = False

    def cursor(self) -> _FakeCursor:
        return self._cursor


def test_checkout_hook_sets_application_name_from_request_context() -> None:
    recorder: list[Any] = []
    conn = _FakeDBAPIConn(recorder)
    token = db_module.request_context.set("request:trace-ctx")
    try:
        db_module._set_application_name_on_checkout(conn, None, None)  # pyright: ignore[reportPrivateUsage]
    finally:
        db_module.request_context.reset(token)

    assert len(recorder) == 1
    sql, params = recorder[0]
    assert "set_config" in sql and "application_name" in sql
    assert params == ("request:trace-ctx",)
    assert conn._cursor.closed is True  # pyright: ignore[reportPrivateUsage]
    # The hook restored the original (non-autocommit) mode after its statement.
    assert conn.autocommit is False


def test_checkout_hook_defaults_to_unknown_without_context() -> None:
    recorder: list[Any] = []
    conn = _FakeDBAPIConn(recorder)
    token = db_module.request_context.set(None)
    try:
        db_module._set_application_name_on_checkout(conn, None, None)  # pyright: ignore[reportPrivateUsage]
    finally:
        db_module.request_context.reset(token)

    assert recorder[0][1] == ("unknown",)


def test_checkout_hook_swallows_errors() -> None:
    """A failure tagging the connection must never break the checkout."""
    conn = _FakeDBAPIConn([], raise_exc=RuntimeError("boom"))
    # Must not raise.
    db_module._set_application_name_on_checkout(conn, None, None)  # pyright: ignore[reportPrivateUsage]


# --- deriver polling backoff math --------------------------------------------


def test_polling_backoff_sequence_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_ENABLED", True)
    monkeypatch.setattr(settings.DERIVER, "POLLING_SLEEP_INTERVAL_SECONDS", 1.0)
    monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_MULTIPLIER", 2.0)
    monkeypatch.setattr(settings.DERIVER, "POLLING_SLEEP_MAX_INTERVAL_SECONDS", 30.0)
    # Disable jitter so the schedule is asserted exactly (jitter is tested
    # separately in tests/deriver/test_queue_processing.py::TestPollingJitter).
    monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.0)

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
    monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.0)

    from src.deriver.queue_manager import QueueManager

    qm = QueueManager()
    assert [qm._advance_poll_interval() for _ in range(3)] == [1.0, 1.0, 1.0]  # pyright: ignore[reportPrivateUsage]


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
    monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.0)

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


@pytest.mark.asyncio
async def test_stale_cleanup_time_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """cleanup_stale_work_units runs at most once per gate interval per
    instance (staleness is a minutes-timescale condition; per-poll cleanup
    multiplies into needless fleet-wide write transactions). First poll always
    runs it so a crashed predecessor's stale rows are recovered immediately."""
    monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.0)
    monkeypatch.setattr(
        settings.DERIVER, "STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS", 60.0
    )

    from src.deriver import queue_manager as qm_mod

    qm = qm_mod.QueueManager()
    runs = {"n": 0}

    async def fake_cleanup() -> None:
        runs["n"] += 1

    monkeypatch.setattr(qm, "cleanup_stale_work_units", fake_cleanup)

    clock = {"now": 1_000.0}
    monkeypatch.setattr(
        "src.deriver.queue_manager.time.monotonic", lambda: clock["now"]
    )

    # First call runs (no prior attempt recorded).
    await qm._maybe_cleanup_stale_work_units()  # pyright: ignore[reportPrivateUsage]
    assert runs["n"] == 1

    # Inside the gate window: skipped.
    clock["now"] += 10.0
    await qm._maybe_cleanup_stale_work_units()  # pyright: ignore[reportPrivateUsage]
    assert runs["n"] == 1

    # Past the gate window: runs again.
    clock["now"] += 60.0
    await qm._maybe_cleanup_stale_work_units()  # pyright: ignore[reportPrivateUsage]
    assert runs["n"] == 2


@pytest.mark.asyncio
async def test_stale_cleanup_gate_failed_attempt_waits_full_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate records the ATTEMPT before running, so a failing cleanup is not
    retried on every poll against a DB that is already struggling."""
    monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.0)
    monkeypatch.setattr(
        settings.DERIVER, "STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS", 60.0
    )

    from src.deriver import queue_manager as qm_mod

    qm = qm_mod.QueueManager()
    attempts = {"n": 0}

    async def failing_cleanup() -> None:
        attempts["n"] += 1
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(qm, "cleanup_stale_work_units", failing_cleanup)

    clock = {"now": 1_000.0}
    monkeypatch.setattr(
        "src.deriver.queue_manager.time.monotonic", lambda: clock["now"]
    )

    with pytest.raises(RuntimeError):
        await qm._maybe_cleanup_stale_work_units()  # pyright: ignore[reportPrivateUsage]
    assert attempts["n"] == 1

    # Immediately after the failure: still gated, no hammering.
    clock["now"] += 1.0
    await qm._maybe_cleanup_stale_work_units()  # pyright: ignore[reportPrivateUsage]
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_stale_cleanup_gate_zero_interval_runs_every_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interval 0.0 preserves legacy run-on-every-poll behavior."""
    monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.0)
    monkeypatch.setattr(
        settings.DERIVER, "STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS", 0.0
    )

    from src.deriver import queue_manager as qm_mod

    qm = qm_mod.QueueManager()
    runs = {"n": 0}

    async def fake_cleanup() -> None:
        runs["n"] += 1

    monkeypatch.setattr(qm, "cleanup_stale_work_units", fake_cleanup)

    await qm._maybe_cleanup_stale_work_units()  # pyright: ignore[reportPrivateUsage]
    await qm._maybe_cleanup_stale_work_units()  # pyright: ignore[reportPrivateUsage]
    assert runs["n"] == 2
