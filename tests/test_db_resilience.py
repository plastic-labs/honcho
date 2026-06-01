"""Unit tests for DB connection resilience + observability.

These are DB-free: they exercise the retry helper against a fake session, the
deriver polling backoff math, and the in-flight gauge listeners directly.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

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
