"""Generic backfill task: registration, enqueue gating, dispatch, and gauge."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver import consumer
from src.reconciler import backfill as backfill_module
from src.reconciler import scheduler as scheduler_module
from src.reconciler.backfill import Backfill, run_backfill_cycle
from src.reconciler.scheduler import (
    RECONCILER_TASKS,
    ReconcilerScheduler,
    backfill_task,
)
from src.telemetry import prometheus_metrics
from src.telemetry.events import BackfillCompletedEvent, BaseEvent
from src.utils.queue_payload import ReconcilerPayload
from src.utils.work_unit import parse_work_unit_key


@pytest.fixture(autouse=True)
def _reset_scheduler_singleton():  # pyright: ignore[reportUnusedFunction]
    ReconcilerScheduler.reset_singleton()
    yield
    ReconcilerScheduler.reset_singleton()


class _Rows:
    """In-memory pending set for a fake backfill."""

    def __init__(self, remaining: int) -> None:
        self.remaining: int = remaining
        self.sessions: list[AsyncSession] = []


def _fake_backfill(rows: _Rows, **overrides: Any) -> Backfill:
    async def has_pending(_db: AsyncSession) -> bool:
        return rows.remaining > 0

    async def count_pending(_db: AsyncSession) -> int:
        return rows.remaining

    async def run_batch(db: AsyncSession, batch_size: int) -> int:
        rows.sessions.append(db)
        count = min(batch_size, rows.remaining)
        rows.remaining -= count
        return count

    return replace(
        Backfill(
            name="fake",
            has_pending=has_pending,
            count_pending=count_pending,
            run_batch=run_batch,
            interval_seconds=60,
            retire_after="never",
            batch_size=2,
        ),
        **overrides,
    )


def _patch_scheduler_db(monkeypatch: pytest.MonkeyPatch, db: AsyncSession) -> None:
    @asynccontextmanager
    async def _db(_: str | None = None) -> AsyncGenerator[AsyncSession]:
        yield db

    monkeypatch.setattr(scheduler_module, "tracked_db", _db)


async def _queue_rows(db: AsyncSession, work_unit_key: str) -> list[models.QueueItem]:
    result = await db.execute(
        select(models.QueueItem).where(models.QueueItem.work_unit_key == work_unit_key)
    )
    return list(result.scalars().all())


def _capture_events(monkeypatch: pytest.MonkeyPatch) -> list[BaseEvent]:
    events: list[BaseEvent] = []
    monkeypatch.setattr(consumer, "emit", events.append)
    return events


def _reconciler_item(payload: dict[str, Any]) -> models.QueueItem:
    return models.QueueItem(
        work_unit_key="reconciler:test",
        payload=payload,
        session_id=None,
        task_type="reconciler",
        workspace_name=None,
        message_id=None,
    )


async def _legacy_document(db: AsyncSession) -> tuple[str, str]:
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.commit()
    peer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db.add(peer)
    await db.commit()
    db.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db.commit()
    parent = str(generate_nanoid())
    doc = models.Document(
        workspace_name=workspace.name,
        observer=peer.name,
        observed=peer.name,
        content="derived",
        level="deductive",
        legacy_source_ids=[parent],
        internal_metadata={},
    )
    db.add(doc)
    await db.commit()
    return doc.id, parent


# ---------------------------------------------------------------------------
# Enqueue gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("remaining", [0, 3])
async def test_generic_enqueue_gated_on_has_pending(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, remaining: int
) -> None:
    _patch_scheduler_db(monkeypatch, db_session)
    task = backfill_task(_fake_backfill(_Rows(remaining)))

    enqueued = await ReconcilerScheduler()._try_enqueue_task(task)  # pyright: ignore[reportPrivateUsage]
    rows = await _queue_rows(db_session, "reconciler:backfill.fake")

    assert enqueued is (remaining > 0)
    assert len(rows) == (1 if remaining else 0)
    if rows:
        assert rows[0].payload == {
            "reconciler_type": "backfill",
            "backfill_name": "fake",
        }
        assert rows[0].task_type == "reconciler"


async def test_document_sources_enqueues_under_legacy_shape(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_scheduler_db(monkeypatch, db_session)
    task = RECONCILER_TASKS["backfill_document_sources"]
    legacy_key = "reconciler:backfill_document_sources"
    assert task.work_unit_key == legacy_key

    scheduler = ReconcilerScheduler()
    assert not await scheduler._try_enqueue_task(task)  # pyright: ignore[reportPrivateUsage]
    assert await _queue_rows(db_session, legacy_key) == []

    await _legacy_document(db_session)
    assert await scheduler._try_enqueue_task(task)  # pyright: ignore[reportPrivateUsage]
    rows = await _queue_rows(db_session, legacy_key)
    assert [row.payload for row in rows] == [
        {"reconciler_type": "backfill_document_sources"}
    ]


# ---------------------------------------------------------------------------
# Consumer dispatch
# ---------------------------------------------------------------------------


async def test_new_worker_processes_legacy_payload(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _capture_events(monkeypatch)
    doc_id, parent = await _legacy_document(db_session)

    await consumer.process_item(
        _reconciler_item({"reconciler_type": "backfill_document_sources"})
    )

    db_session.expire_all()
    edges = await db_session.execute(
        select(models.DocumentSource.source_id).where(
            models.DocumentSource.derived_id == doc_id
        )
    )
    assert list(edges.scalars()) == [parent]
    [event] = events
    assert isinstance(event, BackfillCompletedEvent)
    assert event.backfill_name == "document_sources"
    assert event.rows_touched == 1
    assert event.still_pending is False


async def test_generic_payload_runs_registered_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _capture_events(monkeypatch)
    rows = _Rows(5)
    monkeypatch.setattr(backfill_module, "BACKFILLS", {"fake": _fake_backfill(rows)})

    await consumer.process_item(
        _reconciler_item({"reconciler_type": "backfill", "backfill_name": "fake"})
    )

    assert rows.remaining == 0
    [event] = events
    assert isinstance(event, BackfillCompletedEvent)
    assert (event.backfill_name, event.rows_touched, event.batches) == ("fake", 5, 4)
    assert event.still_pending is False


async def test_unknown_backfill_name_is_an_error() -> None:
    with pytest.raises(ValueError, match="Unknown backfill"):
        await consumer.process_item(
            _reconciler_item({"reconciler_type": "backfill", "backfill_name": "nope"})
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"reconciler_type": "backfill"},
        {"reconciler_type": "sync_vectors", "backfill_name": "fake"},
    ],
)
def test_backfill_name_required_exactly_for_backfill(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ReconcilerPayload(**payload)


def test_backfill_work_unit_key_is_a_two_part_reconciler_key() -> None:
    key = _fake_backfill(_Rows(0)).work_unit_key
    assert key == "reconciler:backfill.fake"
    assert parse_work_unit_key(key).task_type == "reconciler"


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------


async def test_cycle_commits_each_batch_in_its_own_session() -> None:
    rows = _Rows(5)

    result = await run_backfill_cycle(_fake_backfill(rows))

    assert (result.rows_touched, result.batches, result.still_pending) == (5, 4, False)
    assert len({id(session) for session in rows.sessions}) == len(rows.sessions)


async def test_cycle_stops_at_time_budget() -> None:
    rows = _Rows(5)

    result = await run_backfill_cycle(_fake_backfill(rows, time_budget_seconds=0))

    assert (result.rows_touched, result.batches, result.still_pending) == (0, 0, True)
    assert rows.remaining == 5


# ---------------------------------------------------------------------------
# backfill_pending gauge
# ---------------------------------------------------------------------------


async def test_scheduler_loop_refreshes_backfill_gauge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refreshed = asyncio.Event()

    async def _fake_refresh() -> None:
        refreshed.set()

    async def _noop() -> None:
        return None

    async def _never_enqueue(_self: object, _task: object) -> bool:
        return False

    monkeypatch.setattr(scheduler_module, "record_backfill_pending", _fake_refresh)
    monkeypatch.setattr(scheduler_module, "record_pending_embeddings_backlog", _noop)
    monkeypatch.setattr(ReconcilerScheduler, "_try_enqueue_task", _never_enqueue)

    scheduler = ReconcilerScheduler()
    await scheduler.start()
    try:
        await asyncio.wait_for(refreshed.wait(), timeout=5.0)
    finally:
        await scheduler.shutdown()


def test_cycle_does_not_drive_the_gauge() -> None:
    referenced = run_backfill_cycle.__code__.co_names
    assert "record_backfill_pending" not in referenced
    assert "prometheus_metrics" not in referenced


async def test_record_backfill_pending_sets_gauge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def _record(*, task: str, count: int) -> None:
        calls.append((task, count))

    monkeypatch.setattr("src.config.settings.METRICS.ENABLED", True)
    monkeypatch.setattr(prometheus_metrics, "set_backfill_pending", _record)
    monkeypatch.setattr(
        backfill_module, "BACKFILLS", {"fake": _fake_backfill(_Rows(7))}
    )

    await backfill_module.record_backfill_pending()

    assert calls == [("fake", 7)]
