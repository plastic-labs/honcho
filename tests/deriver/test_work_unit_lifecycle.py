"""Regressions for ownership leaks and poison work units (GitHub #1506)."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.queue_manager import QueueManager


async def seed_claimed_work_unit(
    db: AsyncSession, session: models.Session, key: str
) -> tuple[QueueManager, str]:
    item = models.QueueItem(
        workspace_name=session.workspace_name,
        session_id=session.id,
        task_type="summary",
        work_unit_key=key,
        payload={},
        processed=False,
    )
    db.add(item)
    await db.commit()
    manager = QueueManager()
    claimed = await manager.claim_work_units(db, [key])
    await db.commit()
    worker_id = "test-worker"
    manager.track_worker_work_unit(worker_id, key, claimed[key])
    return manager, worker_id


async def assert_claim_released(
    db: AsyncSession, manager: QueueManager, key: str
) -> None:
    assert manager.get_total_owned_work_units() == 0
    assert not manager.semaphore.locked()
    claim = (
        await db.execute(
            select(models.ActiveQueueSession).where(
                models.ActiveQueueSession.work_unit_key == key
            )
        )
    ).scalar_one_or_none()
    assert claim is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_key",
    [
        "summary:workspace:session:None:None:extra",
        "summary:workspace:session",
        "representation:workspace:session:observer:observed:extra",
        "unknown:workspace:session:None:None",
    ],
)
async def test_malformed_key_is_quarantined_and_valid_work_continues(
    db_session: AsyncSession,
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    monkeypatch: pytest.MonkeyPatch,
    bad_key: str,
) -> None:
    """One malformed key must not poison a one-worker process or its restart."""
    monkeypatch.setattr(settings.DERIVER, "WORKERS", 1)
    monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", True)
    session, _ = sample_session_with_peers
    good_key = f"summary:{session.workspace_name}:{session.name}:None:None"
    oldest = datetime.now(UTC) - timedelta(minutes=1)
    bad_items = [
        models.QueueItem(
            workspace_name=session.workspace_name,
            session_id=session.id,
            task_type=bad_key.split(":")[0],
            work_unit_key=bad_key,
            payload={},
            processed=False,
            created_at=oldest,
        )
        for _ in range(2)
    ]
    already_processed = models.QueueItem(
        workspace_name=session.workspace_name,
        session_id=session.id,
        task_type="summary",
        work_unit_key=bad_key,
        payload={},
        processed=True,
    )
    good_item = models.QueueItem(
        workspace_name=session.workspace_name,
        session_id=session.id,
        task_type="summary",
        work_unit_key=good_key,
        payload={},
        processed=False,
    )
    db_session.add_all([*bad_items, already_processed, good_item])
    await db_session.commit()

    manager = QueueManager()
    claimed = await manager.get_and_claim_work_units()
    assert set(claimed) == {bad_key}
    manager.track_worker_work_unit("bad-worker", bad_key, claimed[bad_key])
    with patch(
        "src.deriver.queue_manager.publish_webhook_event", new_callable=AsyncMock
    ) as publish:
        await manager.process_work_unit(bad_key, "bad-worker")
        publish.assert_not_awaited()

    await assert_claim_released(db_session, manager, bad_key)
    for item in bad_items:
        await db_session.refresh(item)
        assert item.processed
        assert item.error is not None and item.error.startswith("ValueError:")
    await db_session.refresh(already_processed)
    assert already_processed.error is None
    await db_session.refresh(good_item)
    assert not good_item.processed
    assert good_item.error is None

    claimed = await manager.get_and_claim_work_units()
    assert set(claimed) == {good_key}
    manager.track_worker_work_unit("good-worker", good_key, claimed[good_key])
    with (
        patch(
            "src.deriver.queue_manager.process_item", new_callable=AsyncMock
        ) as process,
        patch(
            "src.deriver.queue_manager.publish_webhook_event", new_callable=AsyncMock
        ) as publish,
    ):
        await manager.process_work_unit(good_key, "good-worker")
        process.assert_awaited_once()
        publish.assert_awaited_once()
    await assert_claim_released(db_session, manager, good_key)
    await db_session.refresh(good_item)
    assert good_item.processed
    assert await QueueManager().get_and_claim_work_units() == {}


@pytest.mark.asyncio
async def test_cleanup_failure_still_releases_in_memory_ownership(
    db_session: AsyncSession,
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DERIVER, "WORKERS", 1)
    session, _ = sample_session_with_peers
    key = f"summary:{session.workspace_name}:{session.name}:None:None"
    manager, worker = await seed_claimed_work_unit(db_session, session, key)
    with (
        patch.object(manager, "get_next_queue_item", new=AsyncMock(return_value=None)),
        patch.object(
            manager,
            "_cleanup_work_unit",
            new=AsyncMock(side_effect=ConnectionError("cleanup unavailable")),
        ),
        pytest.raises(ConnectionError, match="cleanup unavailable"),
    ):
        await manager.process_work_unit(key, worker)
    assert manager.get_total_owned_work_units() == 0
    assert not manager.semaphore.locked()


@pytest.mark.asyncio
async def test_cancellation_waiting_for_semaphore_releases_claim(
    db_session: AsyncSession,
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DERIVER, "WORKERS", 1)
    session, _ = sample_session_with_peers
    key = f"summary:{session.workspace_name}:{session.name}:None:None"
    manager, worker = await seed_claimed_work_unit(db_session, session, key)
    await manager.semaphore.acquire()
    task = asyncio.create_task(manager.process_work_unit(key, worker))
    try:
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        manager.semaphore.release()
    await assert_claim_released(db_session, manager, key)


@pytest.mark.asyncio
async def test_failed_quarantine_releases_claim_without_discarding_items(
    db_session: AsyncSession,
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
) -> None:
    session, _ = sample_session_with_peers
    key = "summary:workspace:session:None:None:extra"
    manager, worker = await seed_claimed_work_unit(db_session, session, key)
    with (
        patch.object(
            manager,
            "_mark_invalid_work_unit_as_errored",
            new=AsyncMock(side_effect=ConnectionError("quarantine unavailable")),
        ),
        pytest.raises(ConnectionError, match="quarantine unavailable"),
    ):
        await manager.process_work_unit(key, worker)
    await assert_claim_released(db_session, manager, key)
    item = (
        await db_session.execute(
            select(models.QueueItem).where(models.QueueItem.work_unit_key == key)
        )
    ).scalar_one()
    assert not item.processed
    assert item.error is None


@pytest.mark.asyncio
async def test_cancellation_during_processing_releases_claim(
    db_session: AsyncSession,
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
) -> None:
    session, _ = sample_session_with_peers
    key = f"summary:{session.workspace_name}:{session.name}:None:None"
    manager, worker = await seed_claimed_work_unit(db_session, session, key)
    started = asyncio.Event()

    async def wait_for_cancellation(*_: object) -> None:
        started.set()
        await asyncio.Event().wait()

    with patch.object(
        manager, "get_next_queue_item", new=AsyncMock(side_effect=wait_for_cancellation)
    ):
        task = asyncio.create_task(manager.process_work_unit(key, worker))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await assert_claim_released(db_session, manager, key)


@pytest.mark.asyncio
async def test_cancellation_during_cleanup_releases_in_memory_ownership(
    db_session: AsyncSession,
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
) -> None:
    session, _ = sample_session_with_peers
    key = f"summary:{session.workspace_name}:{session.name}:None:None"
    manager, worker = await seed_claimed_work_unit(db_session, session, key)
    with (
        patch.object(manager, "get_next_queue_item", new=AsyncMock(return_value=None)),
        patch.object(
            manager,
            "_cleanup_work_unit",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await manager.process_work_unit(key, worker)
    assert manager.get_total_owned_work_units() == 0
    assert not manager.semaphore.locked()


@pytest.mark.asyncio
async def test_task_exception_is_logged_when_task_finishes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail() -> None:
        raise RuntimeError("unexpected worker failure")

    manager = QueueManager()
    task = asyncio.create_task(fail())
    with caplog.at_level("ERROR", logger="src.deriver.queue_manager"):
        manager.add_task(task)
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    assert not manager.active_tasks
    assert "unexpected worker failure" in caplog.text


@pytest.mark.asyncio
async def test_cancelled_task_is_not_logged_as_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = QueueManager()
    task = asyncio.create_task(asyncio.sleep(60))
    manager.add_task(task)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert not manager.active_tasks
    assert not [record for record in caplog.records if record.levelname == "ERROR"]
