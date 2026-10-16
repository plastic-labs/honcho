"""Progress lifecycle uses persisted attempts, never a process-local status map."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import models
from src.crud.deriver import cleanup_stale_work_units
from src.deriver.queue_manager import QueueManager
from src.utils.work_unit_progress import work_unit_progress


@pytest.mark.asyncio
async def test_progress_is_visible_on_independent_connections_without_renewing_lease(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = async_sessionmaker(db_engine, expire_on_commit=False)

    @asynccontextmanager
    async def independent_db(_name: str):
        async with sessions() as db:
            yield db

    monkeypatch.setattr("src.utils.work_unit_progress.tracked_db", independent_db)
    claim = models.ActiveQueueSession(work_unit_key="dream:omni:test:alice:bob")
    db_session.add(claim)
    await db_session.commit()
    await db_session.refresh(claim)
    started, lease = claim.started_at, claim.last_updated
    report = work_unit_progress(claim.work_unit_key, claim.id, queue_item_id=123)
    await report("deduction")
    async with sessions() as reader:
        visible = (
            await reader.execute(
                select(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.id == claim.id
                )
            )
        ).scalar_one()
        assert visible.stage == "deduction"
        assert visible.started_at == started
        assert visible.last_updated == lease
        assert visible.last_activity_at is not None and started is not None
        assert visible.last_activity_at >= started
    await report("induction")
    await db_session.refresh(claim)
    assert claim.started_at == started
    assert claim.last_updated == lease
    assert claim.stage == "induction"
    observed = (
        claim.started_at,
        claim.stage,
        claim.last_activity_at,
        claim.current_queue_item_id,
    )
    # Existing lease-only writes must not fabricate observed progress either.
    await db_session.execute(
        update(models.ActiveQueueSession)
        .where(models.ActiveQueueSession.id == claim.id)
        .values(last_updated=func.now())
    )
    await db_session.commit()
    await db_session.refresh(claim)
    assert (
        claim.started_at,
        claim.stage,
        claim.last_activity_at,
        claim.current_queue_item_id,
    ) == observed


@pytest.mark.asyncio
async def test_replaced_claim_cannot_be_updated_or_cleaned_by_old_worker(
    db_session: AsyncSession,
) -> None:
    key = "dream:omni:workspace:alice:bob"
    old = models.ActiveQueueSession(work_unit_key=key)
    db_session.add(old)
    await db_session.commit()
    old_id = old.id
    report = work_unit_progress(key, old_id)
    await db_session.execute(
        update(models.ActiveQueueSession)
        .where(models.ActiveQueueSession.id == old_id)
        .values(last_updated=datetime.now(UTC) - timedelta(days=1))
    )
    await db_session.commit()
    await cleanup_stale_work_units(db_session)
    await db_session.commit()
    replacement = models.ActiveQueueSession(work_unit_key=key)
    db_session.add(replacement)
    await db_session.commit()
    await db_session.refresh(replacement)
    original = (
        replacement.started_at,
        replacement.last_updated,
        replacement.last_activity_at,
    )
    await report("induction")
    assert not await QueueManager()._cleanup_work_unit(old_id, key)  # pyright: ignore[reportPrivateUsage]
    await db_session.refresh(replacement)
    assert replacement.stage == "claimed"
    assert (
        replacement.started_at,
        replacement.last_updated,
        replacement.last_activity_at,
    ) == original
    assert replacement.id != old_id
    # The current owner can clean up. A late callback cannot recreate its row.
    assert await QueueManager()._cleanup_work_unit(replacement.id, key)  # pyright: ignore[reportPrivateUsage]
    await work_unit_progress(key, replacement.id)("deduction")
    assert (
        await db_session.execute(
            select(models.ActiveQueueSession).where(
                models.ActiveQueueSession.work_unit_key == key
            )
        )
    ).scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_progress_failures_are_soft_but_cancellation_propagates() -> None:
    for error in (RuntimeError("unavailable"), asyncio.CancelledError()):
        with patch("src.utils.work_unit_progress.tracked_db") as tracked:
            tracked.return_value.__aenter__ = AsyncMock(side_effect=error)
            if isinstance(error, asyncio.CancelledError):
                with pytest.raises(asyncio.CancelledError):
                    await work_unit_progress("key", "claim")("deduction")
            else:
                await work_unit_progress("key", "claim")("deduction")


@pytest.mark.asyncio
async def test_cancelled_worker_removes_only_its_claim(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    from src.deriver.queue_manager import WorkerOwnership

    workspace, peer = sample_data
    key = f"dream:omni:{workspace.name}:{peer.name}:{peer.name}"
    queue_item = models.QueueItem(
        work_unit_key=key,
        workspace_name=workspace.name,
        task_type="dream",
        payload={
            "task_type": "dream",
            "dream_type": "omni",
            "observer": peer.name,
            "observed": peer.name,
        },
        processed=False,
    )
    claim = models.ActiveQueueSession(work_unit_key=key)
    db_session.add_all([queue_item, claim])
    await db_session.commit()
    manager = QueueManager()
    manager.worker_ownership["cancelled"] = WorkerOwnership(
        work_unit_key=key, aqs_id=claim.id
    )
    with (
        patch(
            "src.deriver.queue_manager.process_item",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError(),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await manager.process_work_unit(key, "cancelled")
    await db_session.refresh(queue_item)
    assert queue_item.processed is False
    assert (
        await db_session.execute(
            select(models.ActiveQueueSession).where(
                models.ActiveQueueSession.id == claim.id
            )
        )
    ).scalar_one_or_none() is None
    assert "cancelled" not in manager.worker_ownership
