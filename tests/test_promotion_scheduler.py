"""Tests for the graph-memory promotion scheduler.

These tests exercise the real scheduler scan against a per-test database.
They encode the behaviour the scheduler *should* have: enqueue observations
that have not been promoted, skip already-promoted ones, and respect the
``_PROMOTION_PROCESSING_ENABLED`` flag.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver import promotion_scheduler as scheduler_mod
from src.deriver.promotion_scheduler import PromotionScheduler
from tests.fixtures.graph_memory_fixtures import (  # noqa: F401
    _rewrite_db_host_for_graph_memory_tests,
    clean_graph_memory_queue_tables,
    graph_memory_setup,
)


@pytest.fixture
def scheduler() -> PromotionScheduler:
    return PromotionScheduler()


async def _count_queue_items(db_session: AsyncSession) -> int:
    result = await db_session.execute(select(func.count()).select_from(models.QueueItem))
    return result.scalar() or 0


@pytest.mark.asyncio
async def test_scheduler_enqueues_when_ready_flag_true(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    scheduler: PromotionScheduler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _PROMOTION_PROCESSING_ENABLED is True, observations are enqueued."""
    setup = graph_memory_setup
    workspace = setup["workspace"]

    # Make all observations old enough to pass the promotion delay.
    await db_session.execute(
        select(models.Document)
        .where(models.Document.workspace_name == workspace.name)
    )
    await db_session.execute(
        update(models.Document)
        .where(models.Document.workspace_name == workspace.name)
        .values(created_at=datetime.now(timezone.utc) - timedelta(seconds=30))
    )
    await db_session.commit()

    monkeypatch.setattr(scheduler_mod, "_PROMOTION_PROCESSING_ENABLED", True)
    monkeypatch.setattr(scheduler_mod, "PROMOTION_DELAY_SECONDS", 0)

    await scheduler._scan_and_enqueue()

    queued = await _count_queue_items(db_session)
    assert queued == len(setup["all_docs"]), f"expected {len(setup['all_docs'])} queue items, got {queued}"


@pytest.mark.asyncio
async def test_scheduler_does_not_enqueue_when_ready_flag_false(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    scheduler: PromotionScheduler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _PROMOTION_PROCESSING_ENABLED is False, nothing is enqueued."""
    setup = graph_memory_setup
    workspace = setup["workspace"]

    await db_session.execute(
        update(models.Document)
        .where(models.Document.workspace_name == workspace.name)
        .values(created_at=datetime.now(timezone.utc) - timedelta(seconds=30))
    )
    await db_session.commit()

    monkeypatch.setattr(scheduler_mod, "_PROMOTION_PROCESSING_ENABLED", False)
    monkeypatch.setattr(scheduler_mod, "PROMOTION_DELAY_SECONDS", 0)

    await scheduler._scan_and_enqueue()

    queued = await _count_queue_items(db_session)
    assert queued == 0, f"expected 0 queue items when flag is False, got {queued}"


@pytest.mark.asyncio
async def test_scheduler_skips_already_promoted_observations(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    scheduler: PromotionScheduler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observations that already have a 'promote' access-log event are skipped."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    collection_name = setup["collection_name"]
    promoted_doc = setup["all_docs"][0]

    await db_session.execute(
        update(models.Document)
        .where(models.Document.workspace_name == workspace.name)
        .values(created_at=datetime.now(timezone.utc) - timedelta(seconds=30))
    )

    # Mark the first document as already promoted.
    db_session.add(
        models.AccessLogEntry(
            workspace_name=workspace.name,
            collection_name=collection_name,
            obs_id=promoted_doc.id,
            event_type="promote",
            created_by="promotion-worker",
        )
    )
    await db_session.commit()

    monkeypatch.setattr(scheduler_mod, "_PROMOTION_PROCESSING_ENABLED", True)
    monkeypatch.setattr(scheduler_mod, "PROMOTION_DELAY_SECONDS", 0)

    await scheduler._scan_and_enqueue()

    queued = await _count_queue_items(db_session)
    expected = len(setup["all_docs"]) - 1
    assert queued == expected, f"expected {expected} queue items, got {queued}"


@pytest.mark.asyncio
async def test_scheduler_handles_empty_observation_set(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    scheduler: PromotionScheduler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A workspace with no observations should not crash and should enqueue nothing."""
    monkeypatch.setattr(scheduler_mod, "_PROMOTION_PROCESSING_ENABLED", True)
    monkeypatch.setattr(scheduler_mod, "PROMOTION_DELAY_SECONDS", 0)

    await scheduler._scan_and_enqueue()

    queued = await _count_queue_items(db_session)
    assert queued == 0
