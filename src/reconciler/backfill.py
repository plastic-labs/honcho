"""Predicate-driven backfills that run as reconciler tasks.

A backfill's progress is its own pending predicate: there is no cursor or
state table. The scheduler enqueues it only while ``has_pending`` holds, the
consumer drains it in short committed batches, and every replica exports the
remaining count as ``backfill_pending{task=<name>}``.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.dependencies import tracked_db
from src.reconciler.backfill_document_sources import (
    BACKFILL_BATCH_SIZE,
    count_pending_document_sources,
    drain_document_sources_batch,
    has_pending_document_sources,
)
from src.schemas import ReconcilerType
from src.telemetry import prometheus_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Backfill:
    """A named, batched, idempotent copy of legacy data into a new shape."""

    name: str
    has_pending: Callable[[AsyncSession], Awaitable[bool]]
    count_pending: Callable[[AsyncSession], Awaitable[int]]
    # Processes one batch and returns the rows it touched; 0 means drained.
    run_batch: Callable[[AsyncSession, int], Awaitable[int]]
    interval_seconds: int
    batch_size: int = 500
    time_budget_seconds: float = 240.0
    pause_seconds: float = 0.0
    # Enqueue under this pre-generic reconciler type so older workers can run it.
    legacy_reconciler_type: ReconcilerType | None = None

    @property
    def work_unit_key(self) -> str:
        if self.legacy_reconciler_type is not None:
            return f"reconciler:{self.legacy_reconciler_type.value}"
        return f"reconciler:backfill.{self.name}"

    @property
    def payload(self) -> dict[str, Any]:
        if self.legacy_reconciler_type is not None:
            return {"reconciler_type": self.legacy_reconciler_type.value}
        return {
            "reconciler_type": ReconcilerType.BACKFILL.value,
            "backfill_name": self.name,
        }


@dataclass(frozen=True, slots=True)
class BackfillCycleResult:
    """Outcome of one time-budgeted backfill cycle."""

    rows_touched: int
    batches: int
    duration_ms: float
    still_pending: bool


BACKFILLS: dict[str, Backfill] = {
    backfill.name: backfill
    for backfill in (
        # Retire once no pre-3.2.0 writer of documents.source_ids is live on any
        # deployment and backfill_pending{task="document_sources"} is 0.
        Backfill(
            name="document_sources",
            has_pending=has_pending_document_sources,
            count_pending=count_pending_document_sources,
            run_batch=drain_document_sources_batch,
            interval_seconds=settings.VECTOR_STORE.RECONCILIATION_INTERVAL_SECONDS,
            batch_size=BACKFILL_BATCH_SIZE,
            legacy_reconciler_type=ReconcilerType.BACKFILL_DOCUMENT_SOURCES,
        ),
    )
}


def resolve_backfill(
    reconciler_type: ReconcilerType, backfill_name: str | None
) -> Backfill | None:
    """Backfill a reconciler payload refers to, or None if it is not a backfill."""
    if reconciler_type == ReconcilerType.BACKFILL:
        if backfill_name is None or backfill_name not in BACKFILLS:
            raise ValueError(f"Unknown backfill: {backfill_name}")
        return BACKFILLS[backfill_name]
    for backfill in BACKFILLS.values():
        if backfill.legacy_reconciler_type == reconciler_type:
            return backfill
    return None


async def run_backfill_cycle(backfill: Backfill) -> BackfillCycleResult:
    """Run batches until the backfill drains or its time budget is spent.

    Each batch commits in its own short-lived session.
    """
    start = time.monotonic()
    deadline = start + backfill.time_budget_seconds
    rows_touched = 0
    batches = 0
    while time.monotonic() < deadline:
        async with tracked_db(f"backfill_{backfill.name}") as db:
            count = await backfill.run_batch(db, backfill.batch_size)
            await db.commit()
        if count == 0:
            break
        batches += 1
        rows_touched += count
        if backfill.pause_seconds > 0:
            await asyncio.sleep(backfill.pause_seconds)

    async with tracked_db(f"backfill_{backfill.name}_check", read_only=True) as db:
        still_pending = await backfill.has_pending(db)

    if rows_touched:
        logger.info("Backfill %s touched %d rows", backfill.name, rows_touched)
    return BackfillCycleResult(
        rows_touched=rows_touched,
        batches=batches,
        duration_ms=(time.monotonic() - start) * 1000,
        still_pending=still_pending,
    )


async def record_backfill_pending() -> None:
    """Set ``backfill_pending`` for every registered backfill."""
    if not settings.METRICS.ENABLED:
        return
    for backfill in BACKFILLS.values():
        try:
            async with tracked_db("backfill_pending_count", read_only=True) as db:
                count = await backfill.count_pending(db)
            prometheus_metrics.set_backfill_pending(task=backfill.name, count=count)
        except Exception:
            logger.warning(
                "Failed to record backfill_pending for %s", backfill.name, exc_info=True
            )
