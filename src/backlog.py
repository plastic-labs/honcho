"""Publishes the deriver backlog as Prometheus gauges from the API process."""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import sentry_sdk
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer import execute_dream
from src.schemas import DreamType
from src.telemetry.prometheus import metrics as prometheus_metrics
from src.utils.work_unit import construct_work_unit_key

logger = logging.getLogger(__name__)


@dataclass
class DueDream:
    workspace_name: str
    observer: str
    observed: str
    dream_type: DreamType
    documents_since_last_dream: int


class BacklogMetricsPoller:
    """Refreshes the deriver-backlog gauges on a timer."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._next_dream_poll: datetime | None = None

    async def start(self) -> None:
        if self._task is not None:
            logger.warning("BacklogMetricsPoller already running")
            return

        interval = settings.DERIVER.BACKLOG_METRICS_POLL_INTERVAL_SECONDS
        self._task = asyncio.create_task(self._loop())
        logger.info("BacklogMetricsPoller started (interval=%ss)", interval)

    async def shutdown(self) -> None:
        if self._task is None:
            return

        logger.info("Shutting down BacklogMetricsPoller...")
        self._shutdown_event.set()

        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("BacklogMetricsPoller shutdown timed out, cancelling task")
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        self._task = None
        logger.info("BacklogMetricsPoller stopped")

    async def _loop(self) -> None:
        interval = settings.DERIVER.BACKLOG_METRICS_POLL_INTERVAL_SECONDS

        while not self._shutdown_event.is_set():
            try:
                await self._refresh()
            except Exception as e:
                logger.error("BacklogMetricsPoller refresh failed: %s", e)
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=float(interval)
                )
            except asyncio.TimeoutError:
                continue

    async def _refresh(self) -> None:
        due_dreams: list[DueDream] | None = None

        async with tracked_db("backlog_metrics", read_only=True) as db:
            backlog = await crud.get_deriver_backlog(db)
            prometheus_metrics.prometheus_metrics.set_deriver_backlog(
                eligible_work_units=backlog.eligible_work_units,
                pending_items=backlog.pending_items,
                oldest_pending_age_seconds=backlog.oldest_pending_age_seconds,
            )

            prometheus_metrics.prometheus_metrics.set_seconds_since_last_vector_sync(
                seconds=await _seconds_since_last_vector_sync(db)
            )

            if self._dream_poll_due():
                due_dreams = await find_due_dreams(db)

        if due_dreams is None:
            return

        prometheus_metrics.prometheus_metrics.set_dreams_pending(count=len(due_dreams))

        for dream in due_dreams:
            try:
                await execute_dream(
                    dream.workspace_name,
                    dream.dream_type,
                    observer=dream.observer,
                    observed=dream.observed,
                    trigger_reason="document_threshold",
                    delay_reason="idle_timeout",
                    documents_since_last_dream_at_schedule=dream.documents_since_last_dream,
                    document_threshold=settings.DREAM.DOCUMENT_THRESHOLD,
                )
            except Exception as e:
                logger.error(
                    "Failed to enqueue dream for %s/%s/%s: %s",
                    dream.workspace_name,
                    dream.observer,
                    dream.observed,
                    e,
                )
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)

    def _dream_poll_due(self) -> bool:
        now = datetime.now(timezone.utc)
        if self._next_dream_poll is not None and now < self._next_dream_poll:
            return False
        self._next_dream_poll = now + timedelta(
            seconds=settings.DREAM.POLL_INTERVAL_SECONDS
        )
        return True


async def _seconds_since_last_vector_sync(db: AsyncSession) -> float:
    """-1 when nothing has ever synced"""
    newest = await db.scalar(select(func.max(models.MessageEmbedding.last_sync_at)))
    if newest is None:
        return -1.0
    return (datetime.now(timezone.utc) - newest).total_seconds()


async def find_due_dreams(db: AsyncSession) -> list[DueDream]:
    """Collections whose next dream is due, read-only.

    A dream is due when the collection has gained DOCUMENT_THRESHOLD explicit
    documents since its last dream, has been quiet for IDLE_TIMEOUT_MINUTES, is
    past the MIN_HOURS_BETWEEN_DREAMS gate, and has no dream queued or attempted
    since its newest explicit document.
    """
    dream_types = [
        DreamType(dream_type)
        for dream_type in settings.DREAM.ENABLED_TYPES
        if dream_type == DreamType.OMNI.value
    ]
    if not settings.DREAM.ENABLED or not dream_types:
        return []

    explicit_counts = (
        select(
            models.Document.workspace_name,
            models.Document.observer,
            models.Document.observed,
            func.count(models.Document.id).label("explicit_count"),
            func.max(models.Document.created_at).label("newest_created_at"),
        )
        .where(models.Document.level == "explicit")
        .group_by(
            models.Document.workspace_name,
            models.Document.observer,
            models.Document.observed,
        )
        .subquery()
    )

    rows = (
        await db.execute(
            select(
                models.Collection.workspace_name,
                models.Collection.observer,
                models.Collection.observed,
                models.Collection.internal_metadata,
                func.coalesce(explicit_counts.c.explicit_count, 0),
                explicit_counts.c.newest_created_at,
            ).outerjoin(
                explicit_counts,
                (models.Collection.workspace_name == explicit_counts.c.workspace_name)
                & (models.Collection.observer == explicit_counts.c.observer)
                & (models.Collection.observed == explicit_counts.c.observed),
            )
        )
    ).all()

    now = datetime.now(timezone.utc)
    idle_cutoff = now - timedelta(minutes=settings.DREAM.IDLE_TIMEOUT_MINUTES)
    candidates: dict[str, tuple[DueDream, datetime]] = {}

    for row in rows:
        workspace_name = cast(str, row[0])
        observer = cast(str, row[1])
        observed = cast(str, row[2])
        internal_metadata = cast("dict[str, Any] | None", row[3])
        explicit_count = cast(int, row[4])
        newest_created_at = cast("datetime | None", row[5])

        dream_metadata: dict[str, Any] = (internal_metadata or {}).get("dream", {})
        since_last_dream = explicit_count - int(
            dream_metadata.get("last_dream_document_count", 0)
        )
        if since_last_dream < settings.DREAM.DOCUMENT_THRESHOLD:
            continue

        if newest_created_at is None or newest_created_at > idle_cutoff:
            continue

        last_dream_at = cast("str | None", dream_metadata.get("last_dream_at"))
        if last_dream_at and _within_min_hours_gate(last_dream_at, now):
            continue

        for dream_type in dream_types:
            work_unit_key = construct_work_unit_key(
                workspace_name,
                {
                    "task_type": "dream",
                    "observer": observer,
                    "observed": observed,
                    "dream_type": dream_type.value,
                },
            )
            candidates[work_unit_key] = (
                DueDream(
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    dream_type=dream_type,
                    documents_since_last_dream=since_last_dream,
                ),
                newest_created_at,
            )

    if not candidates:
        return []

    attempt_rows = (
        await db.execute(
            select(
                models.QueueItem.work_unit_key,
                func.max(models.QueueItem.created_at),
            )
            .where(
                models.QueueItem.task_type == "dream",
                models.QueueItem.work_unit_key.in_(candidates.keys()),
            )
            .group_by(models.QueueItem.work_unit_key)
        )
    ).all()
    newest_attempts: dict[str, datetime] = {
        cast(str, row[0]): cast(datetime, row[1]) for row in attempt_rows
    }

    return [
        due
        for work_unit_key, (due, newest_created_at) in candidates.items()
        if work_unit_key not in newest_attempts
        or newest_attempts[work_unit_key] < newest_created_at
    ]


def _within_min_hours_gate(last_dream_at: str, now: datetime) -> bool:
    """True when the last dream is too recent for another one."""
    try:
        last_dream_time = datetime.fromisoformat(last_dream_at)
    except (ValueError, TypeError):
        return False

    hours_since = (now - last_dream_time).total_seconds() / 3600
    return hours_since < settings.DREAM.MIN_HOURS_BETWEEN_DREAMS
