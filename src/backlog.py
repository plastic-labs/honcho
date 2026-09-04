"""Polling of the deriver's outstanding work. Schedules only when it owns scheduling."""

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from logging import getLogger

import sentry_sdk

from src import crud, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.deriver.enqueue import enqueue_dream
from src.dreamer.dream_due import DueDream, list_due_dreams
from src.telemetry import prometheus_metrics

logger = getLogger(__name__)


def active_work_seconds() -> float:
    """The value reported when work is ready for a deriver now."""
    return float(max(settings.DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS, 1))


@dataclass
class DeriverMetricsSnapshot:
    """The last good poll result, served to callers of the route."""

    signal_seconds: float = 0.0
    dreams_due: int = 0
    stats: schemas.DeriverMetrics = field(default_factory=schemas.DeriverMetrics)
    measured_at: float | None = None

    @property
    def age_seconds(self) -> float | None:
        if self.measured_at is None:
            return None
        return max(0.0, time.time() - self.measured_at)


def outstanding_work_seconds(
    stats: schemas.DeriverMetrics, *, dreams_due: int
) -> float:
    """Seconds of outstanding deriver work, 0 when there is nothing to do."""
    if (
        stats.eligible_work_units > 0
        or stats.claimed_work_units > 0
        or stats.embeddings_pending_due > 0
        or dreams_due > 0
    ):
        return active_work_seconds()
    if stats.pending_items > 0:
        return stats.oldest_pending_age_seconds
    return 0.0


class DeriverMetricsPoller:
    """Refreshes the deriver gauges and the cached snapshot on a timer."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._snapshot: DeriverMetricsSnapshot = DeriverMetricsSnapshot()
        self._next_dream_poll: float | None = None
        self._due_dreams: list[DueDream] = []
        self._next_stale_cleanup: float | None = None

    @property
    def snapshot(self) -> DeriverMetricsSnapshot:
        return self._snapshot

    async def start(self) -> None:
        if self._task is not None:
            logger.warning("DeriverMetricsPoller already running")
            return
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "DeriverMetricsPoller started, interval %ss",
            settings.DERIVER.BACKLOG_METRICS_POLL_INTERVAL_SECONDS,
        )

    async def shutdown(self) -> None:
        if self._task is None:
            return
        logger.info("Shutting down DeriverMetricsPoller...")
        self._shutdown_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except TimeoutError:
            logger.warning("DeriverMetricsPoller shutdown timed out, cancelling task")
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        logger.info("DeriverMetricsPoller stopped")

    async def _loop(self) -> None:
        interval = settings.DERIVER.BACKLOG_METRICS_POLL_INTERVAL_SECONDS
        while not self._shutdown_event.is_set():
            try:
                await self.refresh()
            except Exception as e:
                logger.error("DeriverMetricsPoller refresh failed: %s", e)
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)
            if settings.DERIVER.SCHEDULER == "api":
                try:
                    await self._maybe_cleanup_stale_work_units()
                except Exception as e:
                    logger.error("Stale work unit cleanup failed: %s", e)
                    if settings.SENTRY.ENABLED:
                        sentry_sdk.capture_exception(e)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)

    async def refresh(self) -> None:
        """One pass. The snapshot only advances on a complete pass."""
        dreams_refreshed = False
        async with tracked_db("deriver_metrics", read_only=True) as db:
            stats = await crud.get_deriver_metrics(db)
            if self._dream_poll_due():
                self._due_dreams = await list_due_dreams(db)
                self._next_dream_poll = (
                    time.monotonic() + settings.DREAM.DUE_POLL_INTERVAL_SECONDS
                )
                dreams_refreshed = True

        if dreams_refreshed and settings.DERIVER.SCHEDULER == "api":
            await self._enqueue_due_dreams()

        signal = outstanding_work_seconds(stats, dreams_due=len(self._due_dreams))
        measured_at = time.time()

        self._snapshot = DeriverMetricsSnapshot(
            signal_seconds=signal,
            dreams_due=len(self._due_dreams),
            stats=stats,
            measured_at=measured_at,
        )

        metrics = prometheus_metrics
        metrics.set_deriver_metrics(
            eligible_work_units=stats.eligible_work_units,
            claimed_work_units=stats.claimed_work_units,
            pending_items=stats.pending_items,
            oldest_pending_age_seconds=stats.oldest_pending_age_seconds,
            embeddings_pending=stats.embeddings_pending,
            embeddings_pending_due=stats.embeddings_pending_due,
        )
        metrics.set_dreams_due(count=len(self._due_dreams))
        metrics.set_deriver_outstanding_work(seconds=signal)
        metrics.set_deriver_metrics_last_success(timestamp=measured_at)

    def _dream_poll_due(self) -> bool:
        """The dream query is far more expensive, so it runs on its own spacing."""
        return (
            self._next_dream_poll is None or time.monotonic() >= self._next_dream_poll
        )

    async def _enqueue_due_dreams(self) -> None:
        for due_dream in self._due_dreams:
            try:
                await enqueue_dream(
                    due_dream.workspace_name,
                    observer=due_dream.observer,
                    observed=due_dream.observed,
                    dream_type=due_dream.dream_type,
                    session_name=due_dream.session_name,
                    trigger_reason="document_threshold",
                    delay_reason="api_poll",
                )
            except Exception as e:
                logger.error(
                    "Failed to enqueue dream for %s/%s/%s: %s",
                    due_dream.workspace_name,
                    due_dream.observer,
                    due_dream.observed,
                    e,
                )
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)

    async def _maybe_cleanup_stale_work_units(self) -> None:
        interval = settings.DERIVER.STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS
        if (
            self._next_stale_cleanup is not None
            and time.monotonic() < self._next_stale_cleanup
        ):
            return
        self._next_stale_cleanup = time.monotonic() + interval
        async with tracked_db("cleanup_stale_work_units") as db:
            await crud.cleanup_stale_work_units(db)
