"""Polling of the deriver's outstanding work. Schedules only when it owns scheduling."""

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from logging import getLogger

import sentry_sdk

from src import crud, schemas
from src.config import settings
from src.db import tenant_context
from src.dependencies import service_db
from src.deriver.enqueue import enqueue_dream
from src.dreamer.dream_due import (
    CollectionKey,
    CountMemo,
    DueDream,
    list_due_dreams,
    sweep_slots_for_ticks,
    sweep_tick_at,
)
from src.scheduler_lease import AdvisoryLease
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


DREAM_DUE_LEASE = "dream_due"

# Sweep finds past MAX_ENQUEUED_PER_POLL carried to the next poll, at most this
# many times the per-poll cap; the rest wait for their slot's next cycle.
SWEEP_CARRY_POLLS = 10
# How many missed sweep ticks the holder catches up on in one poll.
SWEEP_CATCH_UP_TICKS = 3
# Wake just past the tick boundary, so the clock reads the new tick.
_TICK_MARGIN_SECONDS = 1.0


def seconds_to_next_tick(epoch_seconds: float) -> float:
    """Seconds from this time until just past the next poll tick boundary."""
    interval = settings.DREAM.DUE_POLL_INTERVAL_SECONDS
    return interval - epoch_seconds % interval + _TICK_MARGIN_SECONDS


def ticks_to_sweep(last_tick: int | None, tick: int) -> range:
    """The ticks a poll at `tick` sweeps, given the last tick swept."""
    # region ai
    # A poll timed by "wait one interval after the last poll" drifts by the poll's
    # own duration and the clock-derived slot then jumps two at a time, skipping
    # one. The loop sleeps to tick boundaries instead, and this catches up on the
    # ticks since the last successful sweep (at most SWEEP_CATCH_UP_TICKS), so an
    # overrun or a failed poll doesn't skip slots. A tick already swept is not
    # swept again; with no history (a new holder) only the current tick is.
    # endregion
    if last_tick is None:
        return range(tick, tick + 1)
    return range(max(last_tick + 1, tick - SWEEP_CATCH_UP_TICKS + 1), tick + 1)


class DreamDuePoller:
    """Finds due dreams on its own timer and, when it owns scheduling, enqueues them."""

    # region ai
    # Its own task, apart from DeriverMetricsPoller's 30 s loop: the dream poll is
    # the expensive query, and inside that loop a slow poll froze
    # deriver_metrics_last_success and every KEDA input with it. Under
    # DERIVER.SCHEDULER == "api" only the replica holding the dream_due advisory
    # lease polls, enqueues and reports dreams_due; every other replica reports 0,
    # and KEDA wakes the deriver from the eligible work units the holder enqueues.
    # In deriver mode there is no lease: every API replica keeps measuring
    # dreams_due as a wake signal and enqueues nothing, and the slow sweep, which
    # only an enqueuing poller needs, stays off.
    # endregion

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._due_dreams: list[DueDream] = []
        self._last_sweep_tick: int | None = None
        self._count_memo: CountMemo = {}
        self._revisit: list[CollectionKey] = []
        self._lease: AdvisoryLease | None = (
            AdvisoryLease("honcho:dream-due-poller")
            if settings.DERIVER.SCHEDULER == "api"
            else None
        )

    @property
    def dreams_due(self) -> int:
        return len(self._due_dreams)

    async def start(self) -> None:
        if self._task is not None:
            logger.warning("DreamDuePoller already running")
            return
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "DreamDuePoller started, interval %ss",
            settings.DREAM.DUE_POLL_INTERVAL_SECONDS,
        )

    async def shutdown(self) -> None:
        if self._task is not None:
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except TimeoutError:
                logger.warning("DreamDuePoller shutdown timed out, cancelling task")
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._task = None
        if self._lease is not None:
            await self._lease.release()
            prometheus_metrics.set_scheduler_lease_held(
                lease=DREAM_DUE_LEASE, held=False
            )

    async def _loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await self.poll()
                wait = seconds_to_next_tick(time.time())
            except Exception as e:
                logger.error("DreamDuePoller poll failed: %s", e)
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)
                # ai: a failed poll retries on the metrics interval, not at the next tick
                wait = min(
                    seconds_to_next_tick(time.time()),
                    settings.DERIVER.BACKLOG_METRICS_POLL_INTERVAL_SECONDS,
                )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=wait)

    async def poll(self) -> None:
        """One poll: refresh the due list and dreams_due, and enqueue when scheduling."""
        scheduling = self._lease is not None
        if self._lease is not None:
            try:
                held = await self._lease.ensure()
            finally:
                prometheus_metrics.set_scheduler_lease_held(
                    lease=DREAM_DUE_LEASE, held=self._lease.held
                )
            if not held:
                self._due_dreams = []
                self._last_sweep_tick = None
                self._count_memo.clear()
                self._revisit = []
                prometheus_metrics.set_dreams_due(count=0)
                return

        # region ai
        # Cross-tenant: the due-dream scan reads every tenant's documents, so it runs
        # on the RLS-bypass service session. Under MULTI_TENANT tracked_db would fail
        # closed here (this runs on API boot with no request tenant in scope).
        # read_only: the enqueue below opens its own sessions.
        # endregion
        tick = sweep_tick_at(time.time())
        sweep = (
            sweep_slots_for_ticks(ticks_to_sweep(self._last_sweep_tick, tick))
            if scheduling
            else []
        )
        # ai: bounded so a wedged connection fails the poll instead of stalling the loop
        async with (
            asyncio.timeout(settings.DREAM.DUE_POLL_INTERVAL_SECONDS),
            service_db("dream_due_poll", read_only=True) as db,
        ):
            due = await list_due_dreams(
                db,
                sweep=sweep,
                revisit=self._revisit if scheduling else (),
                count_memo=self._count_memo,
            )
        if scheduling:
            self._last_sweep_tick = tick
        self._due_dreams = due
        prometheus_metrics.set_dreams_due(count=len(due))
        prometheus_metrics.record_dream_due_sweep_found(
            count=sum(1 for due_dream in due if due_dream.found_by_sweep)
        )
        if scheduling:
            await self._enqueue_due_dreams()

    async def _enqueue_due_dreams(self) -> None:
        cap = settings.DREAM.MAX_ENQUEUED_PER_POLL
        # region ai
        # A window find past the cap is seen again next poll; a sweep find isn't
        # until its slot comes round a cycle later, so the overflow is carried and
        # rechecked (it may have been dreamed or changed meanwhile). Due dreams sort
        # oldest first, so carried finds go ahead of window finds until drained.
        # endregion
        self._revisit = list(
            dict.fromkeys(
                (d.tenant_id, d.workspace_name, d.observer, d.observed)
                for d in self._due_dreams[cap:]
                if d.found_by_sweep
            )
        )[: cap * SWEEP_CARRY_POLLS]
        for due_dream in self._due_dreams[:cap]:
            # region ai
            # enqueue_dream opens tracked_db, which fails closed flag-on without a
            # tenant, so bind this due-dream's tenant (its collection's tenant_id, from
            # list_due_dreams). The binding also lets RLS scope enqueue_dream's reads
            # and construct_work_unit_key namespace the queued dream to it, keeping the
            # deriver's batch tenant-homogeneous. No-op flag off.
            # endregion
            tenant_token = (
                tenant_context.set(due_dream.tenant_id)
                if settings.MULTI_TENANT
                else None
            )
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
            finally:
                if tenant_token is not None:
                    tenant_context.reset(tenant_token)


class DeriverMetricsPoller:
    """Refreshes the deriver gauges and the cached snapshot on a timer."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._snapshot: DeriverMetricsSnapshot = DeriverMetricsSnapshot()
        self._next_stale_cleanup: float | None = None
        self.dream_due_poller: DreamDuePoller = DreamDuePoller()

    @property
    def snapshot(self) -> DeriverMetricsSnapshot:
        return self._snapshot

    async def start(self) -> None:
        if self._task is not None:
            logger.warning("DeriverMetricsPoller already running")
            return
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._loop())
        await self.dream_due_poller.start()
        logger.info(
            "DeriverMetricsPoller started, interval %ss",
            settings.DERIVER.BACKLOG_METRICS_POLL_INTERVAL_SECONDS,
        )

    async def shutdown(self) -> None:
        await self.dream_due_poller.shutdown()
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
        # region ai
        # Cross-tenant: the metrics query reads across every tenant's queue, so it
        # runs on the RLS-bypass service session. Under MULTI_TENANT tracked_db would
        # fail closed here (this poller runs on API boot with no request tenant in
        # scope). read_only: no writes on this pass. dreams_due comes from
        # DreamDuePoller's last poll, which runs on its own task and timer.
        # endregion
        async with service_db("deriver_metrics", read_only=True) as db:
            stats = await crud.get_deriver_metrics(db)

        dreams_due = self.dream_due_poller.dreams_due
        signal = outstanding_work_seconds(stats, dreams_due=dreams_due)
        measured_at = time.time()

        self._snapshot = DeriverMetricsSnapshot(
            signal_seconds=signal,
            dreams_due=dreams_due,
            stats=stats,
            measured_at=measured_at,
        )

        metrics = prometheus_metrics
        metrics.set_deriver_metrics(
            eligible_work_units=stats.eligible_work_units,
            excluded_work_units=stats.excluded_work_units,
            excluded_work_units_by_tenant=stats.excluded_work_units_by_tenant,
            claimed_work_units=stats.claimed_work_units,
            inflight_units_by_tenant=stats.inflight_units_by_tenant,
            pending_items=stats.pending_items,
            oldest_pending_age_seconds=stats.oldest_pending_age_seconds,
            embeddings_pending=stats.embeddings_pending,
            embeddings_pending_due=stats.embeddings_pending_due,
        )
        if settings.MULTI_TENANT:
            metrics.set_paused_tenants(count=stats.paused_tenants)
        metrics.set_deriver_outstanding_work(seconds=signal)
        metrics.set_deriver_metrics_last_success(timestamp=measured_at)

    async def _maybe_cleanup_stale_work_units(self) -> None:
        interval = settings.DERIVER.STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS
        if (
            self._next_stale_cleanup is not None
            and time.monotonic() < self._next_stale_cleanup
        ):
            return
        self._next_stale_cleanup = time.monotonic() + interval
        # region ai
        # Cross-tenant: stale-session cleanup spans all tenants (service session, not
        # tracked_db which would fail closed under MULTI_TENANT). Mirrors the deriver's
        # own cleanup_stale_work_units call in queue_manager.
        # endregion
        async with service_db("cleanup_stale_work_units") as db:
            await crud.cleanup_stale_work_units(db)
