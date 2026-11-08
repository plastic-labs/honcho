"""Polling of the deriver's outstanding work. Schedules only when it owns scheduling."""

import asyncio
import contextlib
import os
import socket
import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import sentry_sdk
from nanoid import generate as generate_nanoid

from src import crud, schemas
from src.cache.client import fleet_cache, fleet_key
from src.config import settings
from src.db import tenant_context
from src.dependencies import service_db
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


_LEASE = "deriver_metrics_lease"
_SNAPSHOT = "deriver_metrics_snapshot"


def _coordination_ttl() -> float:
    """How long a lease or a shared snapshot outlives its last refresh."""
    return 3 * settings.DERIVER.BACKLOG_METRICS_POLL_INTERVAL_SECONDS


class DeriverMetricsPoller:
    """Refreshes the deriver gauges and the cached snapshot on a timer.

    Under DERIVER_SCHEDULER=api every API replica runs one, but only the replica
    holding the fleet lease measures: it runs the cross-tenant counts, lists and
    enqueues due dreams, cleans up stale claims, and shares the result. The others
    serve that shared snapshot, so every replica reports the same numbers to the
    load-balanced scrape and to KEDA.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._snapshot: DeriverMetricsSnapshot = DeriverMetricsSnapshot()
        self._next_dream_poll: float | None = None
        self._due_dreams: list[DueDream] = []
        self._next_stale_cleanup: float | None = None
        self._embeddings_pending: tuple[int, int] = (0, 0)
        self._holder: str = (
            f"{socket.gethostname()}:{os.getpid()}:{generate_nanoid(size=8)}"
        )
        self._leading: bool = True

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
        if self._leading and settings.DERIVER.SCHEDULER == "api":
            # Hand over now rather than after the lease expires.
            with contextlib.suppress(Exception):
                await fleet_cache.unlock(fleet_key(_LEASE), self._holder)
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
            if settings.DERIVER.SCHEDULER == "api" and self._leading:
                try:
                    await self._maybe_cleanup_stale_work_units()
                except Exception as e:
                    logger.error("Stale work unit cleanup failed: %s", e)
                    if settings.SENTRY.ENABLED:
                        sentry_sdk.capture_exception(e)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)

    async def refresh(self) -> None:
        """One pass: measure, or serve the leaseholder's snapshot."""
        if settings.DERIVER.SCHEDULER == "api":
            self._leading = await self._hold_lease()
            if not self._leading and await self._follow():
                return
        await self._measure()
        if settings.DERIVER.SCHEDULER == "api" and self._leading:
            await self._share()

    async def _hold_lease(self) -> bool:
        """Take or renew the fleet lease; True when this replica should measure."""
        # region ai
        # Sticky: the holder renews on every pass, so its own dream-poll and
        # stale-cleanup clocks stay meaningful; another replica takes over only
        # after the lease expires (the holder died, or stopped renewing). The
        # renewal is a read then an expire, not atomic: if the lease lapses in
        # between and another replica takes it, both measure for a pass. That
        # is harmless (the counts are read-only, a duplicate dream enqueue hits
        # uq_queue_dream_pending_work_unit_key, stale cleanup is idempotent).
        # Coordination unavailable (no Redis, or an error) means measuring
        # locally, which is what every replica did before the lease.
        # endregion
        key, ttl = fleet_key(_LEASE), _coordination_ttl()
        try:
            if await fleet_cache.set_lock(key, self._holder, expire=ttl):
                return True
            # ai: set_lock stores the holder raw, and get() would unpickle it (None on Redis), so read it raw
            held = await fleet_cache.get_raw(key)
            if isinstance(held, bytes):
                held = held.decode()
            if held == self._holder:
                await fleet_cache.expire(key, ttl)
                return True
            return False
        except Exception:
            logger.debug("Fleet lease unavailable; measuring locally", exc_info=True)
            return True

    async def _follow(self) -> bool:
        """Serve the leaseholder's snapshot; False when there is none to serve."""
        try:
            shared: dict[str, Any] | None = await fleet_cache.get(fleet_key(_SNAPSHOT))
        except Exception:
            logger.debug("Fleet snapshot unavailable", exc_info=True)
            return False
        if not shared:
            return False
        self._publish(
            DeriverMetricsSnapshot(
                signal_seconds=shared["signal_seconds"],
                dreams_due=shared["dreams_due"],
                stats=schemas.DeriverMetrics(**shared["stats"]),
                measured_at=shared["measured_at"],
            )
        )
        return True

    async def _share(self) -> None:
        snapshot = self._snapshot
        try:
            await fleet_cache.set(
                fleet_key(_SNAPSHOT),
                {
                    "signal_seconds": snapshot.signal_seconds,
                    "dreams_due": snapshot.dreams_due,
                    "stats": snapshot.stats.model_dump(mode="json"),
                    "measured_at": snapshot.measured_at,
                },
                expire=_coordination_ttl(),
            )
        except Exception:
            logger.debug("Could not share the deriver metrics snapshot", exc_info=True)

    async def _measure(self) -> None:
        """Run the counts. The snapshot only advances on a complete pass."""
        dream_tick = self._dream_poll_due()
        # region ai
        # Cross-tenant: the metrics query and the due-dream scan both read across every
        # tenant's queue/documents, so they run on the RLS-bypass service session. Under
        # MULTI_TENANT tracked_db would fail closed here (this poller runs on API boot
        # with no request tenant in scope). read_only — no writes on this pass.
        # endregion
        async with service_db("deriver_metrics", read_only=True) as db:
            stats = await crud.get_deriver_metrics(db, count_embeddings=dream_tick)
            if dream_tick:
                self._due_dreams = await list_due_dreams(db)
                self._next_dream_poll = (
                    time.monotonic() + settings.DREAM.DUE_POLL_INTERVAL_SECONDS
                )

        if dream_tick:
            self._embeddings_pending = (
                stats.embeddings_pending,
                stats.embeddings_pending_due,
            )
        # We're not fetching these stats anew this turn; copy old ones
        else:
            stats = stats.model_copy(
                update={
                    "embeddings_pending": self._embeddings_pending[0],
                    "embeddings_pending_due": self._embeddings_pending[1],
                }
            )

        if dream_tick and settings.DERIVER.SCHEDULER == "api":
            await self._enqueue_due_dreams()

        self._publish(
            DeriverMetricsSnapshot(
                signal_seconds=outstanding_work_seconds(
                    stats, dreams_due=len(self._due_dreams)
                ),
                dreams_due=len(self._due_dreams),
                stats=stats,
                measured_at=time.time(),
            )
        )

    def _publish(self, snapshot: DeriverMetricsSnapshot) -> None:
        """Serve a snapshot and set the gauges from it."""
        self._snapshot = snapshot
        stats = snapshot.stats
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
        metrics.set_dreams_due(count=snapshot.dreams_due)
        metrics.set_deriver_outstanding_work(seconds=snapshot.signal_seconds)
        if snapshot.measured_at is not None:
            metrics.set_deriver_metrics_last_success(timestamp=snapshot.measured_at)

    def _dream_poll_due(self) -> bool:
        """The dream query is far more expensive, so it runs on its own spacing."""
        return (
            self._next_dream_poll is None or time.monotonic() >= self._next_dream_poll
        )

    async def _enqueue_due_dreams(self) -> None:
        cap = settings.DREAM.MAX_ENQUEUED_PER_POLL
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
