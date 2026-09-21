"""Which tenants' work the deriver must not claim, mirrored into every claiming process.

The control plane records a pause as ``tenants.derivation_paused`` through the
registry's ``PATCH``. This module turns that column into the id set the claim's
exclusion seam (``crud.deriver.claim_excluded_tenant_ids``) hands the claim
query. The claim builder is sync and has no session in scope, so the set is
held in-process and refreshed on a timer instead of read on every poll.

Failure policy, deliberately asymmetric:

- The FIRST load (``DerivationPauseRefresher.start``) fails closed: a process
  that cannot read the paused set refuses to start claiming. A permanent
  misconfiguration — the column missing, the service role wrong — surfaces
  at boot instead of silently un-pausing every paused tenant.
- Every later refresh fails OPEN on the last known good set. Emptying the set
  on a transient database error would un-pause everybody at once; over-deriving
  for a paused tenant for one interval is the lesser harm. The failure is
  counted and logged, never swallowed.

Flag off (``MULTI_TENANT`` false) nothing here runs: the seam returns None and
the claim SQL is byte-identical to a single-tenant deployment's.
"""

import asyncio
import contextlib
import logging
from collections.abc import Sequence

import sentry_sdk
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dependencies import service_db
from src.telemetry import prometheus_metrics

logger = logging.getLogger(__name__)

# None until the first successful load; a sorted tuple afterwards, so the
# claim's bound parameters are stable across polls.
_paused: tuple[str, ...] | None = None


def excluded_tenant_ids() -> Sequence[str] | None:
    """The paused set in the seam's shape: ids, or None when nothing is excluded."""
    return _paused or None


def paused_tenant_count() -> int:
    return len(_paused or ())


def reset() -> None:
    """Forget the loaded set (tests, or a process that must start cold)."""
    global _paused
    _paused = None


async def load_paused_tenant_ids(db: AsyncSession) -> tuple[str, ...]:
    """Read the paused set from the registry table on the given session."""
    result = await db.execute(
        select(models.Tenant.tenant_id)
        .where(models.Tenant.derivation_paused.is_(True))
        .order_by(models.Tenant.tenant_id)
    )
    return tuple(result.scalars().all())


async def refresh(db: AsyncSession) -> tuple[str, ...]:
    """Reload the set on ``db`` and publish it. Raises on failure; callers choose the policy.

    Also the API process's refresh path: the backlog-metrics poll passes its own
    read-only service session here before it splits eligible from excluded work,
    so the gauges KEDA scales on and the deriver's claim read the same bit.
    """
    global _paused
    paused = await load_paused_tenant_ids(db)
    if paused != _paused:
        logger.info(
            "Derivation paused for %d tenant(s) (was %s)",
            len(paused),
            "unloaded" if _paused is None else len(_paused),
        )
    _paused = paused
    prometheus_metrics.set_paused_tenants(count=len(paused))
    return paused


async def refresh_from_service_db() -> tuple[str, ...]:
    # region ai
    # Cross-tenant by nature (one query returns every paused tenant), so it runs
    # on the RLS-bypass service session; tracked_db would fail closed with no
    # tenant bound, which is every timer tick.
    # endregion
    async with service_db("derivation_pause.refresh", read_only=True) as db:
        return await refresh(db)


class DerivationPauseRefresher:
    """Keeps the in-process paused set fresh on a timer, in the process that claims."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """Load once (raising on failure), then refresh on the configured interval."""
        if not settings.MULTI_TENANT:
            return
        if self._task is not None:
            logger.warning("DerivationPauseRefresher already running")
            return
        # ai: fails closed on purpose — see the module docstring for the split.
        await refresh_from_service_db()
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "DerivationPauseRefresher started, interval %ss",
            settings.DERIVER.PAUSED_TENANTS_REFRESH_SECONDS,
        )

    async def shutdown(self) -> None:
        if self._task is None:
            return
        self._shutdown_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except TimeoutError:
            logger.warning("DerivationPauseRefresher shutdown timed out, cancelling")
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def _loop(self) -> None:
        interval = settings.DERIVER.PAUSED_TENANTS_REFRESH_SECONDS
        while not self._shutdown_event.is_set():
            # start() has just loaded, so sleep first.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
            if self._shutdown_event.is_set():
                break
            await self.refresh_once()

    async def refresh_once(self) -> None:
        """One timer tick: reload, or keep the last known good set on failure."""
        try:
            await refresh_from_service_db()
        except Exception as e:
            # region ai
            # Fail OPEN on the last known good set — the one place this
            # project's fail-closed reflex is wrong. An empty set here
            # would resume derivation for every paused tenant at once; a
            # stale set over-derives for the paused ones for one interval.
            # Counted so "paused tenant still deriving" has a signal.
            # endregion
            prometheus_metrics.record_derivation_pause_refresh_failure()
            logger.error(
                "Paused-tenant refresh failed; keeping the last known set of %d: %s",
                paused_tenant_count(),
                e,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)


derivation_pause_refresher = DerivationPauseRefresher()
