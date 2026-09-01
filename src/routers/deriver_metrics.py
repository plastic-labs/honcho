"""Deriver work metrics as JSON. Errors rather than serving a stale number."""

from logging import getLogger

from fastapi import APIRouter, HTTPException

from src.backlog import DeriverMetricsPoller
from src.config import settings

logger = getLogger(__name__)

router = APIRouter(prefix="/deriver", tags=["deriver"])

MAX_SNAPSHOT_AGE_INTERVALS = 3

_poller: DeriverMetricsPoller | None = None


def set_deriver_metrics_poller(poller: DeriverMetricsPoller | None) -> None:
    global _poller
    _poller = poller


def max_snapshot_age_seconds() -> float:
    return float(
        settings.DERIVER.BACKLOG_METRICS_POLL_INTERVAL_SECONDS
        * MAX_SNAPSHOT_AGE_INTERVALS
    )


@router.get("/metrics")
async def get_deriver_metrics_response() -> dict[str, float | int]:
    """Seconds of outstanding deriver work, plus the raw counts behind it."""
    snapshot = _poller.snapshot if _poller is not None else None
    if snapshot is None or snapshot.measured_at is None:
        raise HTTPException(
            status_code=503, detail="No deriver measurement available yet"
        )

    age = snapshot.age_seconds
    if age is None or age > max_snapshot_age_seconds():
        raise HTTPException(
            status_code=503,
            detail=f"Deriver measurement is stale ({age:.0f}s old)"
            if age is not None
            else "Deriver measurement is stale",
        )

    return {
        "outstanding_work_seconds": snapshot.signal_seconds,
        "eligible_work_units": snapshot.stats.eligible_work_units,
        "claimed_work_units": snapshot.stats.claimed_work_units,
        "pending_items": snapshot.stats.pending_items,
        "oldest_pending_age_seconds": snapshot.stats.oldest_pending_age_seconds,
        "embeddings_pending": snapshot.stats.embeddings_pending,
        "embeddings_pending_due": snapshot.stats.embeddings_pending_due,
        "dreams_due": snapshot.dreams_due,
        "measured_at": snapshot.measured_at,
        "measurement_age_seconds": age,
    }
