"""Deriver work metrics as JSON, with the age of the measurement alongside them."""

from logging import getLogger

from fastapi import APIRouter, HTTPException

from src.backlog import DeriverMetricsPoller

logger = getLogger(__name__)

router = APIRouter(prefix="/deriver", tags=["deriver"])

_poller: DeriverMetricsPoller | None = None


def set_deriver_metrics_poller(poller: DeriverMetricsPoller | None) -> None:
    global _poller
    _poller = poller


@router.get("/metrics")
async def get_deriver_metrics_response() -> dict[str, float | int]:
    """Seconds of outstanding deriver work, plus the raw counts behind it."""
    snapshot = _poller.snapshot if _poller is not None else None
    if snapshot is None or snapshot.measured_at is None:
        raise HTTPException(
            status_code=503, detail="No deriver measurement available yet"
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
        "measurement_age_seconds": snapshot.age_seconds or 0.0,
    }
