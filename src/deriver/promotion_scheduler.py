"""Promotion scheduler — periodically scans for un-promoted observations.

Runs as a background task in the deriver process (sibling to the reconciler
scheduler). Scans for observations that haven't been promoted yet and enqueues
promotion tasks.

This is cleaner than modifying the Deriver's save path — it doesn't risk
breaking the existing observation creation pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from src import models
from src.dependencies import tracked_db
from src.deriver.enqueue import enqueue as enqueue_deriver_task
from src.utils.queue_payload import PromotionPayload

logger = logging.getLogger(__name__)

# How often to scan for un-promoted observations
SCAN_INTERVAL_SECONDS = 60

# How old an observation must be before we consider it for promotion
# (gives the Deriver time to finish saving)
PROMOTION_DELAY_SECONDS = 10

# Maximum observations to promote per scan
MAX_PER_SCAN = 50


class PromotionScheduler:
    """Background scheduler that scans for un-promoted observations."""

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the promotion scheduler loop."""
        logger.info("Starting promotion scheduler (interval: %ds)", SCAN_INTERVAL_SECONDS)
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the promotion scheduler."""
        logger.info("Stopping promotion scheduler")
        self._shutdown_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """Main loop: scan for un-promoted observations and enqueue promotion tasks."""
        while not self._shutdown_event.is_set():
            try:
                await self._scan_and_enqueue()
            except Exception as e:
                logger.error("Promotion scan failed: %s", e)
            
            # Sleep with shutdown awareness
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=SCAN_INTERVAL_SECONDS,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Normal interval elapsed

    async def _scan_and_enqueue(self) -> None:
        """Scan for observations that haven't been promoted yet.
        
        An observation is considered "un-promoted" if it has no corresponding
        'promote' event in the access_log.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=PROMOTION_DELAY_SECONDS)
        
        async with tracked_db("promotion_scheduler.scan") as db:
            # Find observations created more than PROMOTION_DELAY_SECONDS ago
            # that don't have a 'promote' event in the access_log
            result = await db.execute(
                select(models.Document)
                .where(
                    models.Document.created_at < cutoff,
                    models.Document.deleted_at.is_(None),
                    ~select(models.AccessLogEntry.id)
                    .where(
                        models.AccessLogEntry.obs_id == models.Document.id,
                        models.AccessLogEntry.event_type == "promote",
                    )
                    .exists(),
                )
                .limit(MAX_PER_SCAN)
            )
            un_promoted = list(result.scalars().all())
        
        if not un_promoted:
            return
        
        logger.info(
            "Found %d un-promoted observations, enqueuing promotion tasks",
            len(un_promoted),
        )
        
        # Enqueue promotion tasks for each un-promoted observation
        for doc in un_promoted:
            try:
                payload = PromotionPayload(
                    collection_name=doc.collection_name if hasattr(doc, 'collection_name') else "",
                    obs_id=doc.id,
                    observer=doc.observer,
                    observed=doc.observed,
                    session_name=doc.session_name,
                )
                
                await enqueue_deriver_task([{
                    "workspace_name": doc.workspace_name,
                    "session_name": doc.session_name or "",
                    "peer_name": doc.observed,
                    "message_id": 0,  # Not tied to a specific message
                    "task_type": "promotion",
                    "collection_name": payload.collection_name,
                    "obs_id": payload.obs_id,
                    "observer": payload.observer,
                    "observed": payload.observed,
                }])
            except Exception as e:
                logger.error(
                    "Failed to enqueue promotion for observation %s: %s",
                    doc.id, e,
                )


# Singleton
_promotion_scheduler: PromotionScheduler | None = None


def get_promotion_scheduler() -> PromotionScheduler | None:
    return _promotion_scheduler


def set_promotion_scheduler(scheduler: PromotionScheduler) -> None:
    global _promotion_scheduler
    _promotion_scheduler = scheduler
