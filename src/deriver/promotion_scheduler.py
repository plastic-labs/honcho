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

from sqlalchemy import insert, select

from src import models
from src.dependencies import tracked_db
from src.utils.queue_payload import PromotionPayload

logger = logging.getLogger(__name__)

# Graph promotion *processing* is incomplete on the current schema (see the note
# in _scan_and_enqueue). Keep the scheduler crash-free and observable, but do not
# enqueue promotion tasks until the feature is finished. Flip to True once
# process_promotion()/_get_related_documents()/create_edge() work without a
# Document.collection_name column.
_PROMOTION_PROCESSING_READY = False

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
            # Extract plain data while still inside the session: the ORM
            # instances detach (and their attributes expire) once this
            # `async with` block exits, so reading doc.* afterwards raises
            # DetachedInstanceError.
            un_promoted = [
                {
                    "id": d.id,
                    "workspace_name": d.workspace_name,
                    "session_name": d.session_name,
                    "observer": d.observer,
                    "observed": d.observed,
                    "collection_name": getattr(d, "collection_name", None) or "",
                }
                for d in result.scalars().all()
            ]
        
        if not un_promoted:
            return
        
        logger.info(
            "%d observations await graph promotion",
            len(un_promoted),
        )

        # Graph promotion (L1->L2) is wired but NOT functional on this schema:
        # process_promotion() -> _get_related_documents()/create_edge() filter and
        # insert on Document.collection_name, which does not exist on the documents
        # table/model (collections are keyed by observer/observed). Enqueuing would
        # only crash the consumer and grow an un-deduped backlog, so stop here until
        # the promotion feature is completed.
        if not _PROMOTION_PROCESSING_READY:
            return
        
        # Build a `promotion` queue item per observation and insert them
        # directly. We do NOT route through enqueue() — that is the
        # message->representation path and rejects payloads that lack message
        # `content`, which is why promotion never produced any queue items.
        queue_records = []
        for doc in un_promoted:
            payload = PromotionPayload(
                collection_name=doc["collection_name"],
                obs_id=doc["id"],
                observer=doc["observer"],
                observed=doc["observed"],
                session_name=doc["session_name"],
            )
            queue_records.append({
                "work_unit_key": (
                    f"promotion:{doc['workspace_name']}:{doc['observed']}:{doc['id']}"
                ),
                "payload": payload.model_dump(),
                "session_id": None,
                "task_type": "promotion",
                "workspace_name": doc["workspace_name"],
                "message_id": None,  # not tied to a specific message
            })

        try:
            async with tracked_db("promotion_scheduler.enqueue") as db:
                await db.execute(insert(models.QueueItem), queue_records)
                await db.commit()
        except Exception as e:
            logger.error(
                "Failed to enqueue %d promotion tasks: %s", len(queue_records), e
            )


# Singleton
_promotion_scheduler: PromotionScheduler | None = None


def get_promotion_scheduler() -> PromotionScheduler | None:
    return _promotion_scheduler


def set_promotion_scheduler(scheduler: PromotionScheduler) -> None:
    global _promotion_scheduler
    _promotion_scheduler = scheduler
