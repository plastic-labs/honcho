"""
Queue cleanup job.

This module provides a periodic cleanup job that removes old processed queue items.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select

from src import models
from src.config import settings
from src.dependencies import service_db

logger = logging.getLogger(__name__)

QUEUE_CLEANUP_BATCH_SIZE = 5000


async def cleanup_queue_items() -> int:
    """
    Delete processed queue items.

    Successfully processed queue items are deleted immediately,
    while errored queue items are deleted after retention window. Rows go in
    id-ordered batches of QUEUE_CLEANUP_BATCH_SIZE, one transaction each, so no
    single statement holds locks or writes WAL for the whole backlog.

    Returns:
        The number of queue items deleted.
    """
    error_cutoff = datetime.now(UTC) - timedelta(
        seconds=settings.DERIVER.QUEUE_ERROR_RETENTION_SECONDS
    )
    deletable = models.QueueItem.processed & (
        models.QueueItem.error.is_(None)
        | (
            models.QueueItem.error.is_not(None)
            & (models.QueueItem.created_at < error_cutoff)
        )
    )
    batch_ids = (
        select(models.QueueItem.id)
        .where(deletable)
        .order_by(models.QueueItem.id)
        .limit(QUEUE_CLEANUP_BATCH_SIZE)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )

    deleted_count = 0
    async with service_db("cleanup_queue_items") as db:
        while True:
            result = cast(
                CursorResult[Any],
                await db.execute(
                    delete(models.QueueItem).where(models.QueueItem.id.in_(batch_ids))
                ),
            )
            await db.commit()
            deleted_count += result.rowcount
            if result.rowcount < QUEUE_CLEANUP_BATCH_SIZE:
                break
    logger.info("Queue cleanup completed, deleted %d items", deleted_count)
    return deleted_count
