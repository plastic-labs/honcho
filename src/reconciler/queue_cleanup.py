"""
Queue cleanup job.

This module provides a periodic cleanup job that removes old processed queue items.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from src import models
from src.config import settings
from src.dependencies import tracked_db

logger = logging.getLogger(__name__)


async def cleanup_queue_items() -> None:
    """
    Delete processed queue items.

    Successfully processed queue items are deleted immediately,
    while errored queue items are deleted after retention window.
    """
    async with tracked_db("cleanup_queue_items") as db:
        now = datetime.now(timezone.utc)
        error_cutoff = now - timedelta(
            seconds=settings.DERIVER.QUEUE_ERROR_RETENTION_SECONDS
        )

        await db.execute(
            delete(models.QueueItem).where(
                models.QueueItem.processed
                & (
                    models.QueueItem.error.is_(None)
                    | (
                        models.QueueItem.error.is_not(None)
                        & (models.QueueItem.created_at < error_cutoff)
                    )
                )
            )
        )
        await db.commit()
        logger.info("Queue cleanup completed")
