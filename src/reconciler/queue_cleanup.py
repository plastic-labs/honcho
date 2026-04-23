"""
Queue cleanup job.

This module provides a periodic cleanup job that removes old processed queue items.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from sqlalchemy import CursorResult, delete

from src import models
from src.config import settings
from src.dependencies import tracked_db

logger = logging.getLogger(__name__)


async def cleanup_queue_items() -> int:
    """
    Delete processed queue items.

    Successfully processed queue items are deleted immediately,
    while errored queue items are deleted after retention window.

    Returns:
        The number of queue items deleted.
    """
    async with tracked_db("cleanup_queue_items") as db:
        now = datetime.now(timezone.utc)
        error_cutoff = now - timedelta(
            seconds=settings.DERIVER.QUEUE_ERROR_RETENTION_SECONDS
        )

        result = cast(
            CursorResult[Any],
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
            ),
        )
        await db.commit()
        deleted_count = result.rowcount
        logger.info("Queue cleanup completed, deleted %d items", deleted_count)
        return deleted_count
