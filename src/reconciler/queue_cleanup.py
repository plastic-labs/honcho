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


async def cleanup_queue_items() -> int:
    """
    Delete processed queue items.

    Successfully processed queue items are deleted immediately,
    while errored queue items are deleted after retention window.

    Returns:
        The number of queue items deleted.
    """
    async with service_db("cleanup_queue_items") as db:
        now = datetime.now(UTC)
        error_cutoff = now - timedelta(
            seconds=settings.DERIVER.QUEUE_ERROR_RETENTION_SECONDS
        )

        # Skip rows another transaction holds (a session or workspace deletion
        # deleting the same processed rows) instead of waiting on them: this
        # gets them on a later pass, and a statement that never waits can't be
        # a deadlock partner.
        result = cast(
            CursorResult[Any],
            await db.execute(
                delete(models.QueueItem).where(
                    models.QueueItem.id.in_(
                        select(models.QueueItem.id)
                        .where(
                            models.QueueItem.processed
                            & (
                                models.QueueItem.error.is_(None)
                                | (
                                    models.QueueItem.error.is_not(None)
                                    & (models.QueueItem.created_at < error_cutoff)
                                )
                            )
                        )
                        .with_for_update(skip_locked=True)
                    )
                )
            ),
        )
        await db.commit()
        deleted_count = result.rowcount
        logger.info("Queue cleanup completed, deleted %d items", deleted_count)
        return deleted_count
