from datetime import datetime, timezone
from logging import getLogger

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.queue_payload import create_dream_payload
from src.deriver.utils import get_work_unit_key

logger = getLogger(__name__)


async def check_and_schedule_dream(
    db: AsyncSession,
    collection: models.Collection,
) -> bool:
    """
    Check if a collection has reached the document threshold for dreaming and schedule a dream if needed.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        collection_name: Name of the collection

    Returns:
        True if a dream was scheduled, False otherwise
    """
    if not settings.DREAM.ENABLED:
        return False

    # Get dream metadata from internal_metadata
    dream_metadata = collection.internal_metadata.get("dream", {})
    last_dream_document_count = dream_metadata.get("last_dream_document_count", 0)
    last_dream_at = dream_metadata.get("last_dream_at")

    # Count current documents in the collection
    count_stmt = select(models.Document).where(
        models.Document.workspace_name == collection.workspace_name,
        models.Document.peer_name == collection.peer_name,
        models.Document.collection_name == collection.name,
    )
    count_result = await db.execute(count_stmt)
    current_document_count = len(count_result.scalars().all())

    # Calculate documents added since last dream
    documents_since_last_dream = current_document_count - last_dream_document_count

    logger.info(
        f"Dream check for {collection.workspace_name}/{collection.peer_name}/{collection.name}: "
        + f"current={current_document_count}, last_dream_count={last_dream_document_count}, "
        + f"since_last={documents_since_last_dream}, threshold={settings.DREAM.DOCUMENT_THRESHOLD}"
    )

    # Check if we've reached the document threshold
    if documents_since_last_dream >= settings.DREAM.DOCUMENT_THRESHOLD:
        # Check if we're within minimum hours between dreams
        if last_dream_at:
            try:
                last_dream_time = datetime.fromisoformat(last_dream_at)
                hours_since_last_dream = (
                    datetime.now(timezone.utc) - last_dream_time
                ).total_seconds() / 3600

                if hours_since_last_dream < settings.DREAM.MIN_HOURS_BETWEEN_DREAMS:
                    logger.info(
                        f"Skipping dream for {collection.name}: only {hours_since_last_dream:.1f} hours "
                        + f"since last dream (minimum: {settings.DREAM.MIN_HOURS_BETWEEN_DREAMS})"
                    )
                    return False
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid last_dream_at timestamp: {last_dream_at}, error: {e}"
                )

        # Schedule the dream
        await _schedule_dream(db, collection, current_document_count)
        return True

    return False


async def _schedule_dream(
    db: AsyncSession,
    collection: models.Collection,
    current_document_count: int,
) -> None:
    """Schedule a dream task for the collection."""
    # Create dream payload
    dream_payload = create_dream_payload(
        workspace_name=collection.workspace_name,
        sender_name=collection.peer_name,
        target_name=collection.name,
        dream_type="consolidate",
    )

    # Create work unit key for the dream
    work_unit_key = get_work_unit_key(
        {
            "task_type": "dream",
            "workspace_name": collection.workspace_name,
            "sender_name": collection.peer_name,
            "target_name": collection.name,
        }
    )

    # Enqueue the dream directly to the queue
    dream_record = {
        "work_unit_key": work_unit_key,
        "payload": dream_payload,
        "session_id": None,
        "task_type": "dream",
    }

    await db.execute(insert(models.QueueItem), [dream_record])

    # Update collection's dream metadata
    now_iso = datetime.now(timezone.utc).isoformat()
    await _update_collection_dream_metadata(
        db, collection, current_document_count, now_iso
    )

    # Commit the transaction to persist the dream queue item
    await db.commit()

    logger.info(
        f"Scheduled dream for collection {collection.workspace_name}/{collection.peer_name}/{collection.name} "
        + f"with {current_document_count} documents"
    )


async def _update_collection_dream_metadata(
    db: AsyncSession,
    collection: models.Collection,
    document_count: int,
    dream_scheduled_at: str,
) -> None:
    """Update collection's dream metadata."""
    stmt = (
        update(models.Collection)
        .where(
            models.Collection.workspace_name == collection.workspace_name,
            models.Collection.peer_name == collection.peer_name,
            models.Collection.name == collection.name,
        )
        .values(
            internal_metadata=models.Collection.internal_metadata.op("||")(
                {
                    "dream": {
                        "last_dream_document_count": document_count,
                        "last_dream_at": dream_scheduled_at,
                    }
                }
            )
        )
    )
    await db.execute(stmt)
