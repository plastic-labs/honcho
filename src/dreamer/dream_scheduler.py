import asyncio
from datetime import datetime, timezone
from logging import getLogger
from typing import Any

import sentry_sdk
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.utils.dynamic_tables import create_dynamic_document_model
from src.utils.queue_payload import create_dream_payload
from src.utils.work_unit import get_work_unit_key, parse_work_unit_key

logger = getLogger(__name__)


_dream_scheduler: "DreamScheduler | None" = None


def set_dream_scheduler(dream_scheduler: "DreamScheduler") -> None:
    """Set the global dream scheduler reference."""
    global _dream_scheduler
    _dream_scheduler = dream_scheduler


def get_dream_scheduler() -> "DreamScheduler | None":
    """Get the global dream scheduler reference."""
    return _dream_scheduler


def get_affected_dream_keys(message: dict[str, Any]) -> list[str]:
    """
    Get all work unit keys for dreams that might be affected by this message.

    Args:
        message: The message payload

    Returns:
        List of work unit keys that should have their dreams cancelled
    """
    workspace_name = message.get("workspace_name")
    sender_name = message.get("peer_name")

    if not workspace_name or not sender_name:
        return []

    # Generate dream work unit key for this peer's collection
    dream_key = get_work_unit_key(
        {
            "task_type": "dream",
            "workspace_name": workspace_name,
            "sender_name": sender_name,
            "target_name": sender_name,  # Dream work units use collection name as target
        }
    )

    return [dream_key]


class DreamScheduler:
    def __init__(self):
        self.pending_dreams: dict[str, asyncio.Task[None]] = {}

    def schedule_dream(
        self,
        work_unit_key: str,
        workspace_name: str,
        peer_name: str,
        collection_name: str,
        document_count: int,
        delay_minutes: int,
    ) -> None:
        """Schedule a dream for a collection after a delay."""
        if not settings.DREAM.ENABLED:
            return

        # Cancel any existing dream for this collection
        self.cancel_dream(work_unit_key)

        task = asyncio.create_task(
            self._delayed_dream(
                work_unit_key,
                workspace_name,
                peer_name,
                collection_name,
                document_count,
                delay_minutes,
            )
        )
        self.pending_dreams[work_unit_key] = task
        task.add_done_callback(lambda t: self.pending_dreams.pop(work_unit_key, None))

    def cancel_dream(self, work_unit_key: str) -> bool:
        """Cancel a pending dream. Returns True if a dream was cancelled."""
        if work_unit_key in self.pending_dreams:
            task = self.pending_dreams.pop(work_unit_key)
            task.cancel()
            logger.debug(f"Cancelled pending dream for {work_unit_key}")
            return True
        return False

    async def _delayed_dream(
        self,
        work_unit_key: str,
        workspace_name: str,
        peer_name: str,
        collection_name: str,
        document_count: int,
        delay_minutes: int,
    ) -> None:
        try:
            await asyncio.sleep(delay_minutes * 60)

            # Check if collection is still inactive before executing dream
            if await self._should_execute_dream(
                workspace_name, peer_name, collection_name
            ):
                await self._execute_dream(
                    work_unit_key,
                    workspace_name,
                    peer_name,
                    collection_name,
                    document_count,
                )
                logger.info(f"Executed dream for {work_unit_key}")
            else:
                logger.info(
                    f"Skipping dream for {work_unit_key} - collection is active"
                )

        except asyncio.CancelledError:
            logger.info(f"Dream task cancelled for {work_unit_key}")
        except Exception as e:
            logger.error(f"Error in delayed dream for {work_unit_key}: {str(e)}")
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)

    async def _should_execute_dream(
        self, workspace_name: str, peer_name: str, collection_name: str
    ) -> bool:
        """Check if the collection is inactive and should be dreamed upon."""
        async with tracked_db("dream_activity_check") as db:
            # Check for active queue sessions related to this collection
            query = select(models.ActiveQueueSession)
            result = await db.execute(query)
            active_sessions = result.scalars().all()

            # Look for any active work units that match this collection
            for active_session in active_sessions:
                parsed_key = parse_work_unit_key(active_session.work_unit_key)
                if (
                    parsed_key["workspace_name"] == workspace_name
                    and parsed_key["sender_name"] == peer_name
                    and parsed_key["target_name"]
                    == peer_name  # Global representation tasks
                ):
                    logger.debug(
                        f"Collection {collection_name} is active, skipping dream"
                    )
                    return False

        return True

    async def _execute_dream(
        self,
        work_unit_key: str,
        workspace_name: str,
        peer_name: str,
        collection_name: str,
        document_count: int,
    ) -> None:
        """Execute the dream by enqueueing it and updating collection metadata."""
        dream_payload = create_dream_payload(
            workspace_name=workspace_name,
            sender_name=peer_name,
            collection_name=collection_name,
            dream_type="consolidate",
        )

        async with tracked_db("dream_execute") as db:
            dream_record = {
                "work_unit_key": work_unit_key,
                "payload": dream_payload,
                "session_id": None,
                "task_type": "dream",
            }

            await db.execute(insert(models.QueueItem), [dream_record])

            now_iso = datetime.now(timezone.utc).isoformat()
            stmt = (
                update(models.Collection)
                .where(
                    models.Collection.workspace_name == workspace_name,
                    models.Collection.peer_name == peer_name,
                    models.Collection.name == collection_name,
                )
                .values(
                    internal_metadata=models.Collection.internal_metadata.op("||")(
                        {
                            "dream": {
                                "last_dream_document_count": document_count,
                                "last_dream_at": now_iso,
                            }
                        }
                    )
                )
            )
            await db.execute(stmt)
            await db.commit()

        logger.info(
            f"Enqueued dream task for {workspace_name}/{peer_name}/{collection_name}"
        )

    async def shutdown(self) -> None:
        """Cancel all pending dreams during shutdown."""
        if self.pending_dreams:
            logger.info(f"Cancelling {len(self.pending_dreams)} pending dreams...")
            for task in self.pending_dreams.values():
                task.cancel()
            await asyncio.gather(*self.pending_dreams.values(), return_exceptions=True)
            self.pending_dreams.clear()


async def check_and_schedule_dream(
    db: AsyncSession,
    collection: models.Collection,
) -> bool:
    """
    Check if a collection has reached the document threshold and schedule a timer-based dream.

    This function only schedules a timer-based dream if:
    1. Dreams are enabled
    2. Document threshold is reached
    3. Minimum hours between dreams have passed
    4. No dream is already scheduled for this collection

    Args:
        db: Database session
        collection: Collection model to check

    Returns:
        True if a dream timer was scheduled, False otherwise
    """
    if not settings.DREAM.ENABLED:
        return False

    # Get dream metadata from internal_metadata
    dream_metadata = collection.internal_metadata.get("dream", {})
    last_dream_document_count = dream_metadata.get("last_dream_document_count", 0)
    last_dream_at = dream_metadata.get("last_dream_at")

    # Count current documents in the collection
    DocumentModel = create_dynamic_document_model(collection.id)
    count_stmt = select(DocumentModel).where(
        DocumentModel.workspace_name == collection.workspace_name,
        DocumentModel.peer_name == collection.peer_name,
        DocumentModel.collection_name == collection.name,
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

    # Only schedule timer if document threshold is reached
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

        dream_scheduler = get_dream_scheduler()
        if dream_scheduler:
            collection_work_unit_key = get_work_unit_key(
                {
                    "task_type": "dream",
                    "workspace_name": collection.workspace_name,
                    "sender_name": collection.peer_name,
                    "target_name": collection.name,
                }
            )

            dream_scheduler.schedule_dream(
                collection_work_unit_key,
                collection.workspace_name,
                collection.peer_name,
                collection.name,
                current_document_count,
                settings.DREAM.IDLE_TIMEOUT_MINUTES,
            )
            logger.info(
                f"Scheduled dream for {collection.workspace_name}/{collection.peer_name}/{collection.name} "
                + f"(threshold reached: {documents_since_last_dream}/{settings.DREAM.DOCUMENT_THRESHOLD} documents)"
            )
            return True

    return False
