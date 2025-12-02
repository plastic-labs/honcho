import asyncio
import contextlib
from datetime import datetime, timezone
from logging import getLogger
from typing import Any

import sentry_sdk
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.schemas import DreamType
from src.utils.work_unit import construct_work_unit_key, parse_work_unit_key

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
    peer_name = message.get("peer_name")

    if not workspace_name or not peer_name:
        return []

    # Generate dream work unit key for this peer's collection
    dream_key = construct_work_unit_key(
        workspace_name,
        {
            "task_type": "dream",
            "observer": peer_name,
            "observed": peer_name,
        },
    )

    return [dream_key]


class DreamScheduler:
    _instance: "DreamScheduler | None" = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not DreamScheduler._initialized:
            self.pending_dreams: dict[str, asyncio.Task[None]] = {}
            DreamScheduler._initialized = True

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance. Only use this in tests."""
        cls._instance = None
        cls._initialized = False

    async def schedule_dream(
        self,
        work_unit_key: str,
        workspace_name: str,
        document_count: int,
        delay_minutes: int,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
    ) -> None:
        """Schedule a dream for a collection after a delay."""
        if not settings.DREAM.ENABLED:
            return

        # Cancel any existing dream for this collection
        await self.cancel_dream(work_unit_key)

        task = asyncio.create_task(
            self._delayed_dream(
                work_unit_key,
                workspace_name,
                document_count,
                delay_minutes,
                dream_type,
                observer=observer,
                observed=observed,
            )
        )
        self.pending_dreams[work_unit_key] = task
        task.add_done_callback(lambda t: self.pending_dreams.pop(work_unit_key, None))

    async def cancel_dream(self, work_unit_key: str) -> bool:
        """Cancel a pending dream. Returns True if a dream was cancelled."""
        if work_unit_key in self.pending_dreams:
            task = self.pending_dreams.pop(work_unit_key)
            task.cancel()
            # Wait for the task to actually finish (including its done callback)
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return True
        return False

    async def _delayed_dream(
        self,
        work_unit_key: str,
        workspace_name: str,
        document_count: int,
        delay_minutes: int,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
    ) -> None:
        try:
            await asyncio.sleep(delay_minutes * 60)

            # Check if collection is still inactive before executing dream
            if await self._should_execute_dream(
                workspace_name, observer=observer, observed=observed
            ):
                await self.execute_dream(
                    workspace_name,
                    document_count,
                    dream_type,
                    observer=observer,
                    observed=observed,
                )
                logger.info(f"Executed dream for {work_unit_key}")
            else:
                logger.info(
                    f"Skipping dream for {work_unit_key} - collection is active"
                )

        except asyncio.CancelledError:
            logger.debug(f"Dream task cancelled for {work_unit_key}")
        except Exception as e:
            logger.error(f"Error in delayed dream for {work_unit_key}: {str(e)}")
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)

    async def _should_execute_dream(
        self, workspace_name: str, *, observer: str, observed: str
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
                    parsed_key.workspace_name == workspace_name
                    and parsed_key.observer == observer
                    and parsed_key.observed == observed
                ):
                    logger.debug("Collection is active, skipping dream")
                    return False

        return True

    async def execute_dream(
        self,
        workspace_name: str,
        document_count: int,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
    ) -> None:
        """Execute the dream by enqueueing it and updating collection metadata."""
        # Import here to avoid circular dependency
        from src.deriver.enqueue import enqueue_dream

        # Find the most recent session for this observer/observed pair
        async with tracked_db("dream_session_lookup") as db:
            stmt = (
                select(models.Document.session_name)
                .where(
                    models.Document.workspace_name == workspace_name,
                    models.Document.observer == observer,
                    models.Document.observed == observed,
                )
                .order_by(models.Document.created_at.desc())
                .limit(1)
            )
            session_name = await db.scalar(stmt)

        if not session_name:
            logger.warning(
                f"No documents found for {workspace_name}/{observer}/{observed}, skipping dream"
            )
            return

        await enqueue_dream(
            workspace_name,
            observer=observer,
            observed=observed,
            dream_type=dream_type,
            document_count=document_count,
            session_name=session_name,
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
    count_stmt = select(func.count(models.Document.id)).where(
        models.Document.workspace_name == collection.workspace_name,
        models.Document.observer == collection.observer,
        models.Document.observed == collection.observed,
    )
    current_document_count = int(await db.scalar(count_stmt) or 0)

    # Calculate documents added since last dream
    documents_since_last_dream = current_document_count - last_dream_document_count

    logger.debug(
        "Dream check",
        extra={
            "workspace_name": collection.workspace_name,
            "observer": collection.observer,
            "observed": collection.observed,
            "current_document_count": current_document_count,
            "last_dream_document_count": last_dream_document_count,
            "documents_since_last_dream": documents_since_last_dream,
            "document_threshold": settings.DREAM.DOCUMENT_THRESHOLD,
        },
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
                        f"Skipping dream for {collection.observer}/{collection.observed}: only {hours_since_last_dream:.1f} hours "
                        + f"since last dream (minimum: {settings.DREAM.MIN_HOURS_BETWEEN_DREAMS})"
                    )
                    return False
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid last_dream_at timestamp: {last_dream_at}, error: {e}"
                )

        dream_scheduler = get_dream_scheduler()
        if dream_scheduler:
            collection_work_unit_key = construct_work_unit_key(
                collection.workspace_name,
                {
                    "task_type": "dream",
                    "observer": collection.observer,
                    "observed": collection.observed,
                },
            )

            enabled_dream_types = settings.DREAM.ENABLED_TYPES
            for dream_type in enabled_dream_types:
                await dream_scheduler.schedule_dream(
                    collection_work_unit_key,
                    collection.workspace_name,
                    current_document_count,
                    settings.DREAM.IDLE_TIMEOUT_MINUTES,
                    dream_type=DreamType(dream_type),
                    observer=collection.observer,
                    observed=collection.observed,
                )
                logger.debug(
                    "Scheduled dream",
                    extra={
                        "workspace_name": collection.workspace_name,
                        "observer": collection.observer,
                        "observed": collection.observed,
                        "documents_since_last_dream": documents_since_last_dream,
                        "document_threshold": settings.DREAM.DOCUMENT_THRESHOLD,
                        "dream_type": dream_type,
                    },
                )
            return True

    return False
