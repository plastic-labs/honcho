import asyncio
import signal
from asyncio import Task
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any

import httpx
import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src.config import settings
from src.deriver.queue_work_unit import DeriverWorkUnit, WebhookWorkUnit, WorkUnit

from .. import models
from ..dependencies import tracked_db
from .consumer import process_item

logger = getLogger(__name__)

load_dotenv(override=True)


def create_work_unit(
    task_type: str,
    session_id: str | None,
    sender_name: str | None,
    target_name: str | None,
) -> WorkUnit:
    """Factory function to create the appropriate WorkUnit subclass based on task_type."""
    if task_type == "webhook":
        return WebhookWorkUnit(task_type=task_type)
    elif task_type in ("summary", "representation"):
        if session_id is None:
            raise ValueError(f"session_id is required for {task_type} tasks")
        return DeriverWorkUnit(
            task_type=task_type,
            session_id=session_id,
            sender_name=sender_name,
            target_name=target_name,
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def work_unit_from_dict(data: dict[str, Any]) -> WorkUnit:
    """Factory to deserialize WorkUnit from stored JSON data."""
    work_unit_type = data.get("type")
    if work_unit_type == "deriver":
        return DeriverWorkUnit.from_dict(data)
    elif work_unit_type == "webhook":
        return WebhookWorkUnit.from_dict(data)
    else:
        raise ValueError(f"Unknown work unit type: {work_unit_type}")


class QueueManager:
    def __init__(self):
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.active_tasks: set[asyncio.Task[None]] = set()
        self.owned_work_units: set[WorkUnit] = set()
        self.queue_empty_flag: asyncio.Event = asyncio.Event()
        self.client: httpx.AsyncClient = httpx.AsyncClient(
            headers={"User-Agent": "Honcho-Worker/1.0"}, timeout=30.0
        )

        # Initialize from settings
        self.workers: int = settings.DERIVER.WORKERS
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(self.workers)

        # Initialize Sentry if enabled, using settings
        if settings.SENTRY.ENABLED:
            sentry_sdk.init(
                dsn=settings.SENTRY.DSN,
                enable_tracing=True,
                release=settings.SENTRY.RELEASE,
                environment=settings.SENTRY.ENVIRONMENT,
                traces_sample_rate=settings.SENTRY.TRACES_SAMPLE_RATE,
                profiles_sample_rate=settings.SENTRY.PROFILES_SAMPLE_RATE,
                integrations=[AsyncioIntegration()],
            )

    def add_task(self, task: asyncio.Task[None]):
        """Track a new task"""
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    def track_work_unit(self, work_unit: WorkUnit):
        """Track a new work unit owned by this process"""
        self.owned_work_units.add(work_unit)

    def untrack_work_unit(self, work_unit: WorkUnit):
        """Remove a work unit from tracking"""
        self.owned_work_units.discard(work_unit)

    async def initialize(self):
        """Setup signal handlers, initialize client, and start the main polling loop"""
        logger.debug(f"Initializing QueueManager with {self.workers} workers")

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        signals = (signal.SIGTERM, signal.SIGINT)
        for sig in signals:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(self.shutdown(s))
            )
        logger.debug("Signal handlers registered")

        # Run the polling loop directly in this task
        logger.debug("Starting polling loop directly")
        try:
            await self.polling_loop()
        finally:
            await self.cleanup()

    async def shutdown(self, sig: signal.Signals):
        """Handle graceful shutdown"""
        logger.info(f"Received exit signal {sig.name}...")
        self.shutdown_event.set()

        if self.active_tasks:
            logger.info(
                f"Waiting for {len(self.active_tasks)} active tasks to complete..."
            )
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        if self.client:
            await self.client.aclose()

    async def cleanup(self):
        """Clean up owned work units"""
        if self.owned_work_units:
            logger.info(f"Cleaning up {len(self.owned_work_units)} owned work units...")
            try:
                # Use the tracked_db dependency for transaction safety
                async with tracked_db("queue_cleanup") as db:
                    for work_unit in self.owned_work_units:
                        await db.execute(
                            delete(models.ActiveQueueSession).where(
                                models.ActiveQueueSession.work_unit_key
                                == work_unit.get_unique_key()
                            )
                        )
                    await db.commit()
                    logger.info("Cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)

    ##########################
    # Polling and Scheduling #
    ##########################

    async def get_available_work_units(self, db: AsyncSession) -> Sequence[WorkUnit]:
        """
        Get available work units that aren't being processed.
        Returns a list of WorkUnit objects.
        """
        # Clean up stale work units
        five_minutes_ago = datetime.now(UTC) - timedelta(
            minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES
        )
        await db.execute(
            delete(models.ActiveQueueSession).where(
                models.ActiveQueueSession.last_updated < five_minutes_ago
            )
        )

        # Handle two types of work units:
        # 1. Deriver tasks (summary, representation): Grouped by session, sender, target, task
        # 2. Webhook tasks: Grouped only by task_type ('webhook'), as session_id is NULL

        # JSON path expressions
        sender_name_expr = models.QueueItem.payload.get("sender_name").astext
        target_name_expr = models.QueueItem.payload.get("target_name").astext
        task_type_expr = models.QueueItem.payload["task_type"].astext

        # Base query to find work that isn't currently active
        base_query = (
            select(
                models.QueueItem.session_id,
                sender_name_expr.label("sender_name"),
                target_name_expr.label("target_name"),
                task_type_expr.label("task_type"),
            )
            .outerjoin(
                models.ActiveQueueSession,
                (
                    (
                        models.QueueItem.session_id
                        == models.ActiveQueueSession.session_id
                    )
                    | (
                        models.QueueItem.session_id.is_(None)
                        & models.ActiveQueueSession.session_id.is_(None)
                    )
                )
                & (
                    (sender_name_expr == models.ActiveQueueSession.sender_name)
                    | (
                        sender_name_expr.is_(None)
                        & models.ActiveQueueSession.sender_name.is_(None)
                    )
                )
                & (
                    (target_name_expr == models.ActiveQueueSession.target_name)
                    | (
                        target_name_expr.is_(None)
                        & models.ActiveQueueSession.target_name.is_(None)
                    )
                )
                & (task_type_expr == models.ActiveQueueSession.task_type),
            )
            .where(~models.QueueItem.processed)
            .where(models.ActiveQueueSession.id.is_(None))
        )

        # Separate queries for deriver and webhook tasks
        deriver_tasks_query = base_query.where(
            task_type_expr.in_(["summary", "representation"])
        ).group_by(
            models.QueueItem.session_id,
            sender_name_expr,
            target_name_expr,
            task_type_expr,
        )

        webhook_tasks_query = base_query.where(task_type_expr == "webhook").group_by(
            models.QueueItem.session_id,
            sender_name_expr,
            target_name_expr,
            task_type_expr,
        )

        # Union the results
        final_query = deriver_tasks_query.union_all(webhook_tasks_query).limit(
            self.workers
        )

        result = await db.execute(final_query)

        # Convert rows to WorkUnit objects and filter out those that are already active
        available_work_units: list[WorkUnit] = []
        for row in result.fetchall():
            try:
                work_unit = create_work_unit(
                    task_type=row.task_type,
                    session_id=row.session_id,
                    sender_name=row.sender_name,
                    target_name=row.target_name,
                )

                # Check if this work unit is already being processed
                if not await self._is_work_unit_active(db, work_unit):
                    available_work_units.append(work_unit)

                # Stop once we have enough work units
                if len(available_work_units) >= self.workers:
                    break

            except ValueError as e:
                logger.warning(f"Skipping invalid work unit: {e}")
                continue

        return available_work_units

    async def _is_work_unit_active(self, db: AsyncSession, work_unit: WorkUnit) -> bool:
        """Check if a work unit is currently being processed using unique key."""
        result = await db.execute(
            select(models.ActiveQueueSession.id)
            .where(
                models.ActiveQueueSession.work_unit_key == work_unit.get_unique_key()
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def polling_loop(self):
        """Main polling loop to find and process new work units"""
        logger.debug("Starting polling loop")
        try:
            while not self.shutdown_event.is_set():
                if self.queue_empty_flag.is_set():
                    # logger.debug("Queue empty flag set, waiting")
                    await asyncio.sleep(settings.DERIVER.POLLING_SLEEP_INTERVAL_SECONDS)
                    self.queue_empty_flag.clear()
                    continue

                # Check if we have capacity before querying
                if self.semaphore.locked():
                    # logger.debug("All workers busy, waiting")
                    await asyncio.sleep(settings.DERIVER.POLLING_SLEEP_INTERVAL_SECONDS)
                    continue

                # Use the dependency for transaction safety
                async with tracked_db("queue_polling_loop") as db:
                    try:
                        new_work_units = await self.get_available_work_units(db)

                        if new_work_units and not self.shutdown_event.is_set():
                            for work_unit in new_work_units:
                                try:
                                    # Try to claim the work unit using unique key approach
                                    work_unit_dict = work_unit.to_dict()
                                    await db.execute(
                                        insert(models.ActiveQueueSession).values(
                                            work_unit_key=work_unit.get_unique_key(),
                                            work_unit_data=work_unit_dict,
                                            task_type=work_unit_dict.get("task_type"),
                                        )
                                    )
                                    await db.commit()

                                    # Track this work unit
                                    self.track_work_unit(work_unit)
                                    logger.debug(
                                        f"Claimed work unit {work_unit} for processing"
                                    )

                                    # Create a new task for processing this work unit
                                    if not self.shutdown_event.is_set():
                                        task: Task[None] = asyncio.create_task(
                                            self.process_work_unit(work_unit)
                                        )
                                        self.add_task(task)
                                except IntegrityError:
                                    # Note: rollback is handled by tracked_db dependency
                                    logger.debug(
                                        f"Failed to claim work unit {work_unit}, already owned"
                                    )
                        else:
                            self.queue_empty_flag.set()
                            await asyncio.sleep(
                                settings.DERIVER.POLLING_SLEEP_INTERVAL_SECONDS
                            )
                    except Exception as e:
                        logger.exception("Error in polling loop")
                        if settings.SENTRY.ENABLED:
                            sentry_sdk.capture_exception(e)
                        # Note: rollback is handled by tracked_db dependency
                        await asyncio.sleep(
                            settings.DERIVER.POLLING_SLEEP_INTERVAL_SECONDS
                        )
        finally:
            logger.info("Polling loop stopped")

    ######################
    # Queue Worker Logic #
    ######################

    @sentry_sdk.trace
    async def process_work_unit(self, work_unit: WorkUnit):
        """Process all messages for a specific work unit by routing to the correct handler."""
        logger.debug(f"Starting to process work unit {work_unit}")
        async with (
            self.semaphore,
            tracked_db("queue_process_work_unit") as db,
        ):  # Hold the semaphore for the entire work unit duration
            try:
                message_count = 0
                while not self.shutdown_event.is_set():
                    message = await self.get_next_message(db, work_unit)
                    if not message:
                        logger.debug(f"No more messages for work unit {work_unit}")
                        break

                    message_count += 1
                    try:
                        logger.info(
                            f"Processing message {message.payload['message_id']} from work unit {work_unit}"
                        )
                        await process_item(
                            self.client, message.task_type, message.payload
                        )
                        logger.debug(f"Successfully processed message {message.id}")
                    except Exception as e:
                        logger.error(
                            f"Error processing message {message.id}: {str(e)}",
                            exc_info=True,
                        )
                        if settings.SENTRY.ENABLED:
                            sentry_sdk.capture_exception(e)
                    finally:
                        # Prevent malformed messages from stalling queue indefinitely
                        message.processed = True
                        await db.commit()
                        logger.debug(f"Marked message {message.id} as processed")

                    if self.shutdown_event.is_set():
                        logger.debug(
                            f"Shutdown requested, stopping processing for work unit {work_unit}"
                        )
                        break

                    # Update last_updated timestamp to show this work unit is still being processed
                    await db.execute(
                        update(models.ActiveQueueSession)
                        .where(
                            models.ActiveQueueSession.work_unit_key
                            == work_unit.get_unique_key()
                        )
                        .values(last_updated=func.now())
                    )
                    await db.commit()

                logger.debug(
                    f"Completed processing work unit {work_unit}, processed {message_count} messages"
                )
            finally:
                # Remove work unit from active_queue_sessions when done
                logger.debug(f"Removing work unit {work_unit} from active sessions")
                await db.execute(
                    delete(models.ActiveQueueSession).where(
                        models.ActiveQueueSession.work_unit_key
                        == work_unit.get_unique_key()
                    )
                )
                await db.commit()
                self.untrack_work_unit(work_unit)

    @sentry_sdk.trace
    async def get_next_message(self, db: AsyncSession, work_unit: WorkUnit):
        """Get the next unprocessed message for a specific work unit."""
        # Build base query
        query = (
            select(models.QueueItem)
            .where(~models.QueueItem.processed)
            .order_by(models.QueueItem.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )

        # Add work unit specific conditions using polymorphic method
        conditions = work_unit.build_queue_item_conditions(models.QueueItem)
        query = query.where(*conditions)

        result = await db.execute(query)
        return result.scalar_one_or_none()


async def main():
    logger.debug("Starting queue manager")
    manager = QueueManager()
    try:
        await manager.initialize()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sentry_sdk.capture_exception(e)
    finally:
        logger.debug("Main function exiting")
