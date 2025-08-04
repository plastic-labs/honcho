import asyncio
import signal
from asyncio import Task
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from logging import getLogger
from sqlite3 import IntegrityError

import httpx
import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src.config import settings

from .. import models
from ..dependencies import tracked_db
from .consumer import process_item

logger = getLogger(__name__)

load_dotenv(override=True)


class QueueManager:
    def __init__(self):
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.active_tasks: set[asyncio.Task[None]] = set()
        self.owned_work_units: set[str] = set()
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

    def add_task(self, task: asyncio.Task[None]) -> None:
        """Track a new task"""
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    def track_work_unit(self, work_unit_key: str) -> None:
        """Track a new work unit owned by this process"""
        self.owned_work_units.add(work_unit_key)

    def untrack_work_unit(self, work_unit_key: str) -> None:
        """Remove a work unit from tracking"""
        self.owned_work_units.discard(work_unit_key)

    async def initialize(self) -> None:
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

    async def shutdown(self, sig: signal.Signals) -> None:
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

    async def cleanup(self) -> None:
        """Clean up owned work units"""
        if self.owned_work_units:
            logger.info(f"Cleaning up {len(self.owned_work_units)} owned work units...")
            try:
                # Use the tracked_db dependency for transaction safety
                async with tracked_db("queue_cleanup") as db:
                    for work_unit_key in self.owned_work_units:
                        await db.execute(
                            delete(models.ActiveQueueSession).where(
                                models.ActiveQueueSession.work_unit_key == work_unit_key
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

    async def get_available_work_units(self, db: AsyncSession) -> Sequence[str]:
        """
        Get available work units that aren't being processed.
        Returns a list of work unit keys.
        """
        # Clean up stale work units
        five_minutes_ago = datetime.now(timezone.utc) - timedelta(
            minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES
        )
        await db.execute(
            delete(models.ActiveQueueSession).where(
                models.ActiveQueueSession.last_updated < five_minutes_ago
            )
        )

        query = (
            select(models.QueueItem.work_unit_key)
            .where(~models.QueueItem.processed)
            .where(models.QueueItem.work_unit_key.isnot(None))
            .where(
                models.QueueItem.work_unit_key.not_in(
                    select(models.ActiveQueueSession.work_unit_key)
                )
            )
            .distinct()
            .limit(self.workers)
        )

        result = await db.execute(query)
        return result.scalars().all()

    async def polling_loop(self) -> None:
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
                                    await db.execute(
                                        insert(models.ActiveQueueSession)
                                        .values(work_unit_key=work_unit)
                                        .on_conflict_do_nothing()
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
                                    # Rollback the failed transaction to clear the error state
                                    await db.rollback()
                                    logger.debug(
                                        f"Failed to claim work unit {work_unit}, already owned"
                                    )
                            # If we couldn't claim any work units, avoid tight loop
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
    async def process_work_unit(self, work_unit_key: str):
        """Process all messages for a specific work unit by routing to the correct handler."""
        logger.debug(f"Starting to process work unit {work_unit_key}")
        async with (
            self.semaphore,
            tracked_db("queue_process_work_unit") as db,
        ):  # Hold the semaphore for the entire work unit duration
            try:
                message_count = 0
                while not self.shutdown_event.is_set():
                    message = await self.get_next_message(db, work_unit_key)
                    if not message:
                        logger.debug(f"No more messages for work unit {work_unit_key}")
                        break

                    message_count += 1
                    try:
                        logger.info(
                            f"Processing item for task type {message.task_type} with id {message.id} from work unit {work_unit_key}"
                        )
                        await process_item(
                            self.client, message.task_type, message.payload
                        )
                        logger.debug(
                            f"Successfully processed queue item for task type {message.task_type} with id {message.id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error processing queue item for task type {message.task_type} with id {message.id}: {str(e)}",
                            exc_info=True,
                        )
                        if settings.SENTRY.ENABLED:
                            sentry_sdk.capture_exception(e)
                    finally:
                        # Prevent malformed messages from stalling queue indefinitely
                        message.processed = True
                        await db.commit()
                        logger.debug(f"Marked queue item {message.id} as processed")

                    if self.shutdown_event.is_set():
                        logger.debug(
                            f"Shutdown requested, stopping processing for work unit {work_unit_key}"
                        )
                        break

                    # Update last_updated timestamp to show this work unit is still being processed
                    await db.execute(
                        update(models.ActiveQueueSession)
                        .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
                        .values(last_updated=func.now())
                    )
                    await db.commit()

                logger.debug(
                    f"Completed processing work unit {work_unit_key}, processed {message_count} messages"
                )
            finally:
                # Remove work unit from active_queue_sessions when done
                logger.debug(f"Removing work unit {work_unit_key} from active sessions")
                await db.execute(
                    delete(models.ActiveQueueSession).where(
                        models.ActiveQueueSession.work_unit_key == work_unit_key
                    )
                )
                await db.commit()
                self.untrack_work_unit(work_unit_key)

    @sentry_sdk.trace
    async def get_next_message(self, db: AsyncSession, work_unit_key: str):
        """Get the next unprocessed message for a specific work unit."""
        # Build base query - need to add quotes to match database format
        query = (
            select(models.QueueItem)
            .where(models.QueueItem.work_unit_key == work_unit_key)
            .where(~models.QueueItem.processed)
            .order_by(models.QueueItem.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
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
