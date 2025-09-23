import asyncio
import signal
from asyncio import Task
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from logging import getLogger

import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src.config import settings
from src.deriver.utils import get_work_unit_key, parse_work_unit_key
from src.models import QueueItem

from .. import models
from ..dependencies import tracked_db
from .consumer import process_item
from .queue_payload import create_dream_payload

logger = getLogger(__name__)

load_dotenv(override=True)


class DreamScheduler:
    def __init__(self):
        self.pending_dreams: dict[str, asyncio.Task[None]] = {}

    def schedule_dream(
        self, work_unit_key: str, delay_minutes: int | None = None
    ) -> None:
        # Skip scheduling if dreams are disabled
        if not settings.DREAM.ENABLED:
            return

        if delay_minutes is None:
            delay_minutes = settings.DREAM.IDLE_TIMEOUT_MINUTES

        # Cancel any existing dream for this work unit
        self.cancel_dream(work_unit_key)

        task = asyncio.create_task(self._delayed_dream(work_unit_key, delay_minutes))
        self.pending_dreams[work_unit_key] = task
        task.add_done_callback(lambda t: self.pending_dreams.pop(work_unit_key, None))

        logger.info(f"Scheduled dream for {work_unit_key} in {delay_minutes} minutes")

    def cancel_dream(self, work_unit_key: str) -> bool:
        """
        Returns:
            True if a dream was cancelled, False if none was pending
        """
        if work_unit_key in self.pending_dreams:
            task = self.pending_dreams.pop(work_unit_key)
            task.cancel()
            return True
        return False

    async def _delayed_dream(self, work_unit_key: str, delay_minutes: int) -> None:
        try:
            await asyncio.sleep(delay_minutes * 60)

            if await self._should_execute_dreams(work_unit_key):
                await self._execute_dreams(work_unit_key)
                logger.info(f"Executed dreams for {work_unit_key}")
        except asyncio.CancelledError:
            logger.info(f"Dream task cancelled for {work_unit_key}")
        except Exception as e:
            logger.error(f"Error in delayed dream for {work_unit_key}: {str(e)}")
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)

    async def _should_execute_dreams(self, work_unit_key: str) -> bool:
        async with tracked_db("dream_idle_check") as db:
            query = (
                select(func.count(models.QueueItem.id))
                .where(models.QueueItem.work_unit_key == work_unit_key)
                .where(~models.QueueItem.processed)
            )
            result = await db.execute(query)
            unprocessed_count = result.scalar() or 0
            await db.commit()

            # If there are unprocessed messages, user became active again
            if unprocessed_count > 0:
                return False

        # TODO: Add additional logic to determine if dreams are actually needed!!
        # such as number of documents added etc
        return True

    async def _execute_dreams(self, work_unit_key: str) -> None:
        parsed_key = parse_work_unit_key(work_unit_key)

        # Create dream payload using configured settings
        dream_payload = create_dream_payload(
            workspace_name=parsed_key["workspace_name"],
            session_name=parsed_key["session_name"] or "",
            sender_name=parsed_key["sender_name"] or "",
            target_name=parsed_key["target_name"] or "",
            dream_type="consolidate",
        )

        async with tracked_db("dream_enqueue") as db:
            dream_record = {
                "work_unit_key": get_work_unit_key(
                    "dream",
                    {
                        "workspace_name": parsed_key["workspace_name"],
                        "session_name": parsed_key["session_name"],
                        "sender_name": parsed_key["sender_name"],
                        "target_name": parsed_key["target_name"],
                    },
                ),
                "payload": dream_payload,
                "session_id": None,
                "task_type": "dream",
            }

            await db.execute(insert(models.QueueItem), [dream_record])
            await db.commit()

            logger.info(f"Enqueued dream task {work_unit_key}")

    async def shutdown(self) -> None:
        """Cancel all pending dreams during shutdown."""
        if self.pending_dreams:
            logger.info(f"Cancelling {len(self.pending_dreams)} pending dreams...")
            for task in self.pending_dreams.values():
                task.cancel()
            # Wait for all tasks to finish cancellation
            await asyncio.gather(*self.pending_dreams.values(), return_exceptions=True)
            self.pending_dreams.clear()


class QueueManager:
    def __init__(self):
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.active_tasks: set[asyncio.Task[None]] = set()
        self.owned_work_units: set[str] = set()
        self.queue_empty_flag: asyncio.Event = asyncio.Event()

        # Initialize from settings
        self.workers: int = settings.DERIVER.WORKERS
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(self.workers)

        # Initialize dream scheduler
        self.dream_scheduler: DreamScheduler = DreamScheduler()

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

        # Set global dream scheduler reference for enqueue.py
        from .enqueue import set_dream_scheduler

        set_dream_scheduler(self.dream_scheduler)

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

        # Cancel all pending dreams
        await self.dream_scheduler.shutdown()

        if self.active_tasks:
            logger.info(
                f"Waiting for {len(self.active_tasks)} active tasks to complete..."
            )
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

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

    async def get_and_claim_work_units(self) -> Sequence[str]:
        """
        Get available work units that aren't being processed.
        Returns a list of work unit keys.
        """
        claimed_units: list[str] = []

        async with tracked_db("get_available_work_units") as db:
            # Clean up stale work units
            five_minutes_ago = datetime.now(timezone.utc) - timedelta(
                minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES
            )
            await db.execute(
                delete(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.last_updated < five_minutes_ago
                )
            )

            # Get number of available workers
            limit: int = max(0, self.workers - len(self.owned_work_units))

            query = (
                select(models.QueueItem.work_unit_key)
                .outerjoin(
                    models.ActiveQueueSession,
                    models.QueueItem.work_unit_key
                    == models.ActiveQueueSession.work_unit_key,
                )
                .where(~models.QueueItem.processed)
                .where(models.QueueItem.work_unit_key.isnot(None))
                .where(models.ActiveQueueSession.work_unit_key.is_(None))
                .distinct()
                .limit(limit)
            )

            result = await db.execute(query)
            available_units = result.scalars().all()
            if not available_units:
                await db.commit()
                return []

            claimed_units = await self.claim_work_units(db, available_units)
            await db.commit()

            for work_unit in claimed_units:
                self.track_work_unit(work_unit)

            return claimed_units

    async def claim_work_units(
        self, db: AsyncSession, work_unit_keys: Sequence[str]
    ) -> list[str]:
        from sqlalchemy.dialects.postgresql import insert

        values = [{"work_unit_key": key} for key in work_unit_keys]

        stmt = (
            insert(models.ActiveQueueSession)
            .values(values)
            .on_conflict_do_nothing()
            .returning(models.ActiveQueueSession.work_unit_key)
        )

        result = await db.execute(stmt)
        claimed_units = result.scalars().all()
        logger.debug(f"Claimed {len(claimed_units)} work units")
        return list(claimed_units)

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

                try:
                    new_work_units = await self.get_and_claim_work_units()
                    if new_work_units:
                        for work_unit in new_work_units:
                            # Create a new task for processing this work unit
                            if not self.shutdown_event.is_set():
                                task: Task[None] = asyncio.create_task(
                                    self.process_work_unit(work_unit)
                                )
                                self.add_task(task)
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
                    await asyncio.sleep(settings.DERIVER.POLLING_SLEEP_INTERVAL_SECONDS)
        finally:
            logger.info("Polling loop stopped")

    ######################
    # Queue Worker Logic #
    ######################

    @sentry_sdk.trace
    async def process_work_unit(self, work_unit_key: str):
        """Process all messages for a specific work unit by routing to the correct handler."""
        logger.debug(f"Starting to process work unit {work_unit_key}")
        parsed_key = parse_work_unit_key(work_unit_key)
        async with (
            self.semaphore
        ):  # Hold the semaphore for the entire work unit duration
            message_count = 0
            try:
                while not self.shutdown_event.is_set():
                    message = await self.get_next_message(work_unit_key)
                    if not message:
                        logger.debug(f"No more messages for work unit {work_unit_key}")
                        break

                    message_count += 1
                    try:
                        logger.info(
                            f"Processing item for task type {message.task_type} with id {message.id} from work unit {work_unit_key}"
                        )
                        await process_item(message.task_type, message.payload)
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

                    # Prevent malformed messages from stalling queue indefinitely
                    async with tracked_db("process_message") as db:
                        await db.execute(
                            update(models.QueueItem)
                            .where(models.QueueItem.id == message.id)
                            .values(processed=True)
                        )

                        await db.execute(
                            update(models.ActiveQueueSession)
                            .where(
                                models.ActiveQueueSession.work_unit_key == work_unit_key
                            )
                            .values(last_updated=func.now())
                        )

                        await db.commit()

                    if self.shutdown_event.is_set():
                        logger.debug(
                            "Shutdown requested, stopping processing for work unit %s",
                            work_unit_key,
                        )
                        break

                logger.debug(
                    f"Completed processing work unit {work_unit_key}, processed {message_count} messages"
                )

                # Schedule dream if we processed messages and might benefit from dreaming
                if message_count > 0 and parsed_key["task_type"] == "representation":
                    self.dream_scheduler.schedule_dream(work_unit_key)

            finally:
                # Remove work unit from active_queue_sessions when done
                logger.debug(f"Removing work unit {work_unit_key} from active sessions")
                removed = await self._cleanup_work_unit(work_unit_key)

                if removed and message_count > 0:
                    # Only publish webhook if we actually removed an active session
                    try:
                        from src.webhooks.events import (
                            QueueEmptyEvent,
                            publish_webhook_event,
                        )

                        if parsed_key["task_type"] in ["representation", "summary"]:
                            logger.info(
                                f"Publishing queue.empty event for {work_unit_key}"
                            )
                            await publish_webhook_event(
                                QueueEmptyEvent(
                                    workspace_id=parsed_key["workspace_name"],
                                    queue_type=parsed_key["task_type"],
                                    session_id=parsed_key["session_name"],
                                    sender_name=parsed_key["sender_name"],
                                    observer_name=parsed_key["target_name"],
                                )
                            )
                        else:
                            logger.debug(
                                f"Skipping queue.empty event for webhook work unit {work_unit_key}"
                            )
                    except Exception:
                        logger.exception("Error triggering queue_empty webhook")
                else:
                    logger.debug(
                        f"Work unit {work_unit_key} already cleaned up by another worker, skipping webhook"
                    )

                self.untrack_work_unit(work_unit_key)

    @sentry_sdk.trace
    async def get_next_message(self, work_unit_key: str) -> QueueItem | None:
        """Get the next unprocessed message for a specific work unit."""
        async with tracked_db("get_next_message") as db:
            query = (
                select(models.QueueItem)
                .where(models.QueueItem.work_unit_key == work_unit_key)
                .where(~models.QueueItem.processed)
                .order_by(models.QueueItem.id)
                .limit(1)
            )
            result = await db.execute(query)
            message = result.scalar_one_or_none()
            # Important: commit to avoid tracked_db's rollback expiring the instance
            # We rely on expire_on_commit=False to keep attributes accessible post-close
            await db.commit()
            return message

    async def _cleanup_work_unit(self, work_unit_key: str) -> bool:
        async with tracked_db("cleanup_work_unit") as db:
            result = await db.execute(
                delete(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.work_unit_key == work_unit_key
                )
            )
            await db.commit()
            return result.rowcount > 0


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
