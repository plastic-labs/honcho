import asyncio
import signal
from asyncio import Task
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from logging import getLogger

import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src.config import settings
from src.models import QueueItem

from .. import exceptions, models
from ..dependencies import tracked_db
from .consumer import process_items

logger = getLogger(__name__)

load_dotenv(override=True)


class QueueManager:
    def __init__(self):
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.active_tasks: set[asyncio.Task[None]] = set()
        self.owned_work_units: set[str] = set()
        self.queue_empty_flag: asyncio.Event = asyncio.Event()

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
        async with self.semaphore:
            message_count = 0
            try:
                while not self.shutdown_event.is_set():
                    candidate_messages = await self.get_message_batch(
                        work_unit_key,
                        limit=10,  # hard limit of 10 messages per batch
                    )
                    if not candidate_messages:
                        logger.debug(f"No more messages for work unit {work_unit_key}")
                        break

                    next_message = candidate_messages[0]
                    task_type = next_message.task_type

                    messages_to_process: list[QueueItem] = []

                    if task_type != "representation":
                        messages_to_process.append(next_message)
                    else:
                        # It's a representation task, build a batch
                        token_count = 0
                        max_tokens = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS

                        for msg in candidate_messages:
                            msg_tokens = msg.token_count or 0
                            if (
                                not messages_to_process
                                or token_count + msg_tokens <= max_tokens
                            ):
                                messages_to_process.append(msg)
                                token_count += msg_tokens
                            else:
                                break

                    if not messages_to_process:
                        logger.warning(
                            "No messages to process, breaking loop for work unit %s to prevent infinite loop.",
                            work_unit_key,
                        )
                        break

                    # Process the batch/single item
                    try:
                        raw_payloads = [msg.payload for msg in messages_to_process]
                        await process_items(task_type, raw_payloads)
                    except exceptions.LLMError as e:
                        logger.error(
                            f"LLM returned bad JSON for messages in work unit {work_unit_key}, re-queueing",
                        )
                        if settings.SENTRY.ENABLED:
                            sentry_sdk.capture_exception(e)
                        continue  # Don't mark as processed, allow re-queue
                    except Exception as e:
                        logger.error(
                            f"Error processing tasks for work unit {work_unit_key}: {e}",
                            exc_info=True,
                        )
                        if settings.SENTRY.ENABLED:
                            sentry_sdk.capture_exception(e)

                    # Mark messages as processed (only for non-LLM errors)
                    await self.mark_messages_as_processed(
                        messages_to_process, work_unit_key
                    )
                    message_count += len(messages_to_process)

                    # Check for shutdown after processing each batch
                    if self.shutdown_event.is_set():
                        logger.debug(
                            "Shutdown requested, stopping processing for work unit %s",
                            work_unit_key,
                        )
                        break

            finally:
                # Remove work unit from active_queue_sessions when done
                logger.debug(f"Removing work unit {work_unit_key} from active sessions")
                removed = await self._cleanup_work_unit(work_unit_key)

                if removed and message_count > 0:
                    # Only publish webhook if we actually removed an active session
                    try:
                        from src.deriver.utils import parse_work_unit_key
                        from src.webhooks.events import (
                            QueueEmptyEvent,
                            publish_webhook_event,
                        )

                        parsed_key = parse_work_unit_key(work_unit_key)
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
    async def get_message_batch(
        self, work_unit_key: str, limit: int
    ) -> list[QueueItem]:
        """Get a batch of unprocessed messages for a specific work unit ordered by id."""
        async with tracked_db("get_message_batch") as db:
            query = (
                select(models.QueueItem)
                .where(models.QueueItem.work_unit_key == work_unit_key)
                .where(~models.QueueItem.processed)
                .order_by(models.QueueItem.id)
                .limit(limit)
            )
            result = await db.execute(query)
            messages = result.scalars().all()
            # Important: commit to avoid tracked_db's rollback expiring the instance
            # We rely on expire_on_commit=False to keep attributes accessible post-close
            await db.commit()
            return list(messages)

    async def mark_messages_as_processed(
        self, messages: list[QueueItem], work_unit_key: str
    ):
        if not messages:
            return
        async with tracked_db("process_message_batch") as db:
            message_ids = [msg.id for msg in messages]
            await db.execute(
                update(models.QueueItem)
                .where(models.QueueItem.id.in_(message_ids))
                .values(processed=True)
            )
            await db.execute(
                update(models.ActiveQueueSession)
                .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
                .values(last_updated=func.now())
            )
            await db.commit()

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
