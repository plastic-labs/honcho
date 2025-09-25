import asyncio
import signal
from asyncio import Task
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from logging import getLogger
from typing import NamedTuple

import sentry_sdk
from dotenv import load_dotenv
from nanoid import generate as generate_nanoid
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import BigInteger, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.deriver.consumer import process_items
from src.deriver.utils import parse_work_unit_key
from src.models import QueueItem

logger = getLogger(__name__)

load_dotenv(override=True)


class WorkerOwnership(NamedTuple):
    """Represents the instance of a work unit that a worker is processing."""

    work_unit_key: str
    aqs_id: str  # The ID of the ActiveQueueSession that the worker is processing


class QueueManager:
    def __init__(self):
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.active_tasks: set[asyncio.Task[None]] = set()
        self.worker_ownership: dict[str, WorkerOwnership] = {}
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

    def track_worker_work_unit(
        self, worker_id: str, work_unit_key: str, aqs_id: str
    ) -> None:
        """Track a work unit owned by a specific worker"""
        self.worker_ownership[worker_id] = WorkerOwnership(work_unit_key, aqs_id)

    def untrack_worker_work_unit(self, worker_id: str, work_unit_key: str) -> None:
        """Remove a work unit from worker tracking"""
        ownership = self.worker_ownership.get(worker_id)
        if ownership and ownership.work_unit_key == work_unit_key:
            del self.worker_ownership[worker_id]

    def create_worker_id(self) -> str:
        """Generate a unique worker ID for this processing task"""
        return generate_nanoid()

    def get_total_owned_work_units(self) -> int:
        """Get the total number of work units owned by all workers"""
        return len(self.worker_ownership)

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
        total_work_units = self.get_total_owned_work_units()
        if total_work_units > 0:
            logger.debug(f"Cleaning up {total_work_units} owned work units...")
            try:
                # Use the tracked_db dependency for transaction safety
                async with tracked_db("queue_cleanup") as db:
                    aqs_ids = [
                        ownership.aqs_id for ownership in self.worker_ownership.values()
                    ]
                    if aqs_ids:
                        await db.execute(
                            delete(models.ActiveQueueSession).where(
                                models.ActiveQueueSession.id.in_(aqs_ids)
                            )
                        )
                    await db.commit()
                    logger.info("Cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)
            finally:
                self.worker_ownership.clear()

    ##########################
    # Polling and Scheduling #
    ##########################

    async def cleanup_stale_work_units(self) -> None:
        """Clean up stale work units"""
        async with tracked_db("cleanup_stale_work_units") as db:
            cutoff = datetime.now(timezone.utc) - timedelta(
                minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES
            )

            stale_ids = (
                (
                    await db.execute(
                        select(models.ActiveQueueSession.id)
                        .where(models.ActiveQueueSession.last_updated < cutoff)
                        .order_by(models.ActiveQueueSession.last_updated)
                        .with_for_update(skip_locked=True)
                    )
                )
                .scalars()
                .all()
            )

            # Delete only the records we successfully got locks for
            if stale_ids:
                await db.execute(
                    delete(models.ActiveQueueSession).where(
                        models.ActiveQueueSession.id.in_(stale_ids)
                    )
                )
            await db.commit()

    async def get_and_claim_work_units(self) -> dict[str, str]:
        """
        Get available work units that aren't being processed.
        Returns a dict mapping work_unit_key to aqs_id.
        """
        limit: int = max(0, self.workers - self.get_total_owned_work_units())
        if limit == 0:
            return {}
        async with tracked_db(
            "get_available_work_units"
        ) as db:  # Get number of available workers
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
                return {}

            claimed_mapping = await self.claim_work_units(db, available_units)
            await db.commit()

            return claimed_mapping

    async def claim_work_units(
        self, db: AsyncSession, work_unit_keys: Sequence[str]
    ) -> dict[str, str]:
        """
        Claim work units and return a mapping of work_unit_key to aqs_id.
        Returns only the work units that were successfully claimed.
        """
        values = [{"work_unit_key": key} for key in work_unit_keys]

        stmt = (
            insert(models.ActiveQueueSession)
            .values(values)
            .on_conflict_do_nothing()
            .returning(
                models.ActiveQueueSession.work_unit_key, models.ActiveQueueSession.id
            )
        )

        result = await db.execute(stmt)
        claimed_rows = result.all()
        claimed_mapping = {row[0]: row[1] for row in claimed_rows}
        logger.debug(
            f"Claimed {len(claimed_mapping)} work units: {list(claimed_mapping.keys())}"
        )
        return claimed_mapping

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
                    await self.cleanup_stale_work_units()
                    claimed_work_units = await self.get_and_claim_work_units()
                    if claimed_work_units:
                        for work_unit_key, aqs_id in claimed_work_units.items():
                            # Create a new task for processing this work unit
                            if not self.shutdown_event.is_set():
                                # Track worker ownership
                                worker_id = self.create_worker_id()
                                self.track_worker_work_unit(
                                    worker_id, work_unit_key, aqs_id
                                )

                                task: Task[None] = asyncio.create_task(
                                    self.process_work_unit(work_unit_key, worker_id)
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
    async def process_work_unit(self, work_unit_key: str, worker_id: str) -> None:
        """Process all messages for a specific work unit by routing to the correct handler."""
        logger.debug(
            f"Worker {worker_id} starting to process work unit {work_unit_key}"
        )
        async with self.semaphore:
            message_count = 0
            try:
                parsed_key = parse_work_unit_key(work_unit_key)
                task_type = parsed_key["task_type"]

                while not self.shutdown_event.is_set():
                    # Get worker ownership info for verification
                    ownership = self.worker_ownership.get(worker_id)
                    if not ownership or ownership.work_unit_key != work_unit_key:
                        logger.warning(
                            f"Worker {worker_id} lost ownership of work unit {work_unit_key}, stopping processing {work_unit_key}"
                        )
                        break

                    messages_to_process: list[QueueItem] = await self.get_message_batch(
                        task_type, work_unit_key, ownership.aqs_id
                    )
                    logger.debug(
                        f"Worker {worker_id} retrieved {len(messages_to_process)} messages for work unit {work_unit_key} (AQS ID: {ownership.aqs_id})"
                    )
                    if not messages_to_process:
                        logger.debug(
                            f"No more messages to process for work unit {work_unit_key} for worker {worker_id}"
                        )
                        break
                    try:
                        payloads = [msg.payload for msg in messages_to_process]
                        await process_items(task_type, payloads)
                    except Exception as e:
                        logger.error(
                            f"Error processing tasks for work unit {work_unit_key}: {e}",
                            exc_info=True,
                        )
                        if settings.SENTRY.ENABLED:
                            sentry_sdk.capture_exception(e)

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
                ownership: WorkerOwnership | None = self.worker_ownership.get(worker_id)
                if ownership and ownership.work_unit_key == work_unit_key:
                    removed = await self._cleanup_work_unit(
                        ownership.aqs_id, work_unit_key
                    )
                else:
                    removed = False

                self.untrack_worker_work_unit(worker_id, work_unit_key)
                if removed and message_count > 0:
                    # Only publish webhook if we actually removed an active session
                    try:
                        from src.webhooks.events import (
                            QueueEmptyEvent,
                            publish_webhook_event,
                        )

                        parsed_key = parse_work_unit_key(work_unit_key)
                        if parsed_key["task_type"] in ["representation", "summary"]:
                            logger.debug(
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

    @sentry_sdk.trace
    async def get_message_batch(
        self, task_type: str, work_unit_key: str, aqs_id: str
    ) -> list[QueueItem]:
        """
        Get a batch of unprocessed messages for a specific work unit ordered by id.
        For representation tasks, this will be a batch of messages up to REPRESENTATION_BATCH_MAX_TOKENS.
        For other tasks, it will be a single message.

        Args:
            task_type: The type of task to process
            work_unit_key: The key of the work unit to process
            aqs_id: The ID of the active queue session to process

        Returns:
            A list of QueueItem objects
        """
        async with tracked_db("get_message_batch") as db:
            # ActiveQueueSession conditions for worker ownership verification
            aqs_conditions = [
                models.ActiveQueueSession.work_unit_key == work_unit_key,
                models.ActiveQueueSession.id == aqs_id,
            ]

            if task_type != "representation":
                # For non-representation tasks, just get the next single message.
                query = (
                    select(models.QueueItem)
                    .join(
                        models.ActiveQueueSession,
                        models.QueueItem.work_unit_key
                        == models.ActiveQueueSession.work_unit_key,
                    )
                    .where(models.QueueItem.work_unit_key == work_unit_key)
                    .where(~models.QueueItem.processed)
                    .where(*aqs_conditions)
                    .order_by(models.QueueItem.id)
                    .limit(1)
                )
                result = await db.execute(query)
                messages = result.scalars().all()
            else:
                # For representation tasks, get a batch based on token count.
                # Always get at least the first message, then include additional messages
                # as long as cumulative token count stays within limit.
                # Join with messages table to get the actual token_count

                # Create CTE with row numbers and cumulative token counts
                cte = (
                    select(
                        models.QueueItem.id,
                        func.row_number()
                        .over(order_by=models.QueueItem.id)
                        .label("row_num"),
                        func.sum(models.Message.token_count)
                        .over(order_by=models.QueueItem.id)
                        .label("cumulative_token_count"),
                    )
                    .select_from(
                        models.QueueItem.__table__.join(
                            models.Message.__table__,
                            func.cast(
                                models.QueueItem.payload["message_id"].astext,
                                BigInteger,
                            )
                            == models.Message.id,
                        )
                    )
                    .where(models.QueueItem.work_unit_key == work_unit_key)
                    .where(~models.QueueItem.processed)
                    .order_by(models.QueueItem.id)
                    .cte()
                )

                # Select messages where either:
                # 1. It's the first message (row_num = 1), OR
                # 2. The cumulative token count is within the limit
                # Also ensure worker ownership verification by joining with ActiveQueueSession
                query = (
                    select(models.QueueItem)
                    .join(
                        models.ActiveQueueSession,
                        models.QueueItem.work_unit_key
                        == models.ActiveQueueSession.work_unit_key,
                    )
                    .where(
                        models.QueueItem.id.in_(
                            select(cte.c.id).where(
                                (cte.c.row_num == 1)
                                | (
                                    cte.c.cumulative_token_count
                                    <= settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
                                )
                            )
                        )
                    )
                    .where(*aqs_conditions)
                    .order_by(models.QueueItem.id)
                )

                result = await db.execute(query)
                messages = result.scalars().all()

            # Important: commit to avoid tracked_db's rollback expiring the instance
            # We rely on expire_on_commit=False to keep attributes accessible post-close
            await db.commit()
            return list(messages)

    async def mark_messages_as_processed(
        self, messages: list[QueueItem], work_unit_key: str
    ) -> None:
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

    async def _cleanup_work_unit(
        self,
        aqs_id: str,
        work_unit_key: str,
    ) -> bool:
        """
        Clean up a specific work unit session by both work_unit_key and AQS ID.
        """
        async with tracked_db("cleanup_work_unit") as db:
            result = await db.execute(
                delete(models.ActiveQueueSession)
                .where(models.ActiveQueueSession.id == aqs_id)
                .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
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
