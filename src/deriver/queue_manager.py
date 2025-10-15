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
from sqlalchemy import BigInteger, and_, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src import models, prometheus
from src.config import settings
from src.dependencies import tracked_db
from src.deriver.consumer import (
    process_item,
    process_representation_batch,
)
from src.dreamer.dream_scheduler import (
    DreamScheduler,
    get_dream_scheduler,
    set_dream_scheduler,
)
from src.models import QueueItem
from src.sentry import initialize_sentry
from src.utils.work_unit import parse_work_unit_key
from src.webhooks.events import (
    QueueEmptyEvent,
    publish_webhook_event,
)

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
        self._maintenance_task: asyncio.Task[None] | None = None

        # Initialize from settings
        self.workers: int = settings.DERIVER.WORKERS
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(self.workers)

        # Get or create the singleton dream scheduler
        existing_scheduler = get_dream_scheduler()
        if existing_scheduler is None:
            self.dream_scheduler: DreamScheduler = DreamScheduler()
            set_dream_scheduler(self.dream_scheduler)
        else:
            self.dream_scheduler = existing_scheduler

        # Initialize Sentry if enabled, using settings
        if settings.SENTRY.ENABLED:
            initialize_sentry(integrations=[AsyncioIntegration()])

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

        # Start background maintenance loop
        try:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        except Exception:
            logger.exception("Failed to start maintenance loop")

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
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(e)
            finally:
                self.worker_ownership.clear()

        # Cancel maintenance loop if running
        if self._maintenance_task is not None:
            from contextlib import suppress

            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task

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
                .limit(limit)
                .outerjoin(
                    models.ActiveQueueSession,
                    models.QueueItem.work_unit_key
                    == models.ActiveQueueSession.work_unit_key,
                )
                .where(~models.QueueItem.processed)
                .where(models.QueueItem.work_unit_key.isnot(None))
                .where(models.ActiveQueueSession.work_unit_key.is_(None))
                .distinct()
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

    async def cleanup_queue_items(self) -> None:
        """Delete processed queue items.
        Successfully processed queue items are deleted immediately,
        while errored queue items are deleted after retention window."""
        async with tracked_db("cleanup_queue_items") as db:
            now = datetime.now(timezone.utc)
            error_cutoff = now - timedelta(
                seconds=settings.DERIVER.QUEUE_ERROR_RETENTION_SECONDS
            )

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
            )
            await db.commit()

    async def _maintenance_loop(self) -> None:
        """Run periodic maintenance tasks on the queue."""
        try:
            while not self.shutdown_event.is_set():
                try:
                    await self.cleanup_queue_items()
                except Exception:
                    logger.exception("Error during maintenance cleanup")
                    if settings.SENTRY.ENABLED:
                        sentry_sdk.capture_exception()

                # Sleep until interval elapses or shutdown event is set
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=43200,  # 12 hours
                    )
                    break  # Shutdown event set
                except asyncio.TimeoutError:
                    # Timeout means it's time for next cleanup
                    pass
        except asyncio.CancelledError:
            logger.debug("Maintenance loop cancelled")
            raise

    async def _handle_processing_error(
        self,
        error: Exception,
        items: list[QueueItem],
        work_unit_key: str,
        context: str,
    ) -> None:
        """
        Handle processing errors by marking queue items as errored, logging, and forwarding to Sentry.
        We only mark the first queue item as errored so we don't potentially throw away a batch. This allows us
        to incrementally attempt to process the batch while still maintaining progress in a work unit.

        Args:
            error: The exception that occurred
            items: The queue items that were being processed
            work_unit_key: The work unit key for the queue items
            context: Context string describing what was being processed (e.g., "processing representation batch")
        """
        error_msg = f"{error.__class__.__name__}: {str(error)}"
        try:
            if items:
                await self.mark_queue_item_as_errored(
                    items[0], work_unit_key, error_msg
                )
        except Exception as mark_error:
            logger.error(
                f"Failed to mark queue items as errored for work unit {work_unit_key}: {mark_error}",
                exc_info=True,
            )

        logger.error(
            f"Error {context} for work unit {work_unit_key}: {error}",
            exc_info=True,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(error)

    async def process_work_unit(self, work_unit_key: str, worker_id: str) -> None:
        """Process all queue items for a specific work unit by routing to the correct handler."""
        logger.debug(f"Starting to process work unit {work_unit_key}")
        work_unit = parse_work_unit_key(work_unit_key)
        async with self.semaphore:
            queue_item_count = 0
            try:
                while not self.shutdown_event.is_set():
                    # Get worker ownership info for verification
                    ownership = self.worker_ownership.get(worker_id)
                    if not ownership or ownership.work_unit_key != work_unit_key:
                        logger.warning(
                            f"Worker {worker_id} lost ownership of work unit {work_unit_key}, stopping processing {work_unit_key}"
                        )
                        break
                    try:
                        if work_unit.task_type == "representation":
                            (
                                messages_context,
                                items_to_process,
                            ) = await self.get_queue_item_batch(
                                work_unit.task_type, work_unit_key, ownership.aqs_id
                            )
                            logger.debug(
                                f"Worker {worker_id} retrieved {len(messages_context)} messages and {len(items_to_process)} queue items for work unit {work_unit_key} (AQS ID: {ownership.aqs_id})"
                            )
                            if not items_to_process:
                                logger.debug(
                                    f"No more queue items to process for work unit {work_unit_key} for worker {worker_id}"
                                )
                                break

                            try:
                                await process_representation_batch(
                                    messages_context,
                                    observer=work_unit.observer,
                                    observed=work_unit.observed,
                                )
                                await self.mark_queue_items_as_processed(
                                    items_to_process, work_unit_key
                                )
                                queue_item_count += len(items_to_process)
                            except Exception as e:
                                await self._handle_processing_error(
                                    e,
                                    items_to_process,
                                    work_unit_key,
                                    "processing representation batch",
                                )

                        else:
                            queue_item = await self.get_next_queue_item(
                                work_unit.task_type, work_unit_key, ownership.aqs_id
                            )
                            if not queue_item:
                                logger.debug(
                                    f"No more queue items to process for work unit {work_unit_key} for worker {worker_id}"
                                )
                                break

                            try:
                                await process_item(
                                    work_unit.task_type, queue_item.payload
                                )
                                await self.mark_queue_items_as_processed(
                                    [queue_item], work_unit_key
                                )
                                queue_item_count += 1
                            except Exception as e:
                                await self._handle_processing_error(
                                    e,
                                    [queue_item],
                                    work_unit_key,
                                    "processing queue item",
                                )

                    except Exception as e:
                        logger.error(
                            f"Error in processing loop for work unit {work_unit_key}: {e}",
                            exc_info=True,
                        )
                        if settings.SENTRY.ENABLED:
                            sentry_sdk.capture_exception(e)

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
                if removed and queue_item_count > 0:
                    # Only publish webhook if we actually removed an active session
                    try:
                        if work_unit.task_type in ["representation", "summary"]:
                            logger.debug(
                                f"Publishing queue.empty event for {work_unit_key} in workspace {work_unit.workspace_name}"
                            )
                            await publish_webhook_event(
                                QueueEmptyEvent(
                                    workspace_id=work_unit.workspace_name,
                                    queue_type=work_unit.task_type,
                                    session_id=work_unit.session_name,
                                    observer=work_unit.observer,
                                    observed=work_unit.observed,
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
    async def get_next_queue_item(
        self, task_type: str, work_unit_key: str, aqs_id: str
    ) -> QueueItem | None:
        """Get the next queue item to process for a specific work unit."""
        if task_type == "representation":
            raise ValueError(
                "Representation tasks are not supported for get_next_queue_item"
            )
        async with tracked_db("get_next_queue_item") as db:
            # ActiveQueueSession conditions for worker ownership verification
            aqs_conditions = [
                models.ActiveQueueSession.work_unit_key == work_unit_key,
                models.ActiveQueueSession.id == aqs_id,
            ]

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
            queue_item = result.scalar_one_or_none()

            # Important: commit to avoid tracked_db's rollback expiring the instance
            # We rely on expire_on_commit=False to keep attributes accessible post-close
            await db.commit()
            return queue_item

    @sentry_sdk.trace
    async def get_queue_item_batch(
        self,
        task_type: str,
        work_unit_key: str,
        aqs_id: str,
    ) -> tuple[list[models.Message], list[QueueItem]]:
        """
        Representation-only: returns a tuple of (messages_context, items_to_process).
        - messages_context: unique Message rows (conversation turns) forming the context window
        - items_to_process: QueueItems for the current work_unit_key within that window
        """
        if task_type != "representation":
            raise ValueError(
                "Non-representation tasks are not supported for get_queue_item_batch"
            )
        async with tracked_db("get_queue_item_batch") as db:
            # For representation tasks, get a batch based on token limit.
            # Step 1: Parse work_unit_key to get session context and focused sender
            parsed_key = parse_work_unit_key(work_unit_key)

            # Verify worker still owns the work_unit_key
            ownership_check = await db.execute(
                select(models.ActiveQueueSession.id)
                .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
                .where(models.ActiveQueueSession.id == aqs_id)
            )
            if not ownership_check.scalar_one_or_none():
                # Worker lost ownership, return empty
                await db.commit()
                return [], []

            # Step 2: Build a single SQL query that:
            # 1. Finds the earliest unprocessed message for this work_unit_key
            # 2. Gets ALL messages from that point forward (for conversational context)
            # 3. Tracks cumulative tokens and focused sender position
            # 4. Returns empty if focused sender is beyond token limit
            # 5. Otherwise returns messages up to token limit + first focused sender message

            # Find the minimum message_id with an unprocessed queue item across the session
            min_unprocessed_message_id_subq = (
                select(func.min(models.Message.id))
                .select_from(models.QueueItem)
                .join(
                    models.Message,
                    func.cast(models.QueueItem.payload["message_id"].astext, BigInteger)
                    == models.Message.id,
                )
                .where(~models.QueueItem.processed)
                .where(models.Message.session_name == parsed_key.session_name)
                .where(models.Message.workspace_name == parsed_key.workspace_name)
                .where(models.QueueItem.work_unit_key == work_unit_key)
                .scalar_subquery()
            )

            # Build CTE with ALL messages starting from the earliest unprocessed message
            # This includes interleaving messages for conversational context
            cte = (
                select(
                    models.Message.id.label("message_id"),
                    models.Message.token_count.label("token_count"),
                    models.Message.peer_name.label("peer_name"),
                    func.sum(models.Message.token_count)
                    .over(order_by=models.Message.id)
                    .label("cumulative_token_count"),
                )
                .where(models.Message.session_name == parsed_key.session_name)
                .where(models.Message.workspace_name == parsed_key.workspace_name)
                .where(models.Message.id >= min_unprocessed_message_id_subq)
                .order_by(models.Message.id)
                .cte()
            )

            allowed_condition = (
                (
                    cte.c.cumulative_token_count
                    <= settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
                )
                | (
                    cte.c.message_id == min_unprocessed_message_id_subq
                )  # always include the first unprocessed message
            )

            query = (
                select(models.Message, models.QueueItem)
                .select_from(cte)
                .join(models.Message, models.Message.id == cte.c.message_id)
                .outerjoin(
                    models.QueueItem,
                    and_(
                        models.QueueItem.work_unit_key == work_unit_key,
                        ~models.QueueItem.processed,
                        func.cast(
                            models.QueueItem.payload["message_id"].astext, BigInteger
                        )
                        == models.Message.id,
                    ),
                )
                .where(allowed_condition)
                .order_by(models.Message.id, models.QueueItem.id)
            )

            result = await db.execute(query)
            rows = result.all()
            if not rows:
                await db.commit()
                return [], []

            messages_context: list[models.Message] = []
            items_to_process: list[QueueItem] = []
            seen_messages: set[int] = set()
            for m, qi in rows:
                if m.id not in seen_messages:
                    messages_context.append(m)
                    seen_messages.add(m.id)
                if qi is not None:
                    items_to_process.append(qi)

            if items_to_process:
                max_queue_item_message_id = max(
                    [qi.payload["message_id"] for qi in items_to_process]
                )
                messages_context = [  # remove any messages that are after the last message_id from queue items
                    m for m in messages_context if m.id <= max_queue_item_message_id
                ]

            await db.commit()

            return messages_context, items_to_process

    async def mark_queue_items_as_processed(
        self, items: list[QueueItem], work_unit_key: str
    ) -> None:
        if not items:
            return
        async with tracked_db("process_queue_item_batch") as db:
            work_unit = parse_work_unit_key(work_unit_key)
            item_ids = [item.id for item in items]
            await db.execute(
                update(models.QueueItem)
                .where(models.QueueItem.id.in_(item_ids))
                .where(models.QueueItem.work_unit_key == work_unit_key)
                .values(processed=True)
            )
            await db.execute(
                update(models.ActiveQueueSession)
                .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
                .values(last_updated=func.now())
            )
            await db.commit()

            if work_unit.task_type in ["representation", "summary"]:
                prometheus.DERIVER_QUEUE_ITEMS_PROCESSED.labels(
                    workspace_name=work_unit.workspace_name,
                    task_type=work_unit.task_type,
                ).inc(len(items))

    async def mark_queue_item_as_errored(
        self, item: QueueItem, work_unit_key: str, error: str
    ) -> None:
        """Mark queue item as processed with an error"""
        if not item:
            return
        async with tracked_db("mark_queue_item_as_errored") as db:
            await db.execute(
                update(models.QueueItem)
                .where(models.QueueItem.id == item.id)
                .where(models.QueueItem.work_unit_key == work_unit_key)
                .values(processed=True, error=error[:65535])  # Truncate to TEXT limit
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
