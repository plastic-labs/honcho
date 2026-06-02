import asyncio
import contextlib
import random
import signal
from asyncio import Task
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from logging import getLogger
from typing import Any, NamedTuple, cast

import sentry_sdk
from dotenv import load_dotenv
from nanoid import generate as generate_nanoid
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src import models
from src.cache.client import close_cache, init_cache
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
from src.reconciler import (
    ReconcilerScheduler,
    get_reconciler_scheduler,
    set_reconciler_scheduler,
)
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.sentry import initialize_sentry
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


@dataclass(frozen=True)
class QueueBatchResult:
    """Result of `QueueManager.get_queue_item_batch`.

    telemetry needs to know two things in addition to the batch
    contents: whether the cumulative-token cap clamped the batch, and what
    the configured cap was. These flags feed `RepresentationCompletedEvent`
    so analytics can detect "we under-batched because of a flush" vs
    "we hit the cap and kept going".
    """

    messages_context: list[models.Message] = field(default_factory=list)
    items_to_process: list["QueueItem"] = field(default_factory=list)
    configuration: ResolvedConfiguration | None = None
    hit_batch_token_cap: bool = False
    was_flush_enabled: bool = False
    batch_max_tokens: int = 0


def _detach_queue_batch_objects(
    db: AsyncSession,
    messages_context: list[models.Message],
    items_to_process: list[QueueItem],
) -> None:
    """Detach loaded batch objects so they remain usable after tracked_db exits."""
    seen: set[int] = set()
    for obj in [*messages_context, *items_to_process]:
        obj_id = id(obj)
        if obj_id in seen:
            continue
        db.expunge(obj)
        seen.add(obj_id)


def _resolve_batch_configuration(
    items_to_process: list[QueueItem],
) -> tuple[list[QueueItem], ResolvedConfiguration | None]:
    """Keep only the initial homogeneous configuration prefix for a batch."""
    if not items_to_process:
        return [], None

    raw_config = items_to_process[0].payload.get("configuration")
    resolved_config = (
        None if raw_config is None else ResolvedConfiguration.model_validate(raw_config)
    )

    valid_items: list[QueueItem] = []
    for item in items_to_process:
        item_raw_config = item.payload.get("configuration")
        item_config = (
            None
            if item_raw_config is None
            else ResolvedConfiguration.model_validate(item_raw_config)
        )
        if item_config != resolved_config:
            break
        valid_items.append(item)

    return valid_items, resolved_config


class QueueManager:
    def __init__(self):
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.active_tasks: set[asyncio.Task[None]] = set()
        self.worker_ownership: dict[str, WorkerOwnership] = {}
        self.queue_empty_flag: asyncio.Event = asyncio.Event()

        # Current adaptive polling interval; grows while idle/erroring and
        # resets to the base interval as soon as work is claimed.
        self._current_poll_interval: float = (
            settings.DERIVER.POLLING_SLEEP_INTERVAL_SECONDS
        )

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

        # Get or create the singleton reconciler scheduler
        existing_reconciler = get_reconciler_scheduler()
        if existing_reconciler is None:
            self.reconciler_scheduler: ReconcilerScheduler = ReconcilerScheduler()
            set_reconciler_scheduler(self.reconciler_scheduler)
        else:
            self.reconciler_scheduler = existing_reconciler

        # Initialize Sentry if enabled, using settings
        if settings.SENTRY.ENABLED:
            initialize_sentry(
                integrations=[AsyncioIntegration(), SqlalchemyIntegration()]
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

        # Start the reconciler scheduler
        try:
            await self.reconciler_scheduler.start()
        except Exception:
            logger.exception("Failed to start reconciler scheduler")

        # Run the polling loop directly in this task
        logger.debug("Starting polling loop directly")
        try:
            await self._sleep_startup_jitter()
            await self.polling_loop()
        finally:
            await self.cleanup()

    async def shutdown(self, sig: signal.Signals) -> None:
        """Handle graceful shutdown"""
        logger.info(f"Received exit signal {sig.name}...")
        self.shutdown_event.set()

        # Cancel all pending dreams
        await self.dream_scheduler.shutdown()

        # Stop the reconciler scheduler
        await self.reconciler_scheduler.shutdown()

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
        For representation tasks, only returns work units with accumulated tokens
        >= REPRESENTATION_BATCH_MAX_TOKENS (forced batching), unless FLUSH_ENABLED is True.
        Returns a dict mapping work_unit_key to aqs_id.
        """
        limit: int = max(0, self.workers - self.get_total_owned_work_units())
        if limit == 0:
            return {}

        batch_max_tokens = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS

        async with tracked_db("get_available_work_units") as db:
            representation_prefix = "representation:"
            token_stats_subq = (
                select(
                    models.QueueItem.work_unit_key,
                    func.sum(models.Message.token_count).label("total_tokens"),
                )
                .join(
                    models.Message,
                    models.QueueItem.message_id == models.Message.id,
                )
                .where(~models.QueueItem.processed)
                .where(models.QueueItem.work_unit_key.startswith(representation_prefix))
                .group_by(models.QueueItem.work_unit_key)
                .subquery()
            )

            work_units_subq = (
                select(models.QueueItem.work_unit_key)
                .where(~models.QueueItem.processed)
                .group_by(models.QueueItem.work_unit_key)
                .subquery()
            )

            query = (
                select(work_units_subq.c.work_unit_key)
                .limit(limit)
                .outerjoin(
                    token_stats_subq,
                    work_units_subq.c.work_unit_key == token_stats_subq.c.work_unit_key,
                )
                .where(
                    ~select(models.ActiveQueueSession.id)
                    .where(
                        models.ActiveQueueSession.work_unit_key
                        == work_units_subq.c.work_unit_key
                    )
                    .exists()
                )
            )

            # Apply batch threshold filter (skip if FLUSH_ENABLED is True)
            if not settings.DERIVER.FLUSH_ENABLED and batch_max_tokens > 0:
                query = query.where(
                    or_(
                        ~work_units_subq.c.work_unit_key.startswith(
                            representation_prefix
                        ),
                        func.coalesce(token_stats_subq.c.total_tokens, 0)
                        >= batch_max_tokens,
                    )
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

    def _reset_poll_interval(self) -> None:
        """Snap the polling interval back to the base after finding work."""
        self._current_poll_interval = settings.DERIVER.POLLING_SLEEP_INTERVAL_SECONDS

    def _jitter(self, seconds: float) -> float:
        """Scatter a sleep by +/- POLLING_JITTER_RATIO to avoid lockstep polling.

        Returns a uniform-random value in [(1-ratio)*seconds, (1+ratio)*seconds].
        Only the returned sleep is scattered; the underlying backoff schedule is
        left unchanged. A ratio of 0.0 returns ``seconds`` unchanged.
        """
        ratio = settings.DERIVER.POLLING_JITTER_RATIO
        if ratio <= 0.0:
            return seconds
        # Scheduling jitter, not security/crypto — stdlib random is appropriate.
        return seconds * random.uniform(1.0 - ratio, 1.0 + ratio)  # nosec B311

    async def _sleep_startup_jitter(self) -> None:
        """Sleep a random delay before the first poll so instances that start
        together don't poll in lockstep. Interruptible by shutdown so a signal
        during the delay exits promptly. No-op when the window is 0.0.
        """
        window = settings.DERIVER.POLLING_STARTUP_JITTER_SECONDS
        if window <= 0.0:
            return
        # Scheduling jitter, not security/crypto — stdlib random is appropriate.
        delay = random.uniform(0.0, window)  # nosec B311
        logger.debug(f"Startup poll jitter: sleeping {delay:.1f}s before first poll")
        # Timeout (slept the full delay without a shutdown) is the normal path;
        # an early return means shutdown fired and polling_loop will exit at once.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=delay)

    def _advance_poll_interval(self) -> float:
        """Return the current idle/backoff sleep, then grow it toward the cap."""
        interval = self._current_poll_interval
        if settings.DERIVER.POLLING_BACKOFF_ENABLED:
            self._current_poll_interval = min(
                self._current_poll_interval
                * settings.DERIVER.POLLING_BACKOFF_MULTIPLIER,
                settings.DERIVER.POLLING_SLEEP_MAX_INTERVAL_SECONDS,
            )
        return self._jitter(interval)

    async def polling_loop(self) -> None:
        """Main polling loop to find and process new work units"""
        logger.debug("Starting polling loop")
        try:
            while not self.shutdown_event.is_set():
                if self.queue_empty_flag.is_set():
                    # The empty-poll branch below already slept this cycle's
                    # interval; just clear the flag and re-query (no second
                    # sleep — that would double the effective idle interval).
                    self.queue_empty_flag.clear()
                    continue

                # Check if we have capacity before querying. There is work to do
                # (workers are busy), so keep the base interval for fast pickup
                # when capacity frees rather than backing off.
                if self.semaphore.locked():
                    # logger.debug("All workers busy, waiting")
                    await asyncio.sleep(
                        self._jitter(settings.DERIVER.POLLING_SLEEP_INTERVAL_SECONDS)
                    )
                    continue

                try:
                    await self.cleanup_stale_work_units()
                    claimed_work_units = await self.get_and_claim_work_units()
                    if claimed_work_units:
                        self._reset_poll_interval()
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
                        await asyncio.sleep(self._advance_poll_interval())
                except Exception as e:
                    logger.exception("Error in polling loop")
                    if settings.SENTRY.ENABLED:
                        sentry_sdk.capture_exception(e)
                    # Note: rollback is handled by tracked_db dependency.
                    # Back off so a down/saturated DB isn't hammered every cycle.
                    await asyncio.sleep(self._advance_poll_interval())
        finally:
            logger.info("Polling loop stopped")

    ######################
    # Queue Worker Logic #
    ######################

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
                            batch_result = await self.get_queue_item_batch(
                                work_unit.task_type, work_unit_key, ownership.aqs_id
                            )
                            messages_context = batch_result.messages_context
                            items_to_process = batch_result.items_to_process
                            message_level_configuration = batch_result.configuration
                            logger.debug(
                                f"Worker {worker_id} retrieved {len(messages_context)} messages and {len(items_to_process)} queue items for work unit {work_unit_key} (AQS ID: {ownership.aqs_id})"
                            )
                            if not items_to_process:
                                logger.debug(
                                    f"No more queue items to process for work unit {work_unit_key} for worker {worker_id}"
                                )
                                break

                            try:
                                # Extract observers from the payload (handle both old and new format)
                                payload = items_to_process[0].payload
                                observers = payload.get("observers")
                                if observers is None:
                                    # Legacy format: single observer string
                                    legacy_observer = payload.get("observer")
                                    if legacy_observer:
                                        observers = [legacy_observer]
                                    else:
                                        observers = []

                                queue_item_message_ids = [
                                    item.message_id
                                    for item in items_to_process
                                    if item.message_id is not None
                                ]
                                await process_representation_batch(
                                    messages_context,
                                    message_level_configuration,
                                    observers=observers,
                                    observed=work_unit.observed,
                                    queue_item_message_ids=queue_item_message_ids,
                                    hit_batch_token_cap=batch_result.hit_batch_token_cap,
                                    was_flush_enabled=batch_result.was_flush_enabled,
                                    batch_max_tokens=batch_result.batch_max_tokens,
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
                                    f"processing {work_unit.task_type} batch",
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
                                await process_item(queue_item)
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
                        if (
                            work_unit.task_type in ["representation", "summary"]
                            and work_unit.workspace_name is not None
                        ):
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
                "representation tasks are not supported for get_next_queue_item"
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
    ) -> "QueueBatchResult":
        """
        Batch processing for representation and agent tasks.

        Returns a `QueueBatchResult` carrying:
        - messages_context: unique Message rows (conversation turns) forming the context window
        - items_to_process: QueueItems for the current work_unit_key within that window
        - configuration: Resolved configuration for the batch
        - hit_batch_token_cap: True when the cumulative-token window clamped the batch
        - was_flush_enabled: snapshot of `settings.DERIVER.FLUSH_ENABLED` at fetch time
        - batch_max_tokens: snapshot of the cap actually applied to this batch
        """
        if task_type != "representation":
            raise ValueError(
                f"{task_type} tasks are not supported for get_queue_item_batch"
            )

        batch_max_tokens = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
        was_flush_enabled = settings.DERIVER.FLUSH_ENABLED
        parsed_key = parse_work_unit_key(work_unit_key)
        messages_context: list[models.Message] = []
        items_to_process: list[QueueItem] = []

        async with tracked_db("get_queue_item_batch") as db:
            # For batch tasks, get messages based on token limit.
            # Step 1: Verify worker still owns the work_unit_key.
            ownership_check = await db.execute(
                select(models.ActiveQueueSession.id)
                .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
                .where(models.ActiveQueueSession.id == aqs_id)
            )
            if not ownership_check.scalar_one_or_none():
                return QueueBatchResult(
                    was_flush_enabled=was_flush_enabled,
                    batch_max_tokens=batch_max_tokens,
                )

            # Step 2: Build a single SQL query that:
            # 1. Finds the earliest unprocessed message for this work_unit_key
            # 2. Optionally includes the preceding message if from a different peer (for context)
            # 3. Gets ALL messages from that point forward (for conversational context)
            # 4. Tracks cumulative tokens and focused sender position
            # 5. Returns empty if focused sender is beyond token limit
            # 6. Otherwise returns messages up to token limit + first focused sender message

            # Find the minimum message_id with an unprocessed queue item across the session
            min_unprocessed_message_id_subq = (
                select(func.min(models.Message.id))
                .select_from(models.QueueItem)
                .join(
                    models.Message,
                    models.QueueItem.message_id == models.Message.id,
                )
                .where(~models.QueueItem.processed)
                .where(models.Message.session_name == parsed_key.session_name)
                .where(models.Message.workspace_name == parsed_key.workspace_name)
                .where(models.QueueItem.work_unit_key == work_unit_key)
                .scalar_subquery()
            )

            # Find the immediately preceding message ID (the one right before min_unprocessed)
            immediately_preceding_id_subq = (
                select(func.max(models.Message.id))
                .where(models.Message.session_name == parsed_key.session_name)
                .where(models.Message.workspace_name == parsed_key.workspace_name)
                .where(models.Message.id < min_unprocessed_message_id_subq)
                .scalar_subquery()
            )

            # Only include the preceding message if it's from a different peer than observed
            # This provides conversational context (e.g., the question that prompted the response)
            preceding_message_id_subq = (
                select(models.Message.id)
                .where(models.Message.id == immediately_preceding_id_subq)
                .where(models.Message.peer_name != parsed_key.observed)
                .scalar_subquery()
            )

            # Determine the effective start: preceding message if it qualifies, else min_unprocessed
            # We use COALESCE to fall back to min_unprocessed if no preceding message qualifies
            effective_start_id = func.coalesce(
                preceding_message_id_subq, min_unprocessed_message_id_subq
            )

            # Build CTE in two nested selects so we can layer a second window
            # function on top of `cumulative_token_count`. Postgres doesn't
            # allow nesting window functions in a single select; we compute
            # `cumulative_token_count` in `inner_cte`, then `cap_exceeded` as
            # `bool_or(cumulative > cap) OVER ()` in the outer CTE. The flag
            # is identical across every row, so reading it from any returned
            # row tells us whether the SQL cap would have excluded messages —
            # eliminating the separate `SELECT EXISTS` roundtrip that used to
            # run post-fetch.
            inner_cte = (
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
                .where(models.Message.id >= effective_start_id)
                .subquery()
            )

            cte = (
                select(
                    inner_cte.c.message_id,
                    inner_cte.c.token_count,
                    inner_cte.c.peer_name,
                    inner_cte.c.cumulative_token_count,
                    func.bool_or(inner_cte.c.cumulative_token_count > batch_max_tokens)
                    .over()
                    .label("cap_exceeded"),
                )
                .order_by(inner_cte.c.message_id)
                .cte()
            )

            allowed_condition = (
                (cte.c.cumulative_token_count <= batch_max_tokens)
                | (
                    cte.c.message_id == min_unprocessed_message_id_subq
                )  # always include the first unprocessed message
            )

            query = (
                select(
                    models.Message,
                    models.QueueItem,
                    cte.c.cap_exceeded.label("cap_exceeded"),
                )
                .select_from(cte)
                .join(models.Message, models.Message.id == cte.c.message_id)
                .outerjoin(
                    models.QueueItem,
                    and_(
                        models.QueueItem.work_unit_key == work_unit_key,
                        ~models.QueueItem.processed,
                        models.QueueItem.message_id == models.Message.id,
                    ),
                )
                .where(allowed_condition)
                .order_by(models.Message.id, models.QueueItem.id)
            )

            result = await db.execute(query)
            rows = result.all()
            if not rows:
                return QueueBatchResult(
                    was_flush_enabled=was_flush_enabled,
                    batch_max_tokens=batch_max_tokens,
                )

            # cap_exceeded is window-aggregated over the CTE — same value on
            # every row. Read once from the first row; default False if the
            # cap is disabled (`batch_max_tokens == 0`).
            cap_exceeded_from_query: bool = (
                bool(rows[0][2]) if rows and batch_max_tokens > 0 else False
            )

            seen_messages: set[int] = set()
            for m, qi, _cap in rows:
                if m.id not in seen_messages:
                    messages_context.append(m)
                    seen_messages.add(m.id)
                if qi is not None:
                    items_to_process.append(qi)

            # Detach BEFORE config-filter — `_resolve_batch_configuration` is
            # sync and doesn't need the session; `messages_context` is a plain
            # Python list after detach and survives the rest of this block.
            _detach_queue_batch_objects(db, messages_context, items_to_process)

            # The QUEUE-ITEM boundary (not the messages_context tail) is
            # what matters for cap detection. messages_context includes
            # non-queue interleaving context messages — if SQL kept some
            # trailing context past the last queued item, the config
            # filter trims that context but doesn't touch the queue
            # items. Using messages_context[-1].id as a "did config
            # filter shrink the batch" signal produced false negatives
            # for that case.
            last_queued_id_before: int | None = (
                max(
                    qi.message_id
                    for qi in items_to_process
                    if qi.message_id is not None
                )
                if items_to_process
                else None
            )

            items_to_process, resolved_config = _resolve_batch_configuration(
                items_to_process
            )
            if items_to_process:
                max_queue_item_message_id = max(
                    qi.message_id
                    for qi in items_to_process
                    if qi.message_id is not None
                )
                messages_context = [
                    m for m in messages_context if m.id <= max_queue_item_message_id
                ]

            last_queued_id_after: int | None = (
                max(
                    qi.message_id
                    for qi in items_to_process
                    if qi.message_id is not None
                )
                if items_to_process
                else None
            )

            # detect if `batch_max_tokens` clamped this returned batch.
            #
            # `cap_exceeded_from_query` comes from the CTE's
            # `bool_or(cumulative > cap) OVER ()` column — true iff the
            # SQL would have excluded at least one message because of the
            # cap. Combined with the queue-boundary guard below, this
            # tells us the cap was binding on the returned batch:
            #
            #   1. Config filter didn't shrink the QUEUE-ITEM boundary
            #      (`last_queued_id_before == last_queued_id_after`) —
            #      i.e. SQL chose the trailing queue item, not config; AND
            #   2. The CTE detected at least one message past the cap.
            #
            # Both conditions must hold; otherwise the cap wasn't the
            # constraint on this specific returned batch.
            #
            # Previously we issued a separate `SELECT EXISTS` query for
            # the second condition. Folding it into the CTE eliminates the
            # roundtrip — every batch fetch is now one query, not two.
            if (
                batch_max_tokens > 0
                and last_queued_id_before is not None
                and last_queued_id_before == last_queued_id_after
            ):
                hit_batch_token_cap = cap_exceeded_from_query
            else:
                hit_batch_token_cap = False

        return QueueBatchResult(
            messages_context=messages_context,
            items_to_process=items_to_process,
            configuration=resolved_config,
            hit_batch_token_cap=hit_batch_token_cap,
            was_flush_enabled=was_flush_enabled,
            batch_max_tokens=batch_max_tokens,
        )

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

            if (
                work_unit.task_type in ["representation", "summary"]
                and work_unit.workspace_name is not None
                and settings.METRICS.ENABLED
            ):
                prometheus_metrics.record_deriver_queue_item(
                    count=len(items),
                    workspace_name=work_unit.workspace_name,
                    task_type=work_unit.task_type,
                )

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
            result = cast(
                CursorResult[Any],
                await db.execute(
                    delete(models.ActiveQueueSession)
                    .where(models.ActiveQueueSession.id == aqs_id)
                    .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
                ),
            )
            await db.commit()
            return result.rowcount > 0


async def main():
    logger.debug("Starting queue manager")

    try:
        await init_cache()
    except Exception as e:
        logger.warning(
            "Error initializing cache in queue manager; proceeding without cache: %s", e
        )

    manager = QueueManager()
    try:
        await manager.initialize()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sentry_sdk.capture_exception(e)
    finally:
        await close_cache()
        logger.debug("Main function exiting")
