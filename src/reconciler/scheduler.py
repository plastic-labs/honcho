"""
Reconciler scheduler for self-healing background tasks.

This module provides a scheduler for running reconciliation and cleanup tasks
like vector store sync and soft-delete cleanup. It ensures only one task of each
type runs at a time across multiple deriver instances by using the queue table
for coordination.
"""

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta, timezone

import sentry_sdk
from pydantic import BaseModel
from sqlalchemy import exists, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.models import QueueItem

logger = logging.getLogger(__name__)

# System workspace used for global reconciler tasks
SYSTEM_WORKSPACE_NAME = "__system__"


class ReconcilerTask(BaseModel):
    """Definition of a reconciler task."""

    name: str
    work_unit_key: str
    interval_seconds: int


# Task intervals
QUEUE_CLEANUP_INTERVAL_SECONDS = 12 * 3600  # 12 hours

# Task registry - add new tasks here
RECONCILER_TASKS: dict[str, ReconcilerTask] = {
    "sync_vectors": ReconcilerTask(
        name="sync_vectors",
        work_unit_key="global:sync_vectors",
        interval_seconds=settings.VECTOR_STORE.RECONCILIATION_INTERVAL_SECONDS,
    ),
    "cleanup_queue": ReconcilerTask(
        name="cleanup_queue",
        work_unit_key="global:cleanup_queue",
        interval_seconds=QUEUE_CLEANUP_INTERVAL_SECONDS,
    ),
}


_reconciler_scheduler: "ReconcilerScheduler | None" = None


def set_reconciler_scheduler(scheduler: "ReconcilerScheduler") -> None:
    """Set the global reconciler scheduler reference."""
    global _reconciler_scheduler
    _reconciler_scheduler = scheduler


def get_reconciler_scheduler() -> "ReconcilerScheduler | None":
    """Get the global reconciler scheduler reference."""
    return _reconciler_scheduler


class ReconcilerScheduler:
    """
    Scheduler for self-healing reconciliation and cleanup tasks.

    Ensures only one task of each type runs at a time across multiple deriver
    instances by using the queue table for coordination. This provides:
    - Vector store synchronization (syncing pending documents/embeddings)
    - Soft-delete cleanup (removing soft-deleted documents from vector stores)
    - Self-healing behavior (retrying failed syncs)
    - Extensible task registry for adding new maintenance tasks

    Each task type has its own interval and work_unit_key, allowing multiple
    different tasks to be queued simultaneously while preventing duplicates
    of the same task type.
    """

    _instance: "ReconcilerScheduler | None" = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not ReconcilerScheduler._initialized:
            self._scheduler_task: asyncio.Task[None] | None = None
            self._shutdown_event: asyncio.Event = asyncio.Event()
            # Track next run time for each task
            self._next_run: dict[str, datetime] = {}
            ReconcilerScheduler._initialized = True

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance. Only use this in tests."""
        cls._instance = None
        cls._initialized = False

    async def start(self) -> None:
        """Start the reconciler scheduler loop."""
        if self._scheduler_task is not None:
            logger.warning("ReconcilerScheduler already running")
            return

        self._shutdown_event.clear()
        # Initialize next run times to first interval
        now = datetime.now(timezone.utc)
        for task_name, task in RECONCILER_TASKS.items():
            self._next_run[task_name] = now + timedelta(seconds=task.interval_seconds)

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            "ReconcilerScheduler started with %d tasks: %s",
            len(RECONCILER_TASKS),
            list(RECONCILER_TASKS.keys()),
        )

    async def shutdown(self) -> None:
        """Stop the reconciler scheduler."""
        if self._scheduler_task is None:
            return

        logger.info("Shutting down ReconcilerScheduler...")
        self._shutdown_event.set()

        try:
            await asyncio.wait_for(self._scheduler_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("ReconcilerScheduler shutdown timed out, cancelling task")
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task

        self._scheduler_task = None
        self._next_run.clear()
        logger.info("ReconcilerScheduler stopped")

    async def _scheduler_loop(self) -> None:
        """
        Main scheduler loop that enqueues tasks based on their intervals.

        Each task has its own interval and the loop checks all tasks on each
        iteration, enqueueing any that are due.
        """
        try:
            while not self._shutdown_event.is_set():
                now = datetime.now(timezone.utc)

                # Check each task and enqueue if due
                for task_name, task in RECONCILER_TASKS.items():
                    next_run = self._next_run.get(task_name, now)
                    if now >= next_run:
                        try:
                            enqueued = await self._try_enqueue_task(task)
                            if enqueued:
                                logger.debug("Enqueued task: %s", task_name)
                        except Exception as e:
                            logger.exception("Error enqueueing task %s", task_name)
                            if settings.SENTRY.ENABLED:
                                sentry_sdk.capture_exception(e)

                        # Schedule next run regardless of whether we enqueued
                        # (if already pending/in-progress, we'll skip next time too)
                        self._next_run[task_name] = now + timedelta(
                            seconds=task.interval_seconds
                        )

                # Calculate sleep time until next task is due
                if self._next_run:
                    next_task_time = min(self._next_run.values())
                    sleep_seconds = max(
                        1.0,  # At least 1 second to avoid busy loop
                        (next_task_time - datetime.now(timezone.utc)).total_seconds(),
                    )
                else:
                    sleep_seconds = 60.0  # Default if no tasks

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_seconds,
                    )
                    break  # Shutdown event was set
                except asyncio.TimeoutError:
                    # Timeout means interval elapsed, continue loop
                    pass

        except asyncio.CancelledError:
            logger.debug("ReconcilerScheduler loop cancelled")
            raise

    async def _try_enqueue_task(self, task: ReconcilerTask) -> bool:
        """
        Attempt to enqueue a reconciler task.

        This is idempotent - if the task is already in-progress (has an
        ActiveQueueSession) or pending in the queue, the enqueue is skipped.

        Args:
            task: The task definition to enqueue

        Returns:
            True if a task was enqueued, False if skipped
        """
        async with tracked_db("reconciler_enqueue") as db:
            # Ensure the system workspace exists (for FK constraint)
            await self._ensure_system_workspace(db)

            # Check if task is already in progress
            in_progress_check = select(
                exists(
                    select(models.ActiveQueueSession.id).where(
                        models.ActiveQueueSession.work_unit_key == task.work_unit_key
                    )
                )
            )
            is_in_progress = await db.scalar(in_progress_check)

            if is_in_progress:
                logger.debug("Task %s already in progress, skipping enqueue", task.name)
                return False

            # Check if there's already a pending task
            pending_check = select(
                exists(
                    select(QueueItem.id).where(
                        QueueItem.work_unit_key == task.work_unit_key,
                        QueueItem.processed == False,  # noqa: E712
                    )
                )
            )
            is_pending = await db.scalar(pending_check)

            if is_pending:
                logger.debug(
                    "Task %s already pending in queue, skipping enqueue", task.name
                )
                return False

            # Enqueue the task using ORM
            queue_item = QueueItem(
                work_unit_key=task.work_unit_key,
                payload={
                    "reconciler_type": task.name,
                },
                session_id=None,
                task_type="reconciler",
                workspace_name=SYSTEM_WORKSPACE_NAME,
                message_id=None,
            )
            db.add(queue_item)
            try:
                await db.commit()
            except IntegrityError:
                # Another instance already enqueued this task (unique constraint)
                await db.rollback()
                logger.debug(
                    "Task %s already enqueued by another instance, skipping", task.name
                )
                return False

            logger.info("Enqueued reconciler task: %s", task.name)
            return True

    async def _ensure_system_workspace(self, db: AsyncSession) -> None:
        """Ensure the system workspace exists for reconciler tasks."""
        # Use upsert to create workspace if it doesn't exist
        stmt = pg_insert(models.Workspace).values(name=SYSTEM_WORKSPACE_NAME)
        stmt = stmt.on_conflict_do_nothing(index_elements=["name"])
        await db.execute(stmt)
