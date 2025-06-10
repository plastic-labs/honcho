"""
Asynchronous observation saving system using in-memory queue and background workers.

Handles high-throughput observation saving with duplicate detection and batch processing.
Uses embedding similarity to detect duplicates before insertion.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from src import crud, schemas
from src.dependencies import tracked_db

logger = logging.getLogger(__name__)


@dataclass
class ObservationTask:
    """Represents a single observation to be saved."""

    task_id: str
    content: str
    app_id: str
    user_id: str
    collection_id: str
    metadata: dict
    duplicate_threshold: float
    created_at: float  # timestamp

    def is_expired(self, timeout_seconds: float = 300.0) -> bool:
        """Check if task has expired (default 5 minutes)."""
        return time.time() - self.created_at > timeout_seconds


class ObservationSaverQueue:
    """Asynchronous queue for saving observations with duplicate detection."""

    def __init__(self, max_workers: int = 2, batch_size: int = 5):
        self.queue: asyncio.Queue[ObservationTask] = asyncio.Queue()
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.workers: list[asyncio.Task] = []
        self.is_running = False
        self.stats = {
            "observations_queued": 0,
            "observations_saved": 0,
            "observations_duplicate": 0,
            "observations_error": 0,
            "batches_processed": 0,
            "workers_active": 0,
        }
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the background workers."""
        if self.is_running:
            return

        self.is_running = True
        logger.info(
            f"Starting observation saver with {self.max_workers} workers, batch size {self.batch_size}"
        )

        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)

        logger.info("Observation saver started successfully")

    async def stop(self):
        """Stop all workers and wait for completion."""
        if not self.is_running:
            return

        logger.info("Stopping observation saver...")
        self.is_running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("Observation saver stopped")

    async def queue_observation(
        self,
        content: str,
        app_id: str,
        user_id: str,
        collection_id: str,
        metadata: dict,
        duplicate_threshold: float = 0.1,
        task_id: str | None = None,
    ) -> str:
        """Queue an observation for asynchronous saving."""
        if not self.is_running:
            raise RuntimeError("ObservationSaverQueue is not running")

        task_id = task_id or str(uuid.uuid4())

        task = ObservationTask(
            task_id=task_id,
            content=content,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            metadata=metadata,
            duplicate_threshold=duplicate_threshold,
            created_at=time.time(),
        )

        await self.queue.put(task)

        async with self._lock:
            self.stats["observations_queued"] += 1

        logger.debug(f"Queued observation: {content[:50]}... (task_id: {task_id})")
        return task_id

    async def _worker(self, worker_name: str):
        """Background worker that processes observation tasks."""
        logger.debug(f"Worker {worker_name} started")

        async with self._lock:
            self.stats["workers_active"] += 1

        try:
            while self.is_running:
                try:
                    # Collect batch of tasks
                    batch: list[ObservationTask] = []

                    # Get first task (blocking)
                    try:
                        task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                        batch.append(task)
                    except asyncio.TimeoutError:
                        continue  # No tasks available, continue loop

                    # Collect additional tasks for batch (non-blocking)
                    while len(batch) < self.batch_size:
                        try:
                            task = self.queue.get_nowait()
                            batch.append(task)
                        except asyncio.QueueEmpty:
                            break

                    # Process the batch
                    if batch:
                        await self._process_batch(batch, worker_name)

                except Exception as e:
                    logger.error(f"Worker {worker_name} error: {e}")
                    await asyncio.sleep(1)  # Brief pause before retrying

        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_name} cancelled")
        finally:
            async with self._lock:
                self.stats["workers_active"] -= 1
            logger.debug(f"Worker {worker_name} stopped")

    async def _process_batch(self, batch: list[ObservationTask], worker_name: str):
        """Process a batch of observation tasks."""
        logger.debug(
            f"Worker {worker_name} processing batch of {len(batch)} observations"
        )

        async with tracked_db("observation_saver.process_batch") as db:
            for task in batch:
                try:
                    # Check if task has expired
                    if task.is_expired():
                        logger.warning(f"Task {task.task_id} expired, skipping")
                        async with self._lock:
                            self.stats["observations_error"] += 1
                        continue

                    # Check for duplicates
                    duplicates = await crud.get_duplicate_documents(
                        db,
                        app_id=task.app_id,
                        user_id=task.user_id,
                        collection_id=task.collection_id,
                        content=task.content,
                        similarity_threshold=1
                        - task.duplicate_threshold,  # Convert to similarity
                    )

                    if duplicates:
                        logger.debug(
                            f"Duplicate observation found for task {task.task_id}, skipping"
                        )
                        async with self._lock:
                            self.stats["observations_duplicate"] += 1
                        continue

                    # Create new observation
                    document = schemas.DocumentCreate(
                        content=task.content, metadata=task.metadata
                    )

                    await crud.create_document(
                        db,
                        document=document,
                        app_id=task.app_id,
                        user_id=task.user_id,
                        collection_id=task.collection_id,
                        duplicate_threshold=task.duplicate_threshold,
                    )

                    logger.debug(
                        f"Saved observation: {task.content[:50]}... (task_id: {task.task_id})"
                    )
                    async with self._lock:
                        self.stats["observations_saved"] += 1

                except Exception as e:
                    logger.error(
                        f"Error processing observation task {task.task_id}: {e}"
                    )
                    async with self._lock:
                        self.stats["observations_error"] += 1

        async with self._lock:
            self.stats["batches_processed"] += 1

        logger.debug(f"Worker {worker_name} completed batch")

    def get_stats(self) -> dict:
        """Get current statistics."""
        return self.stats.copy()

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()


# Backward compatibility alias
FactSaverQueue = ObservationSaverQueue

# Global instance
_observation_saver_queue: ObservationSaverQueue | None = None


def get_observation_saver_queue() -> ObservationSaverQueue:
    """Get the global observation saver queue instance."""
    global _observation_saver_queue
    if _observation_saver_queue is None:
        _observation_saver_queue = ObservationSaverQueue()
    return _observation_saver_queue


async def initialize_observation_saver(db=None):
    """Initialize the global observation saver queue."""
    queue = get_observation_saver_queue()
    if not queue.is_running:
        await queue.start()
        logger.info("Global observation saver queue initialized")


async def shutdown_observation_saver():
    """Shutdown the global observation saver queue."""
    global _observation_saver_queue
    if _observation_saver_queue and _observation_saver_queue.is_running:
        await _observation_saver_queue.stop()
        logger.info("Global observation saver queue shutdown")
