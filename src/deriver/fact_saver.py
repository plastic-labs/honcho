import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from .. import crud, schemas

logger = logging.getLogger(__name__)


@dataclass
class FactSaveTask:
    """Represents a single fact to be saved to the database."""
    content: str
    app_id: str
    user_id: str
    collection_id: str
    metadata: dict
    duplicate_threshold: float
    task_id: str  # For tracking/debugging


class FactSaverQueue:
    """
    Async queue-based fact saver that serializes database writes to prevent conflicts.
    Maintains a single worker that processes fact save operations sequentially.
    """
    
    def __init__(self):
        self.queue: asyncio.Queue[FactSaveTask] = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.is_running = False
        self._stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_failed': 0
        }
    
    async def start_worker(self, db_session: AsyncSession):
        """Start the background worker that processes the queue."""
        if self.is_running:
            logger.warning("FactSaverQueue worker already running")
            return
            
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker(db_session))
        logger.info("FactSaverQueue worker started")
    
    async def stop_worker(self):
        """Stop the background worker and wait for pending tasks to complete."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Wait for queue to be empty
        await self.queue.join()
        
        # Cancel the worker task
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
                
        logger.info(f"FactSaverQueue worker stopped. Stats: {self._stats}")
    
    async def queue_fact(
        self,
        content: str,
        app_id: str,
        user_id: str,
        collection_id: str,
        metadata: dict,
        duplicate_threshold: float,
        task_id: str
    ):
        """Queue a fact for saving to the database."""
        task = FactSaveTask(
            content=content,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            metadata=metadata,
            duplicate_threshold=duplicate_threshold,
            task_id=task_id
        )
        
        await self.queue.put(task)
        self._stats['total_queued'] += 1
        logger.debug(f"Queued fact for saving: {task_id} - {content[:50]}...")
    
    async def _worker(self, db_session: AsyncSession):
        """Background worker that processes queued fact save operations."""
        logger.info("FactSaverQueue worker started processing")
        
        while self.is_running:
            try:
                # Wait for a task with timeout to allow periodic checks
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Process the task
                await self._save_single_fact(db_session, task)
                
                # Mark task as done
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                # No task available, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in FactSaverQueue worker: {e}")
                self.queue.task_done()
                self._stats['total_failed'] += 1
                continue
        
        logger.info("FactSaverQueue worker stopped processing")
    
    async def _save_single_fact(self, db_session: AsyncSession, task: FactSaveTask):
        """Save a single fact to the database."""
        try:
            await crud.create_document(
                db_session,
                document=schemas.DocumentCreate(content=task.content, metadata=task.metadata),
                app_id=task.app_id,
                user_id=task.user_id,
                collection_id=task.collection_id,
                duplicate_threshold=task.duplicate_threshold,
            )
            
            self._stats['total_processed'] += 1
            logger.debug(f"Successfully saved fact: {task.task_id} - {task.content[:50]}...")
            
        except Exception as e:
            self._stats['total_failed'] += 1
            logger.error(f"Failed to save fact {task.task_id}: {e}")
            logger.debug(f"Failed fact content: {task.content}")
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            **self._stats,
            'queue_size': self.queue.qsize(),
            'is_running': self.is_running
        }


# Global instance
_fact_saver_queue: Optional[FactSaverQueue] = None


def get_fact_saver_queue() -> FactSaverQueue:
    """Get the global FactSaverQueue instance."""
    global _fact_saver_queue
    if _fact_saver_queue is None:
        _fact_saver_queue = FactSaverQueue()
    return _fact_saver_queue


async def initialize_fact_saver(db_session: AsyncSession):
    """Initialize the global fact saver queue."""
    queue = get_fact_saver_queue()
    await queue.start_worker(db_session)


async def shutdown_fact_saver():
    """Shutdown the global fact saver queue."""
    global _fact_saver_queue
    if _fact_saver_queue:
        await _fact_saver_queue.stop_worker()
        _fact_saver_queue = None