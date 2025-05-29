import asyncio
import logging
import os
import signal
from datetime import datetime, timedelta
from logging import getLogger

import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from .. import models
from ..dependencies import tracked_db
from .consumer import process_item

load_dotenv()


def get_log_level(env_var="LOG_LEVEL", default="INFO"):
    """
    Convert log level string from environment variable to logging module constant.
    """
    log_level_str = os.getenv(env_var, default).upper()
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return log_levels.get(log_level_str, logging.INFO)


# Configure logging for deriver process
logging.basicConfig(
    level=get_log_level(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = getLogger(__name__)


class QueueManager:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.active_tasks: set[asyncio.Task] = set()
        self.owned_sessions: set[int] = set()
        self.queue_empty_flag = asyncio.Event()

        # Initialize from environment
        self.workers = int(os.getenv("DERIVER_WORKERS", 1))
        self.semaphore = asyncio.Semaphore(self.workers)

        # Initialize Sentry if enabled
        if os.getenv("SENTRY_ENABLED", "False").lower() == "true":
            sentry_sdk.init(
                dsn=os.getenv("SENTRY_DSN"),
                enable_tracing=True,
                traces_sample_rate=0.1,
                profiles_sample_rate=0.1,
                integrations=[AsyncioIntegration()],
            )

    def add_task(self, task: asyncio.Task):
        """Track a new task"""
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    def track_session(self, session_id: int):
        """Track a new session owned by this process"""
        self.owned_sessions.add(session_id)

    def untrack_session(self, session_id: int):
        """Remove a session from tracking"""
        self.owned_sessions.discard(session_id)

    async def initialize(self):
        """Setup signal handlers and start the main polling loop"""
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

    async def cleanup(self):
        """Clean up owned sessions"""
        if self.owned_sessions:
            logger.info(f"Cleaning up {len(self.owned_sessions)} owned sessions...")
            try:
                # Use the tracked_db dependency for transaction safety
                async with tracked_db("queue_cleanup") as db:
                    await db.execute(
                        delete(models.ActiveQueueSession).where(
                            models.ActiveQueueSession.session_id.in_(
                                self.owned_sessions
                            )
                        )
                    )
                    await db.commit()
                    logger.info("Cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                if os.getenv("SENTRY_ENABLED", "False").lower() == "true":
                    sentry_sdk.capture_exception(e)

    ##########################
    # Polling and Scheduling #
    ##########################

    async def get_available_sessions(self, db: AsyncSession):
        """Get available sessions that aren't being processed"""
        # Clean up stale sessions
        five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
        await db.execute(
            delete(models.ActiveQueueSession).where(
                models.ActiveQueueSession.last_updated < five_minutes_ago
            )
        )

        # Get available sessions
        result = await db.execute(
            select(models.QueueItem.session_id)
            .outerjoin(
                models.ActiveQueueSession,
                models.QueueItem.session_id == models.ActiveQueueSession.session_id,
            )
            .where(models.QueueItem.processed == False)
            .where(
                models.ActiveQueueSession.session_id == None
            )  # Only sessions not in active_sessions
            .group_by(models.QueueItem.session_id)
            .limit(1)
        )
        return result.scalars().all()

    async def polling_loop(self):
        """Main polling loop to find and process new sessions"""
        logger.debug("Starting polling loop")
        try:
            while not self.shutdown_event.is_set():
                if self.queue_empty_flag.is_set():
                    # logger.debug("Queue empty flag set, waiting")
                    await asyncio.sleep(1)
                    self.queue_empty_flag.clear()
                    continue

                # Check if we have capacity before querying
                if self.semaphore.locked():
                    # logger.debug("All workers busy, waiting")
                    await asyncio.sleep(1)  # Wait before trying again
                    continue

                # Use the dependency for transaction safety
                async with tracked_db("queue_polling_loop") as db:
                    try:
                        new_sessions = await self.get_available_sessions(db)

                        if new_sessions and not self.shutdown_event.is_set():
                            for session_id in new_sessions:
                                try:
                                    # Try to claim the session
                                    await db.execute(
                                        insert(models.ActiveQueueSession).values(
                                            session_id=session_id,
                                        )
                                    )
                                    await db.commit()

                                    # Track this session
                                    self.track_session(session_id)
                                    logger.debug(
                                        f"Claimed session {session_id} for processing"
                                    )

                                    # Create a new task for processing this session
                                    if not self.shutdown_event.is_set():
                                        task = asyncio.create_task(
                                            self.process_session(session_id)
                                        )
                                        self.add_task(task)
                                except IntegrityError:
                                    # Note: rollback is handled by tracked_db dependency
                                    logger.debug(
                                        f"Failed to claim session {session_id}, already owned"
                                    )
                        else:
                            self.queue_empty_flag.set()
                            await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error in polling loop: {str(e)}", exc_info=True)
                        if os.getenv("SENTRY_ENABLED", "False").lower() == "true":
                            sentry_sdk.capture_exception(e)
                        # Note: rollback is handled by tracked_db dependency
                        await asyncio.sleep(1)
        finally:
            logger.info("Polling loop stopped")

    ######################
    # Queue Worker Logic #
    ######################

    @sentry_sdk.trace
    async def process_session(self, session_id: int):
        """Process all messages for a session"""
        logger.debug(f"Starting to process session {session_id}")
        # Use the tracked_db dependency for transaction safety
        async with (
            self.semaphore,
            tracked_db("queue_process_session") as db,
        ):  # Hold the semaphore for the entire session duration
            try:
                message_count = 0
                while not self.shutdown_event.is_set():
                    message = await self.get_next_message(db, session_id)
                    if not message:
                        logger.debug(f"No more messages for session {session_id}")
                        break

                    message_count += 1
                    logger.debug(
                        f"Processing message {message.id} for session {session_id} (message {message_count})"
                    )
                    try:
                        logger.info(
                            f"Processing message {message.id} from session {session_id}"
                        )
                        await process_item(db, payload=message.payload)
                        logger.debug(f"Successfully processed message {message.id}")
                    except Exception as e:
                        logger.error(
                            f"Error processing message {message.id}: {str(e)}",
                            exc_info=True,
                        )
                        if os.getenv("SENTRY_ENABLED", "False").lower() == "true":
                            sentry_sdk.capture_exception(e)
                    finally:
                        # Prevent malformed messages from stalling queue indefinitely
                        message.processed = True
                        await db.commit()
                        logger.debug(f"Marked message {message.id} as processed")

                    if self.shutdown_event.is_set():
                        logger.debug(
                            f"Shutdown requested, stopping processing for session {session_id}"
                        )
                        break

                    # Update last_updated timestamp to show this session is still being processed
                    await db.execute(
                        update(models.ActiveQueueSession)
                        .where(models.ActiveQueueSession.session_id == session_id)
                        .values(last_updated=func.now())
                    )
                    await db.commit()

                logger.debug(
                    f"Completed processing session {session_id}, processed {message_count} messages"
                )
            finally:
                # Remove session from active_sessions when done
                logger.debug(f"Removing session {session_id} from active sessions")
                await db.execute(
                    delete(models.ActiveQueueSession).where(
                        models.ActiveQueueSession.session_id == session_id
                    )
                )
                await db.commit()
                self.untrack_session(session_id)

    @sentry_sdk.trace
    async def get_next_message(self, db: AsyncSession, session_id: int):
        """Get the next unprocessed message for a session"""
        result = await db.execute(
            select(models.QueueItem)
            .where(models.QueueItem.session_id == session_id)
            .where(models.QueueItem.processed == False)
            .order_by(models.QueueItem.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
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
