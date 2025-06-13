import asyncio
import logging
import os
import signal
from datetime import datetime, timedelta, timezone
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
        self.owned_users: set[str] = set()
        self.queue_empty_flag = asyncio.Event()
        self.last_metrics_log = datetime.now(timezone.utc)

        # Initialize from environment
        self.workers = int(os.getenv("DERIVER_WORKERS", 10))
        self.semaphore = asyncio.Semaphore(self.workers)
        
        # Queue locking mode: 'session' (default) or 'user'
        self.lock_mode = os.getenv("QUEUE_LOCK_MODE", "session").lower()
        if self.lock_mode not in ["session", "user"]:
            logger.warning(f"Invalid QUEUE_LOCK_MODE '{self.lock_mode}', defaulting to 'session'")
            self.lock_mode = "session"
            
        # Monitoring metrics
        self.metrics = {
            "sessions_claimed": 0,
            "users_claimed": 0, 
            "sessions_processed": 0,
            "messages_processed": 0,
            "claim_failures": 0,
            "processing_errors": 0
        }

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
        
    def track_user(self, user_id: str):
        """Track a new user owned by this process"""
        self.owned_users.add(user_id)

    def untrack_user(self, user_id: str):
        """Remove a user from tracking"""
        self.owned_users.discard(user_id)

    async def initialize(self):
        """Setup signal handlers and start the main polling loop"""
        logger.info(f"Initializing QueueManager with {self.workers} workers, lock_mode='{self.lock_mode}'")

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
        """Clean up owned sessions and users"""
        cleanup_tasks = []
        
        if self.owned_sessions:
            logger.info(f"Cleaning up {len(self.owned_sessions)} owned sessions...")
            cleanup_tasks.append(self._cleanup_sessions())
            
        if self.owned_users:
            logger.info(f"Cleaning up {len(self.owned_users)} owned users...")
            cleanup_tasks.append(self._cleanup_users())
            
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                logger.info("Cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                if os.getenv("SENTRY_ENABLED", "False").lower() == "true":
                    sentry_sdk.capture_exception(e)
                    
    async def _cleanup_sessions(self):
        """Clean up owned sessions"""
        async with tracked_db("queue_cleanup_sessions") as db:
            await db.execute(
                delete(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.session_id.in_(self.owned_sessions)
                )
            )
            await db.commit()
            
    async def _cleanup_users(self):
        """Clean up owned users"""
        async with tracked_db("queue_cleanup_users") as db:
            await db.execute(
                delete(models.ActiveQueueUser).where(
                    models.ActiveQueueUser.user_id.in_(self.owned_users)
                )
            )
            await db.commit()

    ##########################
    # Polling and Scheduling #
    ##########################

    async def get_available_sessions(self, db: AsyncSession):
        """Get available sessions based on the configured lock mode"""
        if self.lock_mode == "user":
            return await self._get_available_sessions_user_mode(db)
        else:
            return await self._get_available_sessions_session_mode(db)
            
    async def _get_available_sessions_session_mode(self, db: AsyncSession):
        """Get available sessions that aren't being processed (session-level locking)"""
        # Clean up stale sessions
        five_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
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
            .where(models.QueueItem.processed.is_(False))
            .where(
                models.ActiveQueueSession.session_id.is_(None)
            )  # Only sessions not in active_sessions
            .group_by(models.QueueItem.session_id)
            .limit(1)
        )
        return result.scalars().all()
        
    async def _get_available_sessions_user_mode(self, db: AsyncSession):
        """Get available sessions from users that aren't being processed (user-level locking)"""
        # Clean up stale users
        five_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
        await db.execute(
            delete(models.ActiveQueueUser).where(
                models.ActiveQueueUser.last_updated < five_minutes_ago
            )
        )

        # Get sessions from users not currently being processed
        # Join QueueItem -> Session -> User, exclude users in ActiveQueueUser
        result = await db.execute(
            select(models.QueueItem.session_id)
            .join(models.Session, models.QueueItem.session_id == models.Session.id)
            .outerjoin(
                models.ActiveQueueUser,
                models.Session.user_id == models.ActiveQueueUser.user_id,
            )
            .where(models.QueueItem.processed.is_(False))
            .where(models.ActiveQueueUser.user_id.is_(None))  # Only users not in active_users
            .group_by(models.QueueItem.session_id)
            .limit(1)
        )
        return result.scalars().all()
        
    async def _claim_session(self, db: AsyncSession, session_id: int) -> bool:
        """Try to claim a session for processing (session-level locking)"""
        try:
            await db.execute(
                insert(models.ActiveQueueSession).values(session_id=session_id)
            )
            await db.commit()
            self.track_session(session_id)
            self.metrics["sessions_claimed"] += 1
            logger.debug(f"Claimed session {session_id} for processing")
            return True
        except IntegrityError:
            # Session already claimed by another process
            self.metrics["claim_failures"] += 1
            logger.debug(f"Failed to claim session {session_id}, already owned")
            return False
            
    async def _claim_user_for_session(self, db: AsyncSession, session_id: int) -> bool:
        """Try to claim a user for processing (user-level locking)"""
        try:
            # First get the user_id for this session
            result = await db.execute(
                select(models.Session.user_id).where(models.Session.id == session_id)
            )
            user_id = result.scalar_one_or_none()
            
            if not user_id:
                logger.warning(f"No user found for session {session_id}")
                return False
                
            # Try to claim the user
            await db.execute(
                insert(models.ActiveQueueUser).values(user_id=user_id)
            )
            await db.commit()
            self.track_user(user_id)
            self.metrics["users_claimed"] += 1
            logger.debug(f"Claimed user {user_id} (session {session_id}) for processing")
            return True
        except IntegrityError:
            # User already claimed by another process
            self.metrics["claim_failures"] += 1
            logger.debug(f"Failed to claim user for session {session_id}, user already owned")
            return False
            
    def _log_metrics_if_needed(self):
        """Log metrics every 5 minutes"""
        now = datetime.now(timezone.utc)
        if (now - self.last_metrics_log).total_seconds() >= 300:  # 5 minutes
            logger.info(
                f"Queue metrics: sessions_claimed={self.metrics['sessions_claimed']}, "
                f"users_claimed={self.metrics['users_claimed']}, "
                f"sessions_processed={self.metrics['sessions_processed']}, "
                f"messages_processed={self.metrics['messages_processed']}, "
                f"claim_failures={self.metrics['claim_failures']}, "
                f"processing_errors={self.metrics['processing_errors']}, "
                f"active_sessions={len(self.owned_sessions)}, "
                f"active_users={len(self.owned_users)}"
            )
            self.last_metrics_log = now

    async def polling_loop(self):
        """Main polling loop to find and process new sessions"""
        logger.debug("Starting polling loop")
        try:
            while not self.shutdown_event.is_set():
                # Log metrics periodically
                self._log_metrics_if_needed()
                
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
                                    # Try to claim the session/user based on lock mode
                                    if self.lock_mode == "user":
                                        claimed = await self._claim_user_for_session(db, session_id)
                                    else:
                                        claimed = await self._claim_session(db, session_id)
                                        
                                    if claimed and not self.shutdown_event.is_set():
                                        # Create a new task for processing this session
                                        task = asyncio.create_task(
                                            self.process_session(session_id)
                                        )
                                        self.add_task(task)
                                except Exception as e:
                                    logger.debug(f"Failed to claim session {session_id}: {str(e)}")
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
                        self.metrics["messages_processed"] += 1
                    except Exception as e:
                        self.metrics["processing_errors"] += 1
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

                    # Update last_updated timestamp based on lock mode
                    if self.lock_mode == "user":
                        # Get user_id and update user timestamp
                        result = await db.execute(
                            select(models.Session.user_id).where(models.Session.id == session_id)
                        )
                        user_id = result.scalar_one_or_none()
                        if user_id:
                            await db.execute(
                                update(models.ActiveQueueUser)
                                .where(models.ActiveQueueUser.user_id == user_id)
                                .values(last_updated=func.now())
                            )
                    else:
                        # Update session timestamp
                        await db.execute(
                            update(models.ActiveQueueSession)
                            .where(models.ActiveQueueSession.session_id == session_id)
                            .values(last_updated=func.now())
                        )
                    await db.commit()

                self.metrics["sessions_processed"] += 1
                logger.debug(
                    f"Completed processing session {session_id}, processed {message_count} messages"
                )
            finally:
                # Remove from active tracking based on lock mode
                if self.lock_mode == "user":
                    # Get user_id and remove user from active tracking
                    result = await db.execute(
                        select(models.Session.user_id).where(models.Session.id == session_id)
                    )
                    user_id = result.scalar_one_or_none()
                    if user_id:
                        logger.debug(f"Removing user {user_id} from active users")
                        await db.execute(
                            delete(models.ActiveQueueUser).where(
                                models.ActiveQueueUser.user_id == user_id
                            )
                        )
                        self.untrack_user(user_id)
                else:
                    # Remove session from active_sessions when done
                    logger.debug(f"Removing session {session_id} from active sessions")
                    await db.execute(
                        delete(models.ActiveQueueSession).where(
                            models.ActiveQueueSession.session_id == session_id
                        )
                    )
                    self.untrack_session(session_id)
                await db.commit()

    @sentry_sdk.trace
    async def get_next_message(self, db: AsyncSession, session_id: int):
        """Get the next unprocessed message for a session"""
        result = await db.execute(
            select(models.QueueItem)
            .where(models.QueueItem.session_id == session_id)
            .where(models.QueueItem.processed.is_(False))
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
