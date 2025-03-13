import asyncio
import os
import signal
from datetime import datetime, timedelta

import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from .. import models
from ..db import SessionLocal
from .consumer import process_item

load_dotenv()


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
        """Initialize the queue manager"""
        print(f"[QUEUE] Initializing QueueManager with {self.workers} workers")
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(self.shutdown(sig))
            )
            
        print("[QUEUE] Signal handlers registered")
        
        # Start polling loop as a separate task
        poll_task = asyncio.create_task(self.polling_loop())
        self.add_task(poll_task)
        print("[QUEUE] Polling loop started")

    async def shutdown(self, sig: signal.Signals):
        """Handle graceful shutdown"""
        print(f"Received exit signal {sig.name}...")
        self.shutdown_event.set()

        if self.active_tasks:
            print(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

    async def cleanup(self):
        """Clean up owned sessions"""
        if self.owned_sessions:
            print(f"Cleaning up {len(self.owned_sessions)} owned sessions...")
            async with SessionLocal() as db:
                await db.execute(
                    delete(models.ActiveQueueSession).where(
                        models.ActiveQueueSession.session_id.in_(self.owned_sessions)
                    )
                )
                await db.commit()

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
        """Continuously poll for new messages to process"""
        print("[QUEUE] Starting polling loop")
        while not self.shutdown_event.is_set():
            try:
                async with SessionLocal() as db:
                    # Check for available sessions
                    sessions = await self.get_available_sessions(db)
                    print(f"[QUEUE] Found {len(sessions)} available sessions to process")
                    
                    if not sessions:
                        # Nothing to process, set the flag
                        self.queue_empty_flag.set()
                        print("[QUEUE] No sessions to process, waiting...")
                        # Wait for a bit before checking again
                        await asyncio.sleep(5)
                        continue
                    else:
                        # Clear the flag, we have work to do
                        self.queue_empty_flag.clear()
                        
                    # Process sessions
                    for session_id in sessions:
                        if session_id not in self.owned_sessions:
                            # Try to claim this session
                            try:
                                # Insert record into ActiveQueueSession to claim it
                                await db.execute(
                                    insert(models.ActiveQueueSession).values(
                                        session_id=session_id,
                                        last_updated=func.now()
                                    )
                                )
                                await db.commit()
                                
                                # Track locally
                                self.track_session(session_id)
                                print(f"[QUEUE] Claimed session {session_id} for processing")
                                
                                # Start processing task
                                task = asyncio.create_task(self.process_session(session_id))
                                self.add_task(task)
                            except IntegrityError:
                                # Someone else already claimed this session
                                await db.rollback()
                                print(f"[QUEUE] Failed to claim session {session_id}, already owned")
            except Exception as e:
                print(f"[QUEUE] Error in polling loop: {str(e)}")
                sentry_sdk.capture_exception(e)
            
            # Wait before next poll
            await asyncio.sleep(1)
        
        print("[QUEUE] Polling loop ending")

    ######################
    # Queue Worker Logic #
    ######################

    @sentry_sdk.trace
    async def process_session(self, session_id: int):
        """Process all messages for a session"""
        print(f"[QUEUE] Starting to process session {session_id}")
        async with self.semaphore:  # Hold the semaphore for the entire session duration
            async with SessionLocal() as db:
                try:
                    message_count = 0
                    while not self.shutdown_event.is_set():
                        message = await self.get_next_message(db, session_id)
                        if not message:
                            print(f"[QUEUE] No more messages for session {session_id}")
                            break
                        
                        message_count += 1
                        print(f"[QUEUE] Processing message {message.id} for session {session_id} (message {message_count})")
                        try:
                            await process_item(db, payload=message.payload)
                            print(f"[QUEUE] Successfully processed message {message.id}")
                        except Exception as e:
                            print(f"[QUEUE] Error processing message {message.id}: {str(e)}")
                            sentry_sdk.capture_exception(e)
                        finally:
                            # Prevent malformed messages from stalling queue indefinitely
                            message.processed = True
                            await db.commit()
                            print(f"[QUEUE] Marked message {message.id} as processed")

                        if self.shutdown_event.is_set():
                            print(f"[QUEUE] Shutdown requested, stopping processing for session {session_id}")
                            break

                        # Update last_updated timestamp to show this session is still being processed
                        await db.execute(
                            update(models.ActiveQueueSession)
                            .where(models.ActiveQueueSession.session_id == session_id)
                            .values(last_updated=func.now())
                        )
                        await db.commit()
                    
                    print(f"[QUEUE] Completed processing session {session_id}, processed {message_count} messages")
                finally:
                    # Remove session from active_sessions when done
                    print(f"[QUEUE] Removing session {session_id} from active sessions")
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
    manager = QueueManager()
    await manager.initialize()
    
    # Wait until shutdown is requested
    print("[QUEUE] Main loop running, waiting for shutdown signal")
    try:
        # Keep the process alive until a shutdown is requested
        await manager.shutdown_event.wait()
    except Exception as e:
        print(f"[QUEUE] Error in main loop: {str(e)}")
        sentry_sdk.capture_exception(e)
    finally:
        # Clean up resources before exiting
        print("[QUEUE] Main loop ending, cleaning up resources")
        await manager.cleanup()
        print("[QUEUE] Cleanup complete")
