import asyncio
import os
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Optional

import sentry_sdk
from dotenv import load_dotenv
from rich import print as rprint
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from .. import models
from ..db import SessionLocal
from .consumer import process_item

load_dotenv()


async def get_next_message_for_session(
    db: AsyncSession, session_id: int
) -> Optional[models.QueueItem]:
    result = await db.execute(
        select(models.QueueItem)
        .where(models.QueueItem.session_id == session_id)
        .where(models.QueueItem.processed == False)
        .order_by(models.QueueItem.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    return result.scalar_one_or_none()


@sentry_sdk.trace
async def process_session_messages(session_id: int):
    async with SessionLocal() as db:
        try:
            while True:
                message = await get_next_message_for_session(db, session_id)
                if not message:
                    break
                try:
                    await process_item(db, payload=message.payload)
                except Exception as e:
                    print(e)
                    sentry_sdk.capture_exception(e)
                finally:
                    # Prevent malformed messages from stalling a queue indefinitely
                    message.processed = True
                    await db.commit()

                # Update last_updated to show this session is still being processed
                await db.execute(
                    update(models.ActiveQueueSession)
                    .where(models.ActiveQueueSession.session_id == session_id)
                    .values(last_updated=func.now())
                )
                await db.commit()
        finally:
            # Remove session from active_sessions when done
            await db.execute(
                delete(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.session_id == session_id
                )
            )
            await db.commit()


@sentry_sdk.trace
async def get_available_sessions(db: AsyncSession, limit: int) -> Sequence[Any]:
    # First, clean up stale sessions (e.g., older than 5 minutes)
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    await db.execute(
        delete(models.ActiveQueueSession).where(
            models.ActiveQueueSession.last_updated < five_minutes_ago
        )
    )

    # Then get available sessions
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
        .limit(limit)
    )
    return result.scalars().all()


@sentry_sdk.trace
async def schedule_session(
    semaphore: asyncio.Semaphore, queue_empty_flag: asyncio.Event
):
    async with (
        semaphore,
        SessionLocal() as db,
    ):
        with sentry_sdk.start_transaction(
            op="deriver_schedule_session", name="Schedule Deriver Session"
        ):
            try:
                # available_slots = semaphore._value
                # print(available_slots)
                new_sessions = await get_available_sessions(db, 1)

                if new_sessions:
                    for session_id in new_sessions:
                        try:
                            # Try to insert the session into active_sessions
                            await db.execute(
                                insert(models.ActiveQueueSession).values(
                                    session_id=session_id
                                )
                            )
                            await db.commit()

                            # If successful, create a task for this session
                            await process_session_messages(session_id)
                        except IntegrityError:
                            # If the session is already in active_sessions, skip it
                            await db.rollback()

                else:
                    # No items to process, set the queue_empty_flag
                    queue_empty_flag.set()
            except Exception as e:
                rprint("==========")
                rprint("Exception")
                rprint(e)
                rprint("==========")
                await db.rollback()


async def polling_loop(semaphore: asyncio.Semaphore, queue_empty_flag: asyncio.Event):
    while True:
        if queue_empty_flag.is_set():
            await asyncio.sleep(1)  # Sleep briefly if the queue is empty
            queue_empty_flag.clear()  # Reset the flag
            continue
        if semaphore.locked():
            await asyncio.sleep(1)  # Sleep briefly if the semaphore is fully locked
            continue
        # Create a task instead of awaiting
        asyncio.create_task(schedule_session(semaphore, queue_empty_flag))
        await asyncio.sleep(0)  # Give other tasks a chance to run


async def main():
    SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
    if SENTRY_ENABLED:
        sentry_sdk.init(
            dsn=os.getenv("SENTRY_DSN"),
            enable_tracing=True,
            traces_sample_rate=0.4,
            profiles_sample_rate=0.4,
            integrations=[
                AsyncioIntegration(),
            ],
        )
    workers = int(os.getenv("DERIVER_WORKERS", 1)) + 1
    semaphore = asyncio.Semaphore(workers)  # Limit to 5 concurrent dequeuing operations
    queue_empty_flag = asyncio.Event()  # Event to signal when the queue is empty
    await polling_loop(semaphore, queue_empty_flag)
