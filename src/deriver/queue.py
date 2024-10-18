import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

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


async def process_session_messages(session_id: int):
    async with SessionLocal() as db:
        try:
            while True:
                message = await get_next_message_for_session(db, session_id)
                if not message:
                    break

                await process_item(db, payload=message.payload)
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


async def schedule_session(
    semaphore: asyncio.Semaphore, queue_empty_flag: asyncio.Event
):
    async with semaphore, SessionLocal() as db:
        try:
            available_slots = semaphore._value
            # print(available_slots)
            new_sessions = await get_available_sessions(db, available_slots)

            if new_sessions:
                tasks = []
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
                        # Pass enable_timing to process_session_messages
                        asyncio.create_task(process_session_messages(session_id))
                    except IntegrityError:
                        # If the session is already in active_sessions, skip it
                        await db.rollback()

                if tasks:
                    await asyncio.gather(*tasks)
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
        await schedule_session(semaphore, queue_empty_flag)
        # await asyncio.sleep(0)  # Yield control to allow tasks to run


async def main():
    SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
    if SENTRY_ENABLED:
        sentry_sdk.init(
            dsn=os.getenv("SENTRY_DSN"),
            enable_tracing=True,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
            integrations=[
                AsyncioIntegration(),
            ],
        )
    semaphore = asyncio.Semaphore(2)  # Limit to 5 concurrent dequeuing operations
    queue_empty_flag = asyncio.Event()  # Event to signal when the queue is empty
    await polling_loop(semaphore, queue_empty_flag)
