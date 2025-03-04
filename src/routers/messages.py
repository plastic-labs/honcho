import logging
import os
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy.sql import insert

from src import crud, schemas
from src.db import SessionLocal
from src.dependencies import db
from src.exceptions import ResourceNotFoundException
from src.models import QueueItem
from src.security import auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/sessions/{session_id}/messages",
    tags=["messages"],
    dependencies=[Depends(auth)],
)


async def enqueue(payload: dict | list[dict]):
    """
    Add message(s) to the deriver queue for processing.

    Args:
        payload: Single message payload or list of message payloads
    """
    async with SessionLocal() as db:
        try:
            if isinstance(payload, list):
                if not payload:  # Empty list check
                    logger.debug("Empty payload list, skipping enqueue")
                    return

                logger.debug(f"Enqueueing batch of {len(payload)} messages")

                # Check session once since all messages are for same session
                try:
                    session = await crud.get_session(
                        db,
                        app_id=payload[0]["app_id"],
                        user_id=payload[0]["user_id"],
                        session_id=payload[0]["session_id"],
                    )
                except ResourceNotFoundException:
                    logger.warning(
                        f"Session {payload[0]['session_id']} not found, skipping enqueue"
                    )
                    return

                logger.debug(f"Session {session.public_id} found for batch enqueue")

                # Check if deriver is disabled for this session
                if (
                    session.h_metadata.get("deriver_disabled") is not None
                    and session.h_metadata.get("deriver_disabled") is not False
                ):
                    logger.info(
                        f"Deriver is disabled for session {session.public_id}, skipping enqueue"
                    )
                    return

                # Process all payloads
                queue_records = [
                    {
                        "payload": {
                            k: str(v) if isinstance(v, str) else v for k, v in p.items()
                        },
                        "session_id": session.id,
                    }
                    for p in payload
                ]

                logger.debug(f"Inserting {len(queue_records)} queue records")

                # Use insert to maintain order
                stmt = insert(QueueItem).returning(QueueItem)
                await db.execute(stmt, queue_records)
                await db.commit()
                logger.info(f"Successfully enqueued batch of {len(payload)} messages")
                return
            else:
                # Single message enqueue
                logger.debug(
                    f"Enqueueing single message for session {payload['session_id']}"
                )

                try:
                    session = await crud.get_session(
                        db,
                        app_id=payload["app_id"],
                        user_id=payload["user_id"],
                        session_id=payload["session_id"],
                    )
                except ResourceNotFoundException:
                    logger.warning(
                        f"Session {payload['session_id']} not found, skipping enqueue"
                    )
                    return

                # Check if deriver is disabled for this session
                deriver_disabled = session.h_metadata.get("deriver_disabled")
                if deriver_disabled is not None and deriver_disabled is not False:
                    logger.info(
                        f"Deriver is disabled for session {payload['session_id']}, skipping enqueue"
                    )
                    return

                processed_payload = {
                    k: str(v) if isinstance(v, str) else v for k, v in payload.items()
                }
                # Use insert for consistency
                stmt = (
                    insert(QueueItem)
                    .values(payload=processed_payload, session_id=session.id)
                    .returning(QueueItem)
                )
                await db.execute(stmt)
                await db.commit()
                logger.info(
                    f"Successfully enqueued message for session {payload['session_id']}"
                )
                return
        except Exception as e:
            logger.error(f"Failed to enqueue message: {str(e)}", exc_info=True)
            if os.getenv("SENTRY_ENABLED", "False").lower() == "true":
                import sentry_sdk

                sentry_sdk.capture_exception(e)
            await db.rollback()


@router.post("", response_model=schemas.Message)
async def create_message_for_session(
    app_id: str,
    user_id: str,
    session_id: str,
    message: schemas.MessageCreate,
    background_tasks: BackgroundTasks,
    db=db,
):
    """Adds a message to a session"""
    try:
        honcho_message = await crud.create_message(
            db, message=message, app_id=app_id, user_id=user_id, session_id=session_id
        )

        # Prepare message payload for background processing
        payload = {
            "app_id": app_id,
            "user_id": user_id,
            "session_id": session_id,
            "message_id": honcho_message.public_id,
            "is_user": honcho_message.is_user,
            "content": honcho_message.content,
            "metadata": honcho_message.h_metadata,
        }

        # Queue message for background processing
        background_tasks.add_task(enqueue, payload)  # type: ignore
        logger.info(
            f"Message {honcho_message.public_id} created and queued for processing"
        )

        return honcho_message
    except ValueError as e:
        logger.error(f"Failed to create message for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.post("/batch", response_model=List[schemas.Message])
async def create_batch_messages_for_session(
    app_id: str,
    user_id: str,
    session_id: str,
    batch: schemas.MessageBatchCreate,
    background_tasks: BackgroundTasks,
    db=db,
):
    """Bulk create messages for a session while maintaining order. Maximum 100 messages per batch."""
    try:
        created_messages = await crud.create_messages(
            db,
            messages=batch.messages,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
        )

        # Create payloads for all messages
        payloads = [
            {
                "app_id": app_id,
                "user_id": user_id,
                "session_id": session_id,
                "message_id": message.public_id,
                "is_user": message.is_user,
                "content": message.content,
                "metadata": message.h_metadata,
            }
            for message in created_messages
        ]

        # Enqueue all messages in one call
        background_tasks.add_task(enqueue, payloads)  # type: ignore
        logger.info(
            f"Batch of {len(created_messages)} messages created and queued for processing"
        )

        return created_messages
    except ValueError as e:
        logger.error(
            f"Failed to create batch messages for session {session_id}: {str(e)}"
        )
        raise ResourceNotFoundException("Session not found") from e


@router.post("/list", response_model=Page[schemas.Message])
async def get_messages(
    app_id: str,
    user_id: str,
    session_id: str,
    options: schemas.MessageGet,
    reverse: Optional[bool] = False,
    db=db,
):
    """Get all messages for a session"""
    try:
        filter = options.filter
        if options.filter == {}:
            filter = None

        messages_query = await crud.get_messages(
            db,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            filter=filter,
            reverse=reverse,
        )

        return await paginate(db, messages_query)
    except ValueError as e:
        logger.warning(f"Failed to get messages for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get("/{message_id}", response_model=schemas.Message)
async def get_message(
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    db=db,
):
    """Get a Message by ID"""
    honcho_message = await crud.get_message(
        db, app_id=app_id, session_id=session_id, user_id=user_id, message_id=message_id
    )
    if honcho_message is None:
        logger.warning(f"Message {message_id} not found in session {session_id}")
        raise ResourceNotFoundException(f"Message with ID {message_id} not found")
    return honcho_message


@router.put("/{message_id}", response_model=schemas.Message)
async def update_message(
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    message: schemas.MessageUpdate,
    db=db,
):
    """Update the metadata of a Message"""
    try:
        updated_message = await crud.update_message(
            db,
            message=message,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
        )
        logger.info(f"Message {message_id} updated successfully")
        return updated_message
    except ValueError as e:
        logger.warning(f"Failed to update message {message_id}: {str(e)}")
        raise ResourceNotFoundException("Message or session not found") from e
