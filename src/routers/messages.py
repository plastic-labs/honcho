from typing import Optional, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy.sql import insert

from src import crud, schemas
from src.db import SessionLocal
from src.dependencies import db
from src.models import QueueItem
from src.security import auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/sessions/{session_id}/messages",
    tags=["messages"],
    dependencies=[Depends(auth)],
)


async def enqueue(payload: dict | list[dict]):
    async with SessionLocal() as db:
        try:
            if isinstance(payload, list):
                if not payload:  # Empty list check
                    return
                print("Payload:\n", payload)

                # Check session once since all messages are for same session
                session = await crud.get_session(
                    db,
                    app_id=payload[0]["app_id"],
                    user_id=payload[0]["user_id"],
                    session_id=payload[0]["session_id"],
                )
                print("Session found:", session is not None)
                if session:
                    print("Session metadata:", session.h_metadata)

                if session is None or (
                    session.h_metadata.get("deriver_disabled") is not None
                    and session.h_metadata.get("deriver_disabled") is not False
                ):
                    print("Skipping enqueue due to session check")
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

                print("Number of queue records to insert:", len(queue_records))

                # Use insert to maintain order
                stmt = insert(QueueItem).returning(QueueItem)
                result = await db.execute(stmt, queue_records)
                await db.commit()
                print("Queue items inserted successfully")
                return
            else:
                # Original single insert logic
                session = await crud.get_session(
                    db,
                    app_id=payload["app_id"],
                    user_id=payload["user_id"],
                    session_id=payload["session_id"],
                )
                if session is not None:
                    deriver_disabled = session.h_metadata.get("deriver_disabled")
                    if deriver_disabled is not None and deriver_disabled is not False:
                        print("=====================")
                        print(
                            f"Deriver is not enabled on session {payload['session_id']}"
                        )
                        print("=====================")
                        return
                else:
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
                return
        except Exception as e:
            print("=====================")
            print("FAILURE: in enqueue")
            print("=====================")
            print(e)
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
        payload = {
            "app_id": app_id,
            "user_id": user_id,
            "session_id": session_id,
            "message_id": honcho_message.public_id,
            "is_user": honcho_message.is_user,
            "content": honcho_message.content,
            "metadata": honcho_message.h_metadata,
        }
        background_tasks.add_task(enqueue, payload)  # type: ignore

        return honcho_message
    except ValueError:
        print("=====================")
        print("FAILURE: in create message")
        print("=====================")
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.post("/batch", response_model=List[schemas.Message])
async def create_batch_messages_for_session(
    app_id: str,
    user_id: str,
    session_id: str,
    messages: List[schemas.MessageCreate],
    background_tasks: BackgroundTasks,
    db=db,
):
    """Bulk create messages for a session while maintaining order"""
    try:
        created_messages = await crud.create_messages(
            db, messages=messages, app_id=app_id, user_id=user_id, session_id=session_id
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

        return created_messages
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


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
        return await paginate(
            db,
            await crud.get_messages(
                db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                filter=filter,
                reverse=reverse,
            ),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


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
        raise HTTPException(status_code=404, detail="Message not found")
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
    if message.metadata is None:
        raise HTTPException(status_code=400, detail="Message metadata cannot be empty")
    try:
        return await crud.update_message(
            db,
            message=message,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None
