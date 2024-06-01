import json
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.db import SessionLocal
from src.dependencies import db
from src.models import QueueItem
from src.security import auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/sessions/{session_id}/messages",
    tags=["messages"],
)


async def enqueue(payload: dict):
    async with SessionLocal() as db:
        try:
            processed_payload = {
                k: str(v) if isinstance(v, uuid.UUID) else v for k, v in payload.items()
            }
            item = QueueItem(payload=processed_payload)
            db.add(item)
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
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message: schemas.MessageCreate,
    background_tasks: BackgroundTasks,
    db=db,
    auth=Depends(auth),
):
    """Adds a message to a session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to add the message to
        message (schemas.MessageCreate): The Message object to add containing the message content and type

    Returns:
        schemas.Message: The Message object of the added message

    Raises:
        HTTPException: If the session is not found

    """
    try:
        honcho_message = await crud.create_message(
            db, message=message, app_id=app_id, user_id=user_id, session_id=session_id
        )
        print("=======")
        print("Should be enqueued")
        payload = {
            "app_id": app_id,
            "user_id": user_id,
            "session_id": session_id,
            "message_id": honcho_message.id,
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


@router.get("", response_model=Page[schemas.Message])
async def get_messages(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    reverse: Optional[bool] = False,
    filter: Optional[str] = None,
    db=db,
    auth=Depends(auth),
):
    """Get all messages for a session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to retrieve
        reverse (bool): Whether to reverse the order of the messages

    Returns:
        list[schemas.Message]: List of Message objects

    Raises:
        HTTPException: If the session is not found

    """
    try:
        data = None
        if filter is not None:
            data = json.loads(filter)
        return await paginate(
            db,
            await crud.get_messages(
                db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                filter=data,
                reverse=reverse,
            ),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.get("/{message_id}", response_model=schemas.Message)
async def get_message(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    db=db,
    auth=Depends(auth),
):
    """ """
    honcho_message = await crud.get_message(
        db, app_id=app_id, session_id=session_id, user_id=user_id, message_id=message_id
    )
    if honcho_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return honcho_message


@router.put("/{message_id}", response_model=schemas.Message)
async def update_message(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    message: schemas.MessageUpdate,
    db=db,
    auth=Depends(auth),
):
    """Update's the metadata of a message"""
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
