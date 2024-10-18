import json
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

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


async def enqueue(payload: dict):
    async with SessionLocal() as db:
        # Get Session and Check metadata
        session = await crud.get_session(
            db,
            app_id=payload["app_id"],
            user_id=payload["user_id"],
            session_id=payload["session_id"],
        )
        # Check if metadata has a "deriver" key
        if session is not None:
            deriver_disabled = session.h_metadata.get("deriver_disabled")
            if deriver_disabled is not None and deriver_disabled is not False:
                print("=====================")
                print(f"Deriver is not enabled on session {payload['session_id']}")
                print("=====================")
                # If deriver is not enabled, do not enqueue
                return
        else:
            # Session doesn't exist return
            return
        try:
            processed_payload = {
                k: str(v) if isinstance(v, str) else v for k, v in payload.items()
            }
            item = QueueItem(payload=processed_payload, session_id=session.id)
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
    app_id: str,
    user_id: str,
    session_id: str,
    message: schemas.MessageCreate,
    background_tasks: BackgroundTasks,
    db=db,
):
    """Adds a message to a session

    Args:
        app_id (str): The ID of the app representing the client application using honcho
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


@router.get("", response_model=Page[schemas.Message])
async def get_messages(
    app_id: str,
    user_id: str,
    session_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[str] = None,
    db=db,
):
    """Get all messages for a session

    Args:
        app_id (str): The ID of the app representing the client application using
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
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    db=db,
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
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    message: schemas.MessageUpdate,
    db=db,
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
