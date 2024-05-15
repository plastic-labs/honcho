import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.security import auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/sessions/{session_id}/metamessages",
    tags=["messages"],
)


@router.post("", response_model=schemas.Metamessage)
async def create_metamessage(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    metamessage: schemas.MetamessageCreate,
    db=db,
    auth=Depends(auth),
):
    """Adds a message to a session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to add the message to
        message (schemas.MessageCreate): The Message object to add containing the
        message content and type

    Returns:
        schemas.Message: The Message object of the added message

    Raises:
        HTTPException: If the session is not found

    """
    try:
        return await crud.create_metamessage(
            db,
            metamessage=metamessage,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.get("", response_model=Page[schemas.Metamessage])
async def get_metamessages(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: Optional[uuid.UUID] = None,
    metamessage_type: Optional[str] = None,
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
        reverse (bool): Whether to reverse the order of the metamessages

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
            await crud.get_metamessages(
                db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                metamessage_type=metamessage_type,
                filter=data,
                reverse=reverse,
            ),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.get(
    "/{metamessage_id}",
    response_model=schemas.Metamessage,
)
async def get_metamessage(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    metamessage_id: uuid.UUID,
    db=db,
    auth=Depends(auth),
):
    """Get a specific Metamessage by ID

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to retrieve

    Returns:
        schemas.Session: The Session object of the requested Session

    Raises:
        HTTPException: If the session is not found
    """
    honcho_metamessage = await crud.get_metamessage(
        db,
        app_id=app_id,
        session_id=session_id,
        user_id=user_id,
        message_id=message_id,
        metamessage_id=metamessage_id,
    )
    if honcho_metamessage is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return honcho_metamessage


@router.put(
    "/{metamessage_id}",
    response_model=schemas.Metamessage,
)
async def update_metamessage(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    metamessage_id: uuid.UUID,
    metamessage: schemas.MetamessageUpdate,
    db=db,
    auth=Depends(auth),
):
    """Update's the metadata of a metamessage"""
    if metamessage.metadata is None:
        raise HTTPException(
            status_code=400, detail="Metamessage metadata cannot be empty"
        )
    try:
        return await crud.update_metamessage(
            db,
            metamessage=metamessage,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            metamessage_id=metamessage_id,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None
