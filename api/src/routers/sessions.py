import json
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import agent, crud, schemas
from src.dependencies import db

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/sessions",
    tags=["sessions"],
)


@router.get("", response_model=Page[schemas.Session])
async def get_sessions(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    location_id: Optional[str] = None,
    is_active: Optional[bool] = False,
    reverse: Optional[bool] = False,
    filter: Optional[str] = None,
    db=db,
):
    """Get All Sessions for a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        location_id (str, optional): Optional Location ID representing the location of a
        session

    Returns:
        list[schemas.Session]: List of Session objects

    """

    data = None
    if filter is not None:
        data = json.loads(filter)

    return await paginate(
        db,
        await crud.get_sessions(
            db,
            app_id=app_id,
            user_id=user_id,
            location_id=location_id,
            reverse=reverse,
            is_active=is_active,
            filter=data,
        ),
    )


@router.post("", response_model=schemas.Session)
async def create_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session: schemas.SessionCreate,
    db=db,
):
    """Create a Session for a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client
        application using honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        session (schemas.SessionCreate): The Session object containing any
        metadata and a location ID

    Returns:
        schemas.Session: The Session object of the new Session

    """
    value = await crud.create_session(
        db, app_id=app_id, user_id=user_id, session=session
    )
    return value


@router.put("/{session_id}", response_model=schemas.Session)
async def update_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    session: schemas.SessionUpdate,
    db=db,
):
    """Update the metadata of a Session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        session_id (uuid.UUID): The ID of the Session to update
        session (schemas.SessionUpdate): The Session object containing any new metadata

    Returns:
        schemas.Session: The Session object of the updated Session

    """
    if session.metadata is None:
        raise HTTPException(status_code=400, detail="Session metadata cannot be empty")
    try:
        return await crud.update_session(
            db, app_id=app_id, user_id=user_id, session_id=session_id, session=session
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.delete("/{session_id}")
async def delete_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    db=db,
):
    """Delete a session by marking it as inactive

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        session_id (uuid.UUID): The ID of the Session to delete

    Returns:
        dict: A message indicating that the session was deleted

    Raises:
        HTTPException: If the session is not found

    """
    response = await crud.delete_session(
        db, app_id=app_id, user_id=user_id, session_id=session_id
    )
    if response:
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}", response_model=schemas.Session)
async def get_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    db=db,
):
    """Get a specific session for a user by ID

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        session_id (uuid.UUID): The ID of the Session to retrieve

    Returns:
        schemas.Session: The Session object of the requested Session

    Raises:
        HTTPException: If the session is not found
    """
    honcho_session = await crud.get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return honcho_session


@router.get("/{session_id}/chat", response_model=schemas.AgentChat)
async def get_chat(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    query: str,
    db=db,
):
    return await agent.chat(
        app_id=app_id, user_id=user_id, session_id=session_id, query=query, db=db
    )
