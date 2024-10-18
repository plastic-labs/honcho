from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.security import auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/sessions/{session_id}/metamessages",
    tags=["metamessages"],
    dependencies=[Depends(auth)],
)

router_user_level = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/metamessages",
    tags=["metamessages"],
    dependencies=[Depends(auth)],
)


@router.post("", response_model=schemas.Metamessage)
async def create_metamessage(
    app_id: str,
    user_id: str,
    session_id: str,
    metamessage: schemas.MetamessageCreate,
    db=db,
):
    """Adds a message to a session

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to add the message to
        metamessage (schemas.MeteamessageCreate): The metamessage creation object

    Returns:
        schemas.Metamessage: The Metamessage object of the added metamessage

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


@router.post("/list", response_model=Page[schemas.Metamessage])
async def get_metamessages(
    app_id: str,
    user_id: str,
    session_id: str,
    options: schemas.MetamessageGet,
    reverse: Optional[bool] = False,
    db=db,
):
    """Get all messages for a session

    Args:
        app_id (str): The ID of the app representing the client application using
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
        return await paginate(
            db,
            await crud.get_metamessages(
                db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                message_id=options.message_id,
                metamessage_type=options.metamessage_type,
                filter=options.filter,
                reverse=reverse,
            ),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router_user_level.post("/list", response_model=Page[schemas.Metamessage])
async def get_metamessages_by_user(
    app_id: str,
    user_id: str,
    options: schemas.MetamessageGetUserLevel,
    reverse: Optional[bool] = False,
    db=db,
):
    """Paginate through the user metamessages for a user

    Args:
        app_id (str): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        reverse (bool): Whether to reverse the order of the metamessages

    Returns:
        list[schemas.Message]: List of Message objects

    Raises:
        HTTPException: If the session is not found

    """
    try:
        return await paginate(
            db,
            await crud.get_metamessages(
                db,
                app_id=app_id,
                user_id=user_id,
                metamessage_type=options.metamessage_type,
                reverse=reverse,
                filter=options.filter,
            ),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found") from None


@router.get(
    "/{metamessage_id}",
    response_model=schemas.Metamessage,
)
async def get_metamessage(
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    metamessage_id: str,
    db=db,
):
    """Get a specific Metamessage by ID

    Args:
        app_id (str): The ID of the app representing the client application using
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
    app_id: str,
    user_id: str,
    session_id: str,
    metamessage_id: str,
    metamessage: schemas.MetamessageUpdate,
    db=db,
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
