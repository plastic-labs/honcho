import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.security import auth

logger = logging.getLogger(__name__)

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
    """Adds a message to a session"""
    try:
        metamessage_obj = await crud.create_metamessage(
            db,
            metamessage=metamessage,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
        )
        logger.info(f"Metamessage created successfully for session {session_id}")
        return metamessage_obj
    except ValueError as e:
        logger.warning(f"Failed to create metamessage for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.post("/list", response_model=Page[schemas.Metamessage])
async def get_metamessages(
    app_id: str,
    user_id: str,
    session_id: str,
    options: schemas.MetamessageGet,
    reverse: Optional[bool] = False,
    db=db,
):
    """Get all messages for a session"""
    try:
        metamessages_query = await crud.get_metamessages(
            db,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            message_id=options.message_id,
            metamessage_type=options.metamessage_type,
            filter=options.filter,
            reverse=reverse,
        )
        
        return await paginate(db, metamessages_query)
    except ValueError as e:
        logger.warning(f"Failed to get metamessages for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router_user_level.post("/list", response_model=Page[schemas.Metamessage])
async def get_metamessages_by_user(
    app_id: str,
    user_id: str,
    options: schemas.MetamessageGetUserLevel,
    reverse: Optional[bool] = False,
    db=db,
):
    """Paginate through the user metamessages for a user"""
    try:
        metamessages_query = await crud.get_metamessages(
            db,
            app_id=app_id,
            user_id=user_id,
            metamessage_type=options.metamessage_type,
            reverse=reverse,
            filter=options.filter,
        )
        
        return await paginate(db, metamessages_query)
    except ValueError as e:
        logger.warning(f"Failed to get metamessages for user {user_id}: {str(e)}")
        raise ResourceNotFoundException("User not found") from e


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
    """Get a specific Metamessage by ID"""
    honcho_metamessage = await crud.get_metamessage(
        db,
        app_id=app_id,
        session_id=session_id,
        user_id=user_id,
        message_id=message_id,
        metamessage_id=metamessage_id,
    )
    if honcho_metamessage is None:
        logger.warning(f"Metamessage {metamessage_id} not found for message {message_id}")
        raise ResourceNotFoundException(f"Metamessage with ID {metamessage_id} not found")
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
        logger.warning(f"Update attempted with empty metadata for metamessage {metamessage_id}")
        raise ValidationException("Metamessage metadata cannot be empty")
        
    try:
        updated_metamessage = await crud.update_metamessage(
            db,
            metamessage=metamessage,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            metamessage_id=metamessage_id,
        )
        logger.info(f"Metamessage {metamessage_id} updated successfully")
        return updated_metamessage
    except ValueError as e:
        logger.warning(f"Failed to update metamessage {metamessage_id}: {str(e)}")
        raise ResourceNotFoundException("Session or metamessage not found") from e
