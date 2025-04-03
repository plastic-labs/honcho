import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/metamessages",
    tags=["metamessages"],
    dependencies=[Depends(require_auth(
        app_id="app_id",
        user_id="user_id"
    ))],
)


@router.post("", response_model=schemas.Metamessage)
async def create_metamessage(
    app_id: str,
    user_id: str,
    metamessage: schemas.MetamessageCreate,
    db=db,
):
    """
    Create a new metamessage associated with a user.
    Optionally link to a session and message by providing those IDs in the request body.
    """
    try:
        # Set the user_id from the URL parameters
        metamessage.user_id = user_id

        metamessage_obj = await crud.create_metamessage(
            db,
            metamessage=metamessage,
            app_id=app_id,
        )
        logger.info(f"Metamessage created successfully for user {user_id}")
        return metamessage_obj
    except (ResourceNotFoundException, ValidationException) as e:
        logger.warning(f"Failed to create metamessage: {str(e)}")
        raise


@router.post("/list", response_model=Page[schemas.Metamessage])
async def get_metamessages(
    app_id: str,
    user_id: str,
    options: schemas.MetamessageGet,
    reverse: Optional[bool] = False,
    db=db,
):
    """
    Get metamessages with flexible filtering.

    - Filter by user only: No additional parameters needed
    - Filter by session: Provide session_id
    - Filter by message: Provide message_id (and session_id)
    - Filter by type: Provide metamessage_type
    - Filter by metadata: Provide filter object
    """
    try:
        # Use user_id from URL path
        metamessages_query = await crud.get_metamessages(
            db,
            app_id=app_id,
            user_id=user_id,
            session_id=options.session_id,
            message_id=options.message_id,
            metamessage_type=options.metamessage_type,
            filter=options.filter,
            reverse=reverse,
        )
        return await paginate(db, metamessages_query)
    except (ResourceNotFoundException, ValidationException) as e:
        logger.warning(f"Failed to get metamessages: {str(e)}")
        raise


@router.get(
    "/{metamessage_id}",
    response_model=schemas.Metamessage,
)
async def get_metamessage(
    app_id: str,
    user_id: str,
    metamessage_id: str,
    db=db,
):
    """Get a specific Metamessage by ID"""
    honcho_metamessage = await crud.get_metamessage(
        db,
        app_id=app_id,
        user_id=user_id,
        metamessage_id=metamessage_id,
    )
    if honcho_metamessage is None:
        logger.warning(f"Metamessage {metamessage_id} not found")
        raise ResourceNotFoundException(
            f"Metamessage with ID {metamessage_id} not found"
        )
    return honcho_metamessage


@router.put(
    "/{metamessage_id}",
    response_model=schemas.Metamessage,
)
async def update_metamessage(
    app_id: str,
    user_id: str,
    metamessage_id: str,
    metamessage: schemas.MetamessageUpdate,
    db=db,
):
    """Update a metamessage's metadata, type, or relationships"""
    # Ensure user_id from URL path is used
    metamessage.user_id = user_id

    try:
        updated_metamessage = await crud.update_metamessage(
            db,
            metamessage=metamessage,
            app_id=app_id,
            metamessage_id=metamessage_id,
        )
        logger.info(f"Metamessage {metamessage_id} updated successfully")
        return updated_metamessage
    except (ResourceNotFoundException, ValidationException) as e:
        logger.warning(f"Failed to update metamessage {metamessage_id}: {str(e)}")
        raise
