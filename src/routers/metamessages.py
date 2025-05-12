import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.routers import transactions
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/metamessages",
    tags=["metamessages"],
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)


@router.post("", response_model=schemas.Metamessage)
async def create_metamessage(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    metamessage: schemas.MetamessageCreate = Body(
        ..., description="Metamessage creation parameters"
    ),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """
    Create a new metamessage associated with a user.
    Optionally link to a session and message by providing those IDs in the request body.
    """
    try:
        metamessage_obj = await crud.create_metamessage(
            db,
            user_id=user_id,
            metamessage=metamessage,
            app_id=app_id,
            transaction_id=transaction_id,
        )
        logger.info(f"Metamessage created successfully for user {user_id}")
        return metamessage_obj
    except (ResourceNotFoundException, ValidationException) as e:
        logger.warning(f"Failed to create metamessage: {str(e)}")
        raise


@router.post("/list", response_model=Page[schemas.Metamessage], dependencies=[Depends(transactions.disallow_transaction_header)])
async def get_metamessages(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    options: schemas.MetamessageGet = Body(
        ..., description="Filtering options for the metamessages list"
    ),
    reverse: Optional[bool] = Query(
        False, description="Whether to reverse the order of results"
    ),
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
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    metamessage_id: str = Path(..., description="ID of the metamessage to retrieve"),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """Get a specific Metamessage by ID"""
    honcho_metamessage = await crud.get_metamessage(
        db,
        app_id=app_id,
        user_id=user_id,
        metamessage_id=metamessage_id,
        transaction_id=transaction_id,
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
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    metamessage_id: str = Path(..., description="ID of the metamessage to update"),
    metamessage: schemas.MetamessageUpdate = Body(
        ..., description="Updated metamessage parameters"
    ),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """Update a metamessage's metadata, type, or relationships"""
    try:
        updated_metamessage = await crud.update_metamessage(
            db,
            metamessage=metamessage,
            app_id=app_id,
            user_id=user_id,
            metamessage_id=metamessage_id,
            transaction_id=transaction_id,
        )
        logger.info(f"Metamessage {metamessage_id} updated successfully")
        return updated_metamessage
    except (ResourceNotFoundException, ValidationException) as e:
        logger.warning(f"Failed to update metamessage {metamessage_id}: {str(e)}")
        raise
