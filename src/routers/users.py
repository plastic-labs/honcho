import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Path, Body
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import (
    AuthenticationException,
    ResourceNotFoundException,
)
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps/{app_id}/users",
    tags=["users"],
)


@router.post(
    "",
    response_model=schemas.User,
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
async def create_user(
    app_id: str = Path(..., description="ID of the app"),
    user: schemas.UserCreate = Body(..., description="User creation parameters"),
    db=db,
):
    """Create a new User"""
    user_obj = await crud.create_user(db, app_id=app_id, user=user)
    return user_obj


@router.post(
    "/list",
    response_model=Page[schemas.User],
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
async def get_users(
    app_id: str = Path(..., description="ID of the app"),
    options: schemas.UserGet = Body(..., description="Filtering options for the users list"),
    reverse: bool = Query(False, description="Whether to reverse the order of results"),
    db=db,
):
    """Get All Users for an App"""
    return await paginate(
        db,
        await crud.get_users(db, app_id=app_id, reverse=reverse, filter=options.filter),
    )


@router.get(
    "",
    response_model=schemas.User,
)
async def get_user(
    app_id: str = Path(..., description="ID of the app"),
    user_id: Optional[str] = Query(
        None, description="User ID to retrieve. If not provided, users JWT token"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db=db,
):
    """
    Get a User by ID

    If user_id is provided as a query parameter, it uses that (must match JWT app_id).
    Otherwise, it uses the user_id from the JWT token.
    """
    # validate app query param
    if not jwt_params.ad and jwt_params.ap is not None and jwt_params.ap != app_id:
        raise AuthenticationException("Unauthorized access to resource")

    if user_id:
        if not jwt_params.ad and jwt_params.us is not None and jwt_params.us != user_id:
            raise AuthenticationException("Unauthorized access to resource")
        target_user_id = user_id
    else:
        # Use user_id from JWT
        if not jwt_params.us:
            raise AuthenticationException("User ID not found in query parameter or JWT")
        target_user_id = jwt_params.us
    user = await crud.get_user(db, app_id=app_id, user_id=target_user_id)
    return user


@router.get(
    "/name/{name}",
    response_model=schemas.User,
    dependencies=[
        Depends(
            require_auth(
                app_id="app_id",
            )
        )
    ],
)
async def get_user_by_name(
    app_id: str = Path(..., description="ID of the app"),
    name: str = Path(..., description="Name of the user to retrieve"),
    db=db,
):
    """Get a User by name"""
    user = await crud.get_user_by_name(db, app_id=app_id, name=name)
    return user


@router.get(
    "/get_or_create/{name}",
    response_model=schemas.User,
    dependencies=[
        Depends(
            require_auth(
                app_id="app_id",
            )
        )
    ],
)
async def get_or_create_user(
    app_id: str = Path(..., description="ID of the app"),
    name: str = Path(..., description="Name of the user to get or create"),
    db=db,
):
    """Get a User or create a new one by the input name"""
    try:
        user = await crud.get_user_by_name(db, app_id=app_id, name=name)
        return user
    except ResourceNotFoundException:
        # User doesn't exist, create it
        user = await create_user(
            db=db, app_id=app_id, user=schemas.UserCreate(name=name)
        )
        return user


@router.put(
    "/{user_id}",
    response_model=schemas.User,
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
async def update_user(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user to update"),
    user: schemas.UserUpdate = Body(..., description="Updated user parameters"),
    db=db,
):
    """Update a User's name and/or metadata"""
    updated_user = await crud.update_user(db, app_id=app_id, user_id=user_id, user=user)
    return updated_user
