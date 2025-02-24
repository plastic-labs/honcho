import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy.exc import IntegrityError

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException, ConflictException
from src.security import auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps/{app_id}/users",
    tags=["users"],
    dependencies=[Depends(auth)],
)


@router.post("", response_model=schemas.User)
async def create_user(
    app_id: str,
    user: schemas.UserCreate,
    db=db,
):
    """Create a new User"""
    user_obj = await crud.create_user(db, app_id=app_id, user=user)
    return user_obj


@router.post("/list", response_model=Page[schemas.User])
async def get_users(
    app_id: str,
    options: schemas.UserGet,
    reverse: bool = False,
    db=db,
):
    """Get All Users for an App"""
    return await paginate(
        db,
        await crud.get_users(db, app_id=app_id, reverse=reverse, filter=options.filter),
    )


@router.get("/name/{name}", response_model=schemas.User)
async def get_user_by_name(
    app_id: str,
    name: str,
    db=db,
):
    """Get a User by name"""
    user = await crud.get_user_by_name(db, app_id=app_id, name=name)
    return user


@router.get("/{user_id}", response_model=schemas.User)
async def get_user(
    app_id: str,
    user_id: str,
    db=db,
):
    """Get a User by ID"""
    user = await crud.get_user(db, app_id=app_id, user_id=user_id)
    return user


@router.get("/get_or_create/{name}", response_model=schemas.User)
async def get_or_create_user(app_id: str, name: str, db=db):
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


@router.put("/{user_id}", response_model=schemas.User)
async def update_user(
    app_id: str,
    user_id: str,
    user: schemas.UserUpdate,
    db=db,
):
    """Update a User's name and/or metadata"""
    updated_user = await crud.update_user(db, app_id=app_id, user_id=user_id, user=user)
    return updated_user
