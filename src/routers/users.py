from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy.exc import IntegrityError

from src import crud, schemas
from src.dependencies import db
from src.security import auth

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
    print("running create_user")
    try:
        return await crud.create_user(db, app_id=app_id, user=user)
    except IntegrityError as e:
        raise HTTPException(
            status_code=406, detail="User with name may already exist"
        ) from e


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
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/{user_id}", response_model=schemas.User)
async def get_user(
    app_id: str,
    user_id: str,
    db=db,
):
    """Get a User by ID"""
    user = await crud.get_user(db, app_id=app_id, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/get_or_create/{name}", response_model=schemas.User)
async def get_or_create_user(app_id: str, name: str, db=db):
    """Get a User or create a new one by the input name"""
    user = await crud.get_user_by_name(db, app_id=app_id, name=name)
    if user is None:
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
    try:
        return await crud.update_user(db, app_id=app_id, user_id=user_id, user=user)
    except ValueError as e:
        raise HTTPException(status_code=406, detail=str(e)) from e
