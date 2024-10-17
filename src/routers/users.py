import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy.exc import IntegrityError

from src import crud, schemas
from src.dependencies import db
from src.security import auth

router = APIRouter(
    prefix="/apps/{app_id}/users",
    tags=["users"],
)


@router.post("", response_model=schemas.User)
async def create_user(
    request: Request,
    app_id: str,
    user: schemas.UserCreate,
    db=db,
    auth=Depends(auth),
):
    """Create a User

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user (schemas.UserCreate): The User object containing any metadata

    Returns:
        schemas.User: Created User object

    """
    print("running create_user")
    try:
        return await crud.create_user(db, app_id=app_id, user=user)
    except IntegrityError as e:
        raise HTTPException(
            status_code=406, detail="User with name may already exist"
        ) from e


@router.get("", response_model=Page[schemas.User])
async def get_users(
    request: Request,
    app_id: str,
    reverse: bool = False,
    filter: Optional[str] = None,
    db=db,
    auth=Depends(auth),
):
    """Get All Users for an App

    Args:
        app_id (str): The ID of the app representing the client
        application using honcho

    Returns:
        list[schemas.User]: List of User objects

    """
    data = None
    if filter is not None:
        data = json.loads(filter)

    return await paginate(
        db, await crud.get_users(db, app_id=app_id, reverse=reverse, filter=data)
    )


@router.get("/name/{name}", response_model=schemas.User)
async def get_user_by_name(
    request: Request,
    app_id: str,
    name: str,
    db=db,
    auth=Depends(auth),
):
    """Get a User

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user

    Returns:
        schemas.User: User object

    """
    user = await crud.get_user_by_name(db, app_id=app_id, name=name)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/{user_id}", response_model=schemas.User)
async def get_user(
    request: Request,
    app_id: str,
    user_id: str,
    db=db,
    auth=Depends(auth),
):
    """Get a User

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user

    Returns:
        schemas.User: User object

    """
    user = await crud.get_user(db, app_id=app_id, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/get_or_create/{name}", response_model=schemas.User)
async def get_or_create_user(
    request: Request, app_id: str, name: str, db=db, auth=Depends(auth)
):
    """Get or Create a User

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user

    Returns:
        schemas.User: User object

    """
    user = await crud.get_user_by_name(db, app_id=app_id, name=name)
    if user is None:
        user = await create_user(
            request=request, db=db, app_id=app_id, user=schemas.UserCreate(name=name)
        )
    return user


@router.put("/{user_id}", response_model=schemas.User)
async def update_user(
    request: Request,
    app_id: str,
    user_id: str,
    user: schemas.UserUpdate,
    db=db,
    auth=Depends(auth),
):
    """Update a User

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        user (schemas.UserCreate): The User object containing any metadata

    Returns:
        schemas.User: Updated User object

    """
    try:
        return await crud.update_user(db, app_id=app_id, user_id=user_id, user=user)
    except ValueError as e:
        raise HTTPException(status_code=406, detail=str(e)) from e
