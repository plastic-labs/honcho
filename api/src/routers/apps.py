import os
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db

router = APIRouter(
    prefix="/apps",
    tags=["apps"],
)


@router.get("/{app_id}", response_model=schemas.App)
async def get_app(request: Request, app_id: uuid.UUID, db=db):
    """Get an App by ID

    Args:
        app_id (uuid.UUID): The ID of the app

    Returns:
        schemas.App: App object

    """
    app = await crud.get_app(db, app_id=app_id)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


@router.get("/name/{name}", response_model=schemas.App)
async def get_app_by_name(request: Request, name: str, db=db):
    """Get an App by Name

    Args:
        app_name (str): The name of the app

    Returns:
        schemas.App: App object

    """
    app = await crud.get_app_by_name(db, name=name)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


@router.post("", response_model=schemas.App)
async def create_app(request: Request, app: schemas.AppCreate, db=db):
    """Create an App

    Args:
        app (schemas.AppCreate): The App object containing any metadata

    Returns:
        schemas.App: Created App object

    """
    USE_AUTH_SERVICE = os.getenv("USE_AUTH_SERVICE", "False").lower() == "true"
    if USE_AUTH_SERVICE:
        AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
        authorization: Optional[str] = request.headers.get("Authorization")
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if token is not None:
                honcho_app = await crud.create_app(db, app=app)
                if token == "default":
                    return honcho_app
                res = httpx.put(
                    f"{AUTH_SERVICE_URL}/organizations",
                    json={
                        "id": str(honcho_app.id),
                        "name": honcho_app.name,
                        "token": token,
                    },
                )
                data = res.json()
                if data:
                    return honcho_app
    else:
        honcho_app = await crud.create_app(db, app=app)
        return honcho_app


@router.get("/get_or_create/{name}", response_model=schemas.App)
async def get_or_create_app(request: Request, name: str, db=db):
    """Get or Create an App

    Args:
        app_name (str): The name of the app

    Returns:
        schemas.App: App object

    """
    print("name", name)
    app = await crud.get_app_by_name(db=db, name=name)
    if app is None:
        app = await create_app(request=request, db=db, app=schemas.AppCreate(name=name))
    return app


@router.put("/{app_id}", response_model=schemas.App)
async def update_app(
    request: Request, app_id: uuid.UUID, app: schemas.AppUpdate, db=db
):
    """Update an App

    Args:
        app_id (uuid.UUID): The ID of the app to update
        app (schemas.AppUpdate): The App object containing any new metadata

    Returns:
        schemas.App: The App object of the updated App

    """
    honcho_app = await crud.update_app(db, app_id=app_id, app=app)
    if honcho_app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return honcho_app
