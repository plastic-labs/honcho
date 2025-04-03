import logging

from fastapi import APIRouter, Depends

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException
from src.security import auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps",
    tags=["apps"],
    dependencies=[Depends(auth)],
)


@router.get("/{app_id}", response_model=schemas.App)
async def get_app(app_id: str, db=db):
    """Get an App by ID"""
    # ResourceNotFoundException will be caught by global handler if app not found
    app = await crud.get_app(db, app_id=app_id)
    return app


@router.get("/name/{name}", response_model=schemas.App)
async def get_app_by_name(name: str, db=db):
    """Get an App by Name"""
    # ResourceNotFoundException will be caught by global handler if app not found
    app = await crud.get_app_by_name(db, name=name)
    return app


@router.post("", response_model=schemas.App)
async def create_app(app: schemas.AppCreate, db=db):
    """Create a new App"""
    honcho_app = await crud.create_app(db, app=app)
    return honcho_app


@router.get("/get_or_create/{name}", response_model=schemas.App)
async def get_or_create_app(name: str, db=db):
    """Get or Create an App"""
    try:
        app = await crud.get_app_by_name(db=db, name=name)
        return app
    except ResourceNotFoundException:
        # App doesn't exist, create it
        app = await create_app(db=db, app=schemas.AppCreate(name=name))
        return app


@router.put("/{app_id}", response_model=schemas.App)
async def update_app(
    app_id: str,
    app: schemas.AppUpdate,
    db=db,
):
    """Update an App"""
    # ResourceNotFoundException will be caught by global handler if app not found
    honcho_app = await crud.update_app(db, app_id=app_id, app=app)
    return honcho_app
