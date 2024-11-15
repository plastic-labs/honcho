import traceback

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError

from src import crud, schemas
from src.dependencies import db
from src.security import auth

router = APIRouter(
    prefix="/apps",
    tags=["apps"],
    dependencies=[Depends(auth)],
)


@router.get("/{app_id}", response_model=schemas.App)
async def get_app(app_id: str, db=db):
    """Get an App by ID"""
    app = await crud.get_app(db, app_id=app_id)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


@router.get("/name/{name}", response_model=schemas.App)
async def get_app_by_name(name: str, db=db):
    """Get an App by Name"""
    app = await crud.get_app_by_name(db, name=name)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


@router.post("", response_model=schemas.App)
async def create_app(app: schemas.AppCreate, db=db):
    """Create a new App"""
    try:
        honcho_app = await crud.create_app(db, app=app)
        return honcho_app
    except IntegrityError as e:
        raise HTTPException(
            status_code=406, detail="App with name may already exist"
        ) from e
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail="Unknown Error") from e


@router.get("/get_or_create/{name}", response_model=schemas.App)
async def get_or_create_app(name: str, db=db):
    """Get or Create an App"""
    print("name", name)
    app = await crud.get_app_by_name(db=db, name=name)
    if app is None:
        app = await create_app(db=db, app=schemas.AppCreate(name=name))
    return app


@router.put("/{app_id}", response_model=schemas.App)
async def update_app(
    app_id: str,
    app: schemas.AppUpdate,
    db=db,
):
    """Update an App"""
    honcho_app = await crud.update_app(db, app_id=app_id, app=app)
    if honcho_app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return honcho_app
