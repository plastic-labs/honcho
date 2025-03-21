import logging

from fastapi import APIRouter, Depends

from src import crud, schemas
from src.dependencies import db
from src.exceptions import AuthenticationException, ResourceNotFoundException
from src.security import JWTParams, auth, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps",
    tags=["apps"],
)


@router.get(
    "/",
    response_model=schemas.App,
    # include_in_schema=False,  XX can use this if desired to skip docs
)
async def get_app_from_token(jwt_params: JWTParams = Depends(auth), db=db):
    """Get an App by ID from the app_id provided in the JWT.
    If no app_id is provided, return a 404.
    """
    if jwt_params.ap is None:
        raise AuthenticationException("App not found in JWT")
    return await crud.get_app(db, app_id=jwt_params.ap)


@router.get(
    "/{app_id}",
    response_model=schemas.App,
    dependencies=[Depends(require_auth(app_id="app_id"))],
)
async def get_app(app_id: str, db=db):
    """Get an App by ID"""
    app = await crud.get_app(db, app_id=app_id)
    return app


@router.get(
    "/name/{name}",
    response_model=schemas.App,
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_app_by_name(name: str, db=db):
    """Get an App by Name"""
    # ResourceNotFoundException will be caught by global handler if app not found
    app = await crud.get_app_by_name(db, name=name)
    return app


@router.post(
    "", response_model=schemas.App, dependencies=[Depends(require_auth(admin=True))]
)
async def create_app(app: schemas.AppCreate, db=db):
    """Create a new App"""
    honcho_app = await crud.create_app(db, app=app)
    return honcho_app


@router.get(
    "/get_or_create/{name}",
    response_model=schemas.App,
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_or_create_app(name: str, db=db):
    """Get or Create an App"""
    try:
        app = await crud.get_app_by_name(db=db, name=name)
        return app
    except ResourceNotFoundException:
        # App doesn't exist, create it
        app = await create_app(db=db, app=schemas.AppCreate(name=name))
        return app


@router.put(
    "/{app_id}",
    response_model=schemas.App,
    dependencies=[Depends(require_auth(app_id="app_id"))],
)
async def update_app(
    app_id: str,
    app: schemas.AppUpdate,
    db=db,
):
    """Update an App"""
    # ResourceNotFoundException will be caught by global handler if app not found
    honcho_app = await crud.update_app(db, app_id=app_id, app=app)
    return honcho_app
