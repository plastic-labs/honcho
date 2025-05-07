import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import AuthenticationException, ResourceNotFoundException
from src.routers import transactions
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps",
    tags=["apps"],
)


@router.get("", response_model=schemas.App)
async def get_app(
    app_id: Optional[str] = Query(
        None, description="App ID to retrieve. If not provided, uses JWT token"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """
    Get an App by ID.

    If app_id is provided as a query parameter, it uses that (must match JWT app_id).
    Otherwise, it uses the app_id from the JWT token.
    """
    # If app_id provided in query, check if it matches jwt or user is admin
    if app_id:
        if not jwt_params.ad and jwt_params.ap != app_id:
            raise AuthenticationException("Unauthorized access to resource")
        target_app_id = app_id
    else:
        # Use app_id from JWT
        if not jwt_params.ap:
            raise AuthenticationException("App ID not found in query parameter or JWT")
        target_app_id = jwt_params.ap

    return await crud.get_app(db, app_id=target_app_id, transaction_id=transaction_id)


@router.post(
    "/list",
    response_model=Page[schemas.App],
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_all_apps(
    options: schemas.AppGet = Body(
        ..., description="Filtering and pagination options for the apps list"
    ),
    reverse: Optional[bool] = Query(
        False, description="Whether to reverse the order of results"
    ),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """Get all Apps"""
    return await paginate(
        db,
        await crud.get_all_apps(
            db,
            reverse=reverse,
            filter=options.filter,
            transaction_id=transaction_id,
        ),
    )


@router.get(
    "/name/{name}",
    response_model=schemas.App,
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_app_by_name(
    name: str = Path(..., description="Name of the app to retrieve"),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """Get an App by Name"""
    # ResourceNotFoundException will be caught by global handler if app not found
    app = await crud.get_app_by_name(db, name=name, transaction_id=transaction_id)
    return app


@router.post(
    "", response_model=schemas.App, dependencies=[Depends(require_auth(admin=True))]
)
async def create_app(
    app: schemas.AppCreate = Body(..., description="App creation parameters"),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """Create a new App"""
    honcho_app = await crud.create_app(db, app=app, transaction_id=transaction_id)
    return honcho_app


@router.get(
    "/get_or_create/{name}",
    response_model=schemas.App,
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_or_create_app(
    name: str = Path(..., description="Name of the app to get or create"),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """Get or Create an App"""
    try:
        app = await crud.get_app_by_name(db, name=name, transaction_id=transaction_id)
        return app
    except ResourceNotFoundException:
        # App doesn't exist, create it
        app = await create_app(
            db=db,
            app=schemas.AppCreate(name=name),
            transaction_id=transaction_id,
        )
        return app


@router.put(
    "/{app_id}",
    response_model=schemas.App,
    dependencies=[Depends(require_auth(app_id="app_id"))],
)
async def update_app(
    app_id: str = Path(..., description="ID of the app to update"),
    app: schemas.AppUpdate = Body(..., description="Updated app parameters"),
    transaction_id: int | None = Depends(transactions.get_transaction_id),
    db=db,
):
    """Update an App"""
    # ResourceNotFoundException will be caught by global handler if app not found
    honcho_app = await crud.update_app(
        db, app_id=app_id, app=app, transaction_id=transaction_id
    )
    return honcho_app
