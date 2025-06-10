import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import AuthenticationException, ResourceNotFoundException
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)


@router.get("", response_model=schemas.Workspace)
async def get_workspace(
    workspace_id: Optional[str] = Query(
        None, description="Workspace ID to retrieve. If not provided, uses JWT token"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db=db,
):
    """
    Get a Workspace by ID.

    If workspace_id is provided as a query parameter, it uses that (must match JWT workspace_id).
    Otherwise, it uses the workspace_id from the JWT token.
    """
    # If workspace_id provided in query, check if it matches jwt or user is admin
    if workspace_id:
        if not jwt_params.ad and jwt_params.ap != workspace_id:
            raise AuthenticationException("Unauthorized access to resource")
        target_workspace_id = workspace_id
    else:
        # Use workspace_id from JWT
        if not jwt_params.ap:
            raise AuthenticationException("Workspace ID not found in query parameter or JWT")
        target_workspace_id = jwt_params.ap

    return await crud.get_workspace(db, workspace_id=target_workspace_id)


@router.post(
    "/list",
    response_model=Page[schemas.Workspace],
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_all_workspaces(
    options: Optional[schemas.WorkspaceGet] = Body(
        None, description="Filtering and pagination options for the workspaces list"
    ),
    reverse: Optional[bool] = Query(
        False, description="Whether to reverse the order of results"
    ),
    db=db,
):
    """Get all Workspaces"""
    filter_param = None
    if options and hasattr(options, "filter"):
        filter_param = options.filter
        if filter_param == {}:
            filter_param = None

    return await paginate(
        db,
        await crud.get_all_workspaces(
            reverse=reverse,
            filter=filter_param,
        ),
    )


@router.get(
    "/name/{name}",
    response_model=schemas.Workspace,
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_workspace_by_name(
    name: str = Path(..., description="Name of the workspace to retrieve"),
    db=db,
):
    """Get a Workspace by Name"""
    # ResourceNotFoundException will be caught by global handler if workspace not found
    workspace = await crud.get_workspace_by_name(db, name=name)
    return workspace


@router.post(
    "", response_model=schemas.Workspace, dependencies=[Depends(require_auth(admin=True))]
)
async def create_workspace(
    workspace: schemas.WorkspaceCreate = Body(..., description="Workspace creation parameters"),
    db=db,
):
    """Create a new Workspace"""
    honcho_workspace = await crud.create_workspace(db, workspace=workspace)
    return honcho_workspace


@router.get(
    "/get_or_create/{name}",
    response_model=schemas.Workspace,
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_or_create_workspace(
    name: str = Path(..., description="Name of the workspace to get or create"),
    db=db,
):
    """Get or Create a Workspace"""
    try:
        workspace = await crud.get_workspace_by_name(db=db, name=name)
        return workspace
    except ResourceNotFoundException:
        # Workspace doesn't exist, create it
        workspace = await create_workspace(db=db, workspace=schemas.WorkspaceCreate(name=name))
        return workspace


@router.put(
    "/{workspace_id}",
    response_model=schemas.Workspace,
    dependencies=[Depends(require_auth(app_id="workspace_id"))],
)
async def update_workspace(
    workspace_id: str = Path(..., description="ID of the workspace to update"),
    workspace: schemas.WorkspaceUpdate = Body(..., description="Updated workspace parameters"),
    db=db,
):
    """Update a Workspace"""
    # ResourceNotFoundException will be caught by global handler if workspace not found
    honcho_workspace = await crud.update_workspace(db, workspace_id=workspace_id, workspace=workspace)
    return honcho_workspace
