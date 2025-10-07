import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db
from src.exceptions import AuthenticationException
from src.security import JWTParams, require_auth
from src.utils.search import search

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)


@router.post("", response_model=schemas.Workspace)
async def get_or_create_workspace(
    workspace: schemas.WorkspaceCreate = Body(
        ..., description="Workspace creation parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Get a Workspace by ID.

    If workspace_id is provided as a query parameter, it uses that (must match JWT workspace_id).
    Otherwise, it uses the workspace_id from the JWT.
    """
    # If workspace_id provided in query, check if it matches jwt or user is admin
    if workspace.name:
        if not jwt_params.ad and jwt_params.w != workspace.name:
            raise AuthenticationException("Unauthorized access to resource")
    else:
        # Use workspace_id from JWT
        if not jwt_params.w:
            raise AuthenticationException(
                "Workspace ID not found in query parameter or JWT"
            )
        workspace.name = jwt_params.w

    return await crud.get_or_create_workspace(db, workspace=workspace)


@router.post(
    "/list",
    response_model=Page[schemas.Workspace],
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_all_workspaces(
    options: schemas.WorkspaceGet | None = Body(
        None, description="Filtering and pagination options for the workspaces list"
    ),
    db: AsyncSession = db,
):
    """Get all Workspaces"""
    filter_param = None
    if options and hasattr(options, "filters"):
        filter_param = options.filters
        if filter_param == {}:
            filter_param = None

    return await apaginate(
        db,
        await crud.get_all_workspaces(filters=filter_param),
    )


@router.put(
    "/{workspace_id}",
    response_model=schemas.Workspace,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def update_workspace(
    workspace_id: str = Path(..., description="ID of the workspace to update"),
    workspace: schemas.WorkspaceUpdate = Body(
        ..., description="Updated workspace parameters"
    ),
    db: AsyncSession = db,
):
    """Update a Workspace"""
    # ResourceNotFoundException will be caught by global handler if workspace not found
    honcho_workspace = await crud.update_workspace(
        db, workspace_name=workspace_id, workspace=workspace
    )
    return honcho_workspace


@router.post(
    "/{workspace_id}/search",
    response_model=list[schemas.Message],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def search_workspace(
    workspace_id: str = Path(..., description="ID of the workspace to search"),
    body: schemas.MessageSearchOptions = Body(
        ..., description="Message search parameters "
    ),
    db: AsyncSession = db,
):
    """Search a Workspace"""
    # take user-provided filter and add workspace_id to it
    filters = body.filters or {}
    filters["workspace_id"] = workspace_id
    return await search(db, body.query, filters=filters, limit=body.limit)


@router.get(
    "/{workspace_id}/deriver/status",
    response_model=schemas.DeriverStatus,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_deriver_status(
    workspace_id: str = Path(..., description="ID of the workspace"),
    observer_id: str | None = Query(
        None, description="Optional observer ID to filter by"
    ),
    sender_id: str | None = Query(None, description="Optional sender ID to filter by"),
    session_id: str | None = Query(
        None, description="Optional session ID to filter by"
    ),
    db: AsyncSession = db,
):
    """Get the deriver processing status, optionally scoped to an observer, sender, and/or session"""
    try:
        return await crud.get_deriver_status(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            observer=observer_id,
            observed=sender_id,
        )
    except ValueError as e:
        logger.warning(f"Invalid request parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
