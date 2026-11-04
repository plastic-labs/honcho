"""Representation History: the changes each dream made to a representation."""

from fastapi import APIRouter, Body, Depends, Path
from fastapi_pagination.bases import AbstractParams
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import read_db
from src.security import require_auth
from src.utils.pagination import (
    CursorPage,
    Page,
    paginate_offset_or_cursor,
    pagination_params,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/dreams",
    tags=["dreams"],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)


@router.post(
    "/list",
    response_model=Page[schemas.DreamRun] | CursorPage[schemas.DreamRun],
)
async def list_dreams(
    workspace_id: str = Path(...),
    options: schemas.DreamRunList | None = Body(
        None, description="Filtering options for the dream run list"
    ),
    params: AbstractParams = Depends(pagination_params),
    db: AsyncSession = read_db,
):
    """
    List dream runs, newest first. Results are paginated.

    Each run records what one dream changed in a representation. Filter by
    `observed_id` for a peer's history, and add `observer_id` for a single
    representation. Runs are kept for the server's retention window (30 days
    by default).
    """
    stmt = crud.list_runs_stmt(
        workspace_id,
        observed=options.observed_id if options else None,
        observer=options.observer_id if options else None,
    )
    return await paginate_offset_or_cursor(db, stmt, params)


@router.get("/{run_id}", response_model=schemas.DreamRunDetail)
async def get_dream(
    workspace_id: str = Path(...),
    run_id: str = Path(..., description="The dream's run ID"),
    db: AsyncSession = read_db,
) -> schemas.DreamRunDetail:
    """
    Get one dream run with every change it committed.

    The response lists the conclusions the dream created with the conclusions
    each was derived from, the conclusions it removed as they read when
    removed, and the peer card before and after.
    """
    return await crud.get_run_detail(db, workspace_id, run_id)
