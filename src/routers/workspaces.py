import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.dependencies import db
from src.deriver.enqueue import enqueue_deletion, enqueue_dream
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
    response: Response,
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

    result = await crud.get_or_create_workspace(db, workspace=workspace)
    await db.commit()
    await result.post_commit()
    response.status_code = 201 if result.created else 200
    return result.resource


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
    """Get all Workspaces, paginated with optional filters."""
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
    workspace_id: str = Path(...),
    workspace: schemas.WorkspaceUpdate = Body(
        ..., description="Updated workspace parameters"
    ),
    db: AsyncSession = db,
):
    """Update Workspace metadata and/or configuration."""
    # ResourceNotFoundException will be caught by global handler if workspace not found
    honcho_workspace = await crud.update_workspace(
        db, workspace_name=workspace_id, workspace=workspace
    )
    return honcho_workspace


@router.delete(
    "/{workspace_id}",
    status_code=202,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def delete_workspace(
    workspace_id: str = Path(...),
    db: AsyncSession = db,
):
    """
    Delete a Workspace. This accepts the deletion request and processes it in the background,
    permanently deleting all peers, messages, conclusions, and other resources associated
    with the workspace.

    Returns 409 Conflict if the workspace contains active sessions.
    Delete all sessions first, then delete the workspace.

    This action cannot be undone.
    """
    # Verify workspace exists
    await crud.get_workspace(db, workspace_name=workspace_id)

    # Check for active sessions before accepting
    await crud.check_no_active_sessions(db, workspace_name=workspace_id)

    # Enqueue for background deletion
    await enqueue_deletion(workspace_id, "workspace", workspace_id, db_session=db)
    await db.commit()

    return {"message": "Workspace deletion accepted"}


@router.post(
    "/{workspace_id}/search",
    response_model=list[schemas.Message],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def search_workspace(
    workspace_id: str = Path(...),
    body: schemas.MessageSearchOptions = Body(
        ..., description="Message search parameters"
    ),
    db: AsyncSession = db,
):
    """
    Search messages in a Workspace using optional filters. Use `limit` to control the number of
    results returned.
    """
    # take user-provided filter and add workspace_id to it
    filters = body.filters or {}
    filters["workspace_id"] = workspace_id
    return await search(db, body.query, filters=filters, limit=body.limit)


@router.get(
    "/{workspace_id}/queue/status",
    response_model=schemas.QueueStatus,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_queue_status(
    workspace_id: str = Path(...),
    observer_id: str | None = Query(
        None, description="Optional observer ID to filter by"
    ),
    sender_id: str | None = Query(None, description="Optional sender ID to filter by"),
    session_id: str | None = Query(
        None, description="Optional session ID to filter by"
    ),
    db: AsyncSession = db,
):
    """
    Get the processing queue status for a Workspace, optionally scoped to an observer, sender,
    and/or session.

    Only tracks user-facing task types (representation, summary, dream).
    Internal infrastructure tasks (reconciler, webhook, deletion) are excluded.
    Note: completed counts reflect items since the last periodic queue cleanup,
    not lifetime totals.
    """
    try:
        return await crud.get_queue_status(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            observer=observer_id,
            observed=sender_id,
        )
    except ValueError as e:
        logger.warning(f"Invalid request parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post(
    "/{workspace_id}/schedule_dream",
    status_code=204,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def schedule_dream(
    workspace_id: str = Path(...),
    request: schemas.ScheduleDreamRequest = Body(
        ..., description="Dream scheduling parameters"
    ),
    db: AsyncSession = db,
):
    """
    Manually schedule a dream task for a specific collection.

    This endpoint bypasses all automatic dream conditions (document threshold,
    minimum hours between dreams) and schedules the dream task for a future execution.

    Currently this endpoint only supports scheduling immediate dreams. In the future,
    users may pass a cron-style expression to schedule dreams at specific times.
    """
    # Check if dreams are enabled
    if not settings.DREAM.ENABLED:
        raise HTTPException(
            status_code=400,
            detail="Dreams are not enabled in the system configuration",
        )

    # Default observed to observer if not provided
    observer = request.observer
    observed = request.observed if request.observed is not None else request.observer
    dream_type = request.dream_type

    # Count documents in the collection
    count_stmt = select(func.count(models.Document.id)).where(
        models.Document.workspace_name == workspace_id,
        models.Document.observer == observer,
        models.Document.observed == observed,
    )
    document_count = int(await db.scalar(count_stmt) or 0)

    # Enqueue the dream task for immediate processing
    await enqueue_dream(
        workspace_id,
        observer=observer,
        observed=observed,
        dream_type=dream_type,
        document_count=document_count,
        session_name=request.session_id,
    )

    logger.info(
        "Manually scheduled dream: %s for %s/%s/%s (session: %s)",
        dream_type.value,
        workspace_id,
        observer,
        observed,
        request.session_id,
    )
