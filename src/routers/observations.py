import logging
from collections.abc import Sequence

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/sessions/{session_id}/observations",
    tags=["observations"],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)


@router.post("/list", response_model=Page[schemas.Observation])
async def list_observations(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    options: schemas.ObservationGet | None = Body(
        None, description="Filtering options for the observations list"
    ),
    reverse: bool | None = Query(
        False, description="Whether to reverse the order of results"
    ),
    db: AsyncSession = db,
):
    """
    List all observations for a session.

    Returns paginated observations (documents) associated with this session.
    Observations can be filtered by observer_id and observed_id using the filters parameter.
    """
    try:
        filters = None
        if options and hasattr(options, "filters"):
            filters = options.filters
            if filters == {}:
                filters = None

        # Query all documents for this session
        stmt = (
            select(models.Document)
            .where(models.Document.workspace_name == workspace_id)
            .where(models.Document.session_name == session_id)
        )

        # Apply additional filters if provided
        if filters:
            from src.utils.filter import apply_filter

            stmt = apply_filter(stmt, models.Document, filters)

        # Order by created_at (newest first by default)
        if reverse:
            stmt = stmt.order_by(models.Document.created_at.asc())
        else:
            stmt = stmt.order_by(models.Document.created_at.desc())

        return await apaginate(db, stmt)
    except ValueError as e:
        logger.warning(f"Failed to get observations for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.post("/query", response_model=list[schemas.Observation])
async def query_observations(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    body: schemas.ObservationQuery = Body(
        ..., description="Semantic search parameters for observations"
    ),
    db: AsyncSession = db,
) -> Sequence[models.Document]:
    """
    Query observations using semantic search.

    Performs vector similarity search on observations to find semantically relevant results.
    If observer_id and observed_id are provided in filters, only observations matching
    those criteria will be searched. Otherwise, all observations for the session are searched.
    """
    try:
        # Extract observer and observed from filters if provided
        observer = None
        observed = None
        if body.filters:
            observer = body.filters.get("observer") or body.filters.get("observer_id")
            observed = body.filters.get("observed") or body.filters.get("observed_id")

        # If no observer/observed specified, we need to query across all session documents
        # For now, we'll require these to be specified for semantic search
        if not observer or not observed:
            raise ValidationException(
                "observer and observed must be specified for semantic search"
            )
        else:
            # Query specific observer/observed pair
            results = await crud.query_documents(
                db,
                workspace_name=workspace_id,
                query=body.query,
                observer=observer,
                observed=observed,
                filters=body.filters,
                max_distance=body.distance,
                top_k=body.top_k,
            )
            return results
    except ValueError as e:
        logger.warning(
            f"Failed to query observations for session {session_id}: {str(e)}"
        )
        raise ResourceNotFoundException("Session not found") from e


@router.delete("/{observation_id}")
async def delete_observation(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    observation_id: str = Path(..., description="ID of the observation to delete"),
    db: AsyncSession = db,
):
    """
    Delete a specific observation.

    This permanently deletes the observation (document) from the theory-of-mind system.
    This action cannot be undone.
    """
    try:
        # We need to find the document first to get observer/observed
        stmt = (
            select(models.Document)
            .where(models.Document.id == observation_id)
            .where(models.Document.workspace_name == workspace_id)
            .where(models.Document.session_name == session_id)
        )
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()

        if document is None:
            raise ResourceNotFoundException(
                f"Observation {observation_id} not found in session {session_id}"
            )

        # Now delete it using the CRUD function
        await crud.delete_document(
            db,
            workspace_name=workspace_id,
            document_id=observation_id,
            observer=document.observer,
            observed=document.observed,
            session_name=session_id,
        )

        logger.debug("Observation %s deleted successfully", observation_id)
        return {"message": "Observation deleted successfully"}
    except ResourceNotFoundException:
        raise
    except ValueError as e:
        logger.warning(f"Failed to delete observation {observation_id}: {str(e)}")
        raise ResourceNotFoundException("Observation not found") from e
