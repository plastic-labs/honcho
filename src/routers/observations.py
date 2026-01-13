import logging

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/observations",
    tags=["observations"],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)


@router.post(
    "",
    response_model=list[schemas.Observation],
    deprecated=True,
)
async def create_observations(
    workspace_id: str = Path(..., description="ID of the workspace"),
    body: schemas.ObservationBatchCreate = Body(
        ..., description="Batch of observations to create"
    ),
    db: AsyncSession = db,
) -> list[schemas.Observation]:
    """
    Create one or more observations.

    Creates observations (theory-of-mind facts) for the specified observer/observed peer pairs.
    Each observation must reference existing peers and a session within the workspace.
    Embeddings are automatically generated for semantic search.

    Maximum of 100 observations per request.
    """
    documents = await crud.create_observations(
        db,
        observations=body.conclusions,
        workspace_name=workspace_id,
    )

    logger.debug(
        "Created %d observations in workspace %s",
        len(documents),
        workspace_id,
    )
    return [schemas.Observation.model_validate(doc) for doc in documents]


@router.post(
    "/list",
    response_model=Page[schemas.Observation],
    deprecated=True,
)
async def list_observations(
    workspace_id: str = Path(..., description="ID of the workspace"),
    options: schemas.ObservationGet | None = Body(
        None, description="Filtering options for the observations list"
    ),
    reverse: bool | None = Query(
        False, description="Whether to reverse the order of results"
    ),
    db: AsyncSession = db,
):
    """
    List all observations using custom filters. Observations are listed by recency unless `reverse` is set to `true`.

    Observations can be filtered by session_id, observer_id and observed_id using the filters parameter.
    """
    try:
        filters = None
        if options and hasattr(options, "filters"):
            filters = options.filters
            if filters == {}:
                filters = None

        stmt = crud.get_documents_with_filters(
            workspace_name=workspace_id,
            filters=filters,
            reverse=reverse or False,
        )

        return await apaginate(db, stmt)
    except ValueError as e:
        logger.warning(f"Failed to list observations: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.post(
    "/query",
    response_model=list[schemas.Observation],
    deprecated=True,
)
async def query_observations(
    workspace_id: str = Path(..., description="ID of the workspace"),
    body: schemas.ObservationQuery = Body(
        ..., description="Semantic search parameters for observations"
    ),
    db: AsyncSession = db,
) -> list[schemas.Observation]:
    """
    Query observations using semantic search.

    Performs vector similarity search on observations to find semantically relevant results.
    Observer and observed are required for semantic search and must be provided in filters.
    """
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
        documents = await crud.query_documents(
            db,
            workspace_name=workspace_id,
            query=body.query,
            observer=observer,
            observed=observed,
            filters=body.filters,
            max_distance=body.distance,
            top_k=body.top_k,
        )
        return [schemas.Observation.model_validate(doc) for doc in documents]


@router.delete(
    "/{observation_id}",
    deprecated=True,
)
async def delete_observation(
    workspace_id: str = Path(..., description="ID of the workspace"),
    observation_id: str = Path(..., description="ID of the observation to delete"),
    db: AsyncSession = db,
):
    """
    Delete a specific observation.

    This permanently deletes the observation (document) from the theory-of-mind system.
    This action cannot be undone.
    """
    try:
        await crud.delete_document_by_id(
            db,
            workspace_name=workspace_id,
            document_id=observation_id,
        )

        logger.debug("Observation %s deleted successfully", observation_id)
        return {"message": "Observation deleted successfully"}
    except ResourceNotFoundException:
        raise
    except ValueError as e:
        logger.warning(f"Failed to delete observation {observation_id}: {str(e)}")
        raise ResourceNotFoundException("Observation not found") from e
