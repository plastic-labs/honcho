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
    prefix="/workspaces/{workspace_id}/conclusions",
    tags=["conclusions"],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)


@router.post(
    "",
    response_model=list[schemas.Conclusion],
)
async def create_conclusions(
    workspace_id: str = Path(..., description="ID of the workspace"),
    body: schemas.ConclusionBatchCreate = Body(
        ...,
        description="Batch of conclusions to create",
    ),
    db: AsyncSession = db,
) -> list[schemas.Conclusion]:
    """
    Create one or more conclusions.

    Conclusions are theory-of-mind facts derived from interactions between peers.
    """
    documents = await crud.create_observations(
        db,
        observations=body.conclusions,
        workspace_name=workspace_id,
    )

    logger.debug(
        "Created %d conclusions in workspace %s",
        len(documents),
        workspace_id,
    )
    return [schemas.Conclusion.model_validate(doc) for doc in documents]


@router.post(
    "/list",
    response_model=Page[schemas.Conclusion],
)
async def list_conclusions(
    workspace_id: str = Path(..., description="ID of the workspace"),
    options: schemas.ConclusionGet | None = Body(
        None,
        description="Filtering options for the conclusions list",
    ),
    reverse: bool | None = Query(
        False,
        description="Whether to reverse the order of results",
    ),
    db: AsyncSession = db,
):
    """
    List conclusions using custom filters, ordered by recency unless `reverse` is true.
    """
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


@router.post(
    "/query",
    response_model=list[schemas.Conclusion],
)
async def query_conclusions(
    workspace_id: str = Path(..., description="ID of the workspace"),
    body: schemas.ConclusionQuery = Body(
        ...,
        description="Semantic search parameters for conclusions",
    ),
    db: AsyncSession = db,
) -> list[schemas.Conclusion]:
    """
    Query conclusions using semantic search.
    """
    observer = None
    observed = None
    if body.filters:
        observer = body.filters.get("observer") or body.filters.get("observer_id")
        observed = body.filters.get("observed") or body.filters.get("observed_id")

    if not observer or not observed:
        raise ValidationException(
            "observer and observed must be specified for semantic search"
        )

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
    return [schemas.Conclusion.model_validate(doc) for doc in documents]


@router.delete(
    "/{conclusion_id}",
)
async def delete_conclusion(
    workspace_id: str = Path(..., description="ID of the workspace"),
    conclusion_id: str = Path(..., description="ID of the conclusion to delete"),
    db: AsyncSession = db,
):
    """
    Delete a specific conclusion (document).
    """
    try:
        await crud.delete_document_by_id(
            db,
            workspace_name=workspace_id,
            document_id=conclusion_id,
        )

        logger.debug("Conclusion %s deleted successfully", conclusion_id)
        return {"message": "Conclusion deleted successfully"}
    except ResourceNotFoundException:
        raise
    except ValueError as e:
        logger.warning(f"Failed to delete conclusion {conclusion_id}: {str(e)}")
        raise ResourceNotFoundException("Conclusion not found") from e
