import logging

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.api import create_page
from fastapi_pagination.bases import AbstractParams
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.dependencies import db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/conclusions",
    tags=["conclusions"],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)


def _merge_conclusion_memory_filters(
    filters: dict[str, object] | None,
    *,
    memory_domains: list[str] | None = None,
    memory_horizons: list[str] | None = None,
    memory_thesis_kinds: list[str] | None = None,
) -> dict[str, object] | None:
    merged = dict(filters or {})

    if memory_domains:
        merged["metadata.memory.domain"] = {"in": memory_domains}
    if memory_horizons:
        merged["metadata.memory.horizon"] = {"in": memory_horizons}
    if memory_thesis_kinds:
        merged["metadata.memory.thesis_kind"] = {"in": memory_thesis_kinds}

    return merged or None


def _normalize_memory_lifecycle(document: schemas.Conclusion) -> schemas.Conclusion:
    memory = document.memory if isinstance(document.memory, dict) else None
    if not memory:
        return document

    normalized = crud.normalize_memory_taxonomy(memory)
    if normalized is not None:
        document.memory = normalized
    return document


def _page_params_from_page(page: Page[schemas.Conclusion]) -> AbstractParams:
    return Page[schemas.Conclusion].__params_type__(
        page=page.page,
        size=page.size,
    )


@router.post(
    "",
    response_model=list[schemas.Conclusion],
    status_code=201,
)
async def create_conclusions(
    workspace_id: str = Path(...),
    body: schemas.ConclusionBatchCreate = Body(
        ...,
        description="Batch of Conclusions to create",
    ),
    db: AsyncSession = db,
) -> list[schemas.Conclusion]:
    """
    Create one or more Conclusions.

    Conclusions are logical certainties derived from interactions between Peers. They form the basis of a Peer's Representation.
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
    workspace_id: str = Path(...),
    options: schemas.ConclusionGet | None = Body(
        None,
        description="Filtering options for the Conclusions list",
    ),
    reverse: bool | None = Query(
        False,
        description="Whether to reverse the order of results",
    ),
    db: AsyncSession = db,
):
    """
    List Conclusions using optional filters, ordered by recency unless `reverse` is true. Results are paginated.
    """
    filters = None
    if options and hasattr(options, "filters"):
        filters = options.filters
        if filters == {}:
            filters = None

    if options:
        filters = _merge_conclusion_memory_filters(
            filters,
            memory_domains=options.memory_domains,
            memory_horizons=options.memory_horizons,
            memory_thesis_kinds=options.memory_thesis_kinds,
        )

    stmt = crud.get_documents_with_filters(
        workspace_name=workspace_id,
        filters=filters,
        reverse=reverse or False,
    )

    page = await apaginate(db, stmt)
    items = [
        _normalize_memory_lifecycle(schemas.Conclusion.model_validate(item))
        for item in page.items
    ]
    if options and options.exclude_expired:
        items = [
            item
            for item in items
            if not crud.is_document_memory_expired(
                models.Document(internal_metadata={"memory": item.memory or {}})
            )
        ]
        return create_page(
            items,
            total=len(items),
            params=_page_params_from_page(page),
        )
    page.items = items
    return page


@router.post(
    "/query",
    response_model=list[schemas.Conclusion],
)
async def query_conclusions(
    workspace_id: str = Path(...),
    body: schemas.ConclusionQuery = Body(
        ...,
        description="Semantic search parameters for Conclusions",
    ),
    db: AsyncSession = db,
) -> list[schemas.Conclusion]:
    """
    Query Conclusions using semantic search. Use `top_k` to control the number of results returned.
    """
    observer = None
    observed = None
    filters = _merge_conclusion_memory_filters(
        body.filters,
        memory_domains=body.memory_domains,
        memory_horizons=body.memory_horizons,
        memory_thesis_kinds=body.memory_thesis_kinds,
    )
    if filters:
        observer = filters.get("observer") or filters.get("observer_id")
        observed = filters.get("observed") or filters.get("observed_id")

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
        filters=filters,
        max_distance=body.distance,
        top_k=body.top_k,
        exclude_expired=body.exclude_expired,
    )
    return [
        _normalize_memory_lifecycle(schemas.Conclusion.model_validate(doc))
        for doc in documents
    ]


@router.delete(
    "/{conclusion_id}",
    status_code=204,
    response_model=None,
)
async def delete_conclusion(
    workspace_id: str = Path(...),
    conclusion_id: str = Path(...),
    db: AsyncSession = db,
):
    """
    Delete a single Conclusion by ID.

    This action cannot be undone.
    """
    try:
        await crud.delete_document_by_id(
            db,
            workspace_name=workspace_id,
            document_id=conclusion_id,
        )

        logger.debug("Conclusion %s deleted successfully", conclusion_id)
    except ResourceNotFoundException:
        raise
    except ValueError as e:
        logger.warning(f"Failed to delete conclusion {conclusion_id}: {str(e)}")
        raise ResourceNotFoundException("Conclusion not found") from e
