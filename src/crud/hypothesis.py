"""CRUD operations for Hypothesis model."""

from logging import getLogger
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.workspace import get_or_create_workspace
from src.exceptions import ResourceNotFoundException
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def create_hypothesis(
    db: AsyncSession,
    hypothesis: schemas.HypothesisCreate,
    workspace_name: str,
) -> models.Hypothesis:
    """
    Create a new hypothesis.

    Args:
        db: Database session
        hypothesis: Hypothesis creation schema
        workspace_name: Name of the workspace

    Returns:
        Created hypothesis object

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    # Ensure workspace exists
    await get_or_create_workspace(db, schemas.WorkspaceCreate(name=workspace_name))

    # Create hypothesis object
    hypothesis_obj = models.Hypothesis(
        content=hypothesis.content,
        observer=hypothesis.observer,
        observed=hypothesis.observed,
        status=hypothesis.status or "active",
        confidence=hypothesis.confidence if hypothesis.confidence is not None else 0.5,
        source_premise_ids=hypothesis.source_premise_ids,
        unaccounted_premises_count=hypothesis.unaccounted_premises_count or 0,
        search_coverage=hypothesis.search_coverage or 0,
        tier=hypothesis.tier or 0,
        reasoning_metadata=hypothesis.reasoning_metadata or {},
        workspace_name=workspace_name,
        collection_id=hypothesis.collection_id,
    )

    db.add(hypothesis_obj)
    await db.commit()
    await db.refresh(hypothesis_obj)

    logger.debug(
        "Hypothesis %s created successfully in workspace %s",
        hypothesis_obj.id,
        workspace_name,
    )
    return hypothesis_obj


async def get_hypothesis(
    db: AsyncSession,
    workspace_name: str,
    hypothesis_id: str,
) -> models.Hypothesis:
    """
    Get a hypothesis by ID.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        hypothesis_id: ID of the hypothesis

    Returns:
        The hypothesis if found

    Raises:
        ResourceNotFoundException: If the hypothesis does not exist
    """
    stmt = (
        select(models.Hypothesis)
        .where(models.Hypothesis.workspace_name == workspace_name)
        .where(models.Hypothesis.id == hypothesis_id)
    )
    result = await db.execute(stmt)
    hypothesis = result.scalar_one_or_none()

    if hypothesis is None:
        raise ResourceNotFoundException(
            f"Hypothesis {hypothesis_id} not found in workspace {workspace_name}"
        )

    return hypothesis


async def update_hypothesis(
    db: AsyncSession,
    workspace_name: str,
    hypothesis_id: str,
    hypothesis_update: schemas.HypothesisUpdate,
) -> models.Hypothesis:
    """
    Update a hypothesis.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        hypothesis_id: ID of the hypothesis to update
        hypothesis_update: Hypothesis update schema

    Returns:
        The updated hypothesis

    Raises:
        ResourceNotFoundException: If the hypothesis does not exist
    """
    # Get existing hypothesis
    hypothesis = await get_hypothesis(db, workspace_name, hypothesis_id)

    # Track if any changes were made
    needs_update = False

    # Update fields if provided
    if hypothesis_update.content is not None:
        hypothesis.content = hypothesis_update.content
        needs_update = True

    if hypothesis_update.status is not None:
        hypothesis.status = hypothesis_update.status
        needs_update = True

    if hypothesis_update.confidence is not None:
        hypothesis.confidence = hypothesis_update.confidence
        needs_update = True

    if hypothesis_update.source_premise_ids is not None:
        hypothesis.source_premise_ids = hypothesis_update.source_premise_ids
        needs_update = True

    if hypothesis_update.unaccounted_premises_count is not None:
        hypothesis.unaccounted_premises_count = hypothesis_update.unaccounted_premises_count
        needs_update = True

    if hypothesis_update.search_coverage is not None:
        hypothesis.search_coverage = hypothesis_update.search_coverage
        needs_update = True

    if hypothesis_update.tier is not None:
        hypothesis.tier = hypothesis_update.tier
        needs_update = True

    if hypothesis_update.reasoning_metadata is not None:
        hypothesis.reasoning_metadata = hypothesis_update.reasoning_metadata
        needs_update = True

    if hypothesis_update.collection_id is not None:
        hypothesis.collection_id = hypothesis_update.collection_id
        needs_update = True

    # Early exit if unchanged
    if not needs_update:
        logger.debug(
            "Hypothesis %s unchanged in workspace %s, skipping update",
            hypothesis_id,
            workspace_name,
        )
        return hypothesis

    await db.commit()
    await db.refresh(hypothesis)

    logger.debug(
        "Hypothesis %s updated successfully in workspace %s",
        hypothesis_id,
        workspace_name,
    )
    return hypothesis


async def delete_hypothesis(
    db: AsyncSession,
    workspace_name: str,
    hypothesis_id: str,
) -> bool:
    """
    Delete a hypothesis.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        hypothesis_id: ID of the hypothesis to delete

    Returns:
        True if deleted successfully

    Raises:
        ResourceNotFoundException: If the hypothesis does not exist
    """
    # Get existing hypothesis (will raise if not found)
    hypothesis = await get_hypothesis(db, workspace_name, hypothesis_id)

    await db.delete(hypothesis)
    await db.commit()

    logger.debug(
        "Hypothesis %s deleted successfully from workspace %s",
        hypothesis_id,
        workspace_name,
    )
    return True


async def list_hypotheses(
    workspace_name: str,
    observer: str | None = None,
    observed: str | None = None,
    collection_id: str | None = None,
    status: str | None = None,
    filters: dict[str, Any] | None = None,
) -> Select[tuple[models.Hypothesis]]:
    """
    List hypotheses with optional filtering.

    Args:
        workspace_name: Name of the workspace
        observer: Filter by observer peer name
        observed: Filter by observed peer name
        collection_id: Filter by collection ID
        status: Filter by status (active | superseded | falsified)
        filters: Additional metadata filters

    Returns:
        SQLAlchemy Select statement for the hypotheses
    """
    stmt = select(models.Hypothesis).where(
        models.Hypothesis.workspace_name == workspace_name
    )

    # Apply optional filters
    if observer is not None:
        stmt = stmt.where(models.Hypothesis.observer == observer)

    if observed is not None:
        stmt = stmt.where(models.Hypothesis.observed == observed)

    if collection_id is not None:
        stmt = stmt.where(models.Hypothesis.collection_id == collection_id)

    if status is not None:
        stmt = stmt.where(models.Hypothesis.status == status)

    # Apply metadata filters
    stmt = apply_filter(stmt, models.Hypothesis, filters)

    # Order by created_at descending (most recent first)
    stmt = stmt.order_by(models.Hypothesis.created_at.desc())

    return stmt
