"""CRUD operations for Induction model."""

from logging import getLogger
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.workspace import get_or_create_workspace
from src.exceptions import ResourceNotFoundException
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def create_induction(
    db: AsyncSession,
    induction: schemas.InductionCreate,
    workspace_name: str,
) -> models.Induction:
    """
    Create a new induction.

    Args:
        db: Database session
        induction: Induction creation schema
        workspace_name: Name of the workspace

    Returns:
        Created induction object

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    # Ensure workspace exists
    await get_or_create_workspace(db, schemas.WorkspaceCreate(name=workspace_name))

    # Create induction object
    induction_obj = models.Induction(
        content=induction.content,
        observer=induction.observer,
        observed=induction.observed,
        pattern_type=induction.pattern_type,
        source_prediction_ids=induction.source_prediction_ids,
        source_premise_ids=induction.source_premise_ids,
        confidence=induction.confidence or "medium",
        stability_score=induction.stability_score,
        workspace_name=workspace_name,
        collection_id=induction.collection_id,
    )

    db.add(induction_obj)
    await db.commit()
    await db.refresh(induction_obj)

    logger.debug(
        "Induction %s created successfully in workspace %s",
        induction_obj.id,
        workspace_name,
    )
    return induction_obj


async def get_induction(
    db: AsyncSession,
    workspace_name: str,
    induction_id: str,
) -> models.Induction:
    """
    Get an induction by ID.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        induction_id: ID of the induction

    Returns:
        The induction if found

    Raises:
        ResourceNotFoundException: If the induction does not exist
    """
    stmt = (
        select(models.Induction)
        .where(models.Induction.workspace_name == workspace_name)
        .where(models.Induction.id == induction_id)
    )
    result = await db.execute(stmt)
    induction = result.scalar_one_or_none()

    if induction is None:
        raise ResourceNotFoundException(
            f"Induction {induction_id} not found in workspace {workspace_name}"
        )

    return induction


async def delete_induction(
    db: AsyncSession,
    workspace_name: str,
    induction_id: str,
) -> bool:
    """
    Delete an induction.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        induction_id: ID of the induction to delete

    Returns:
        True if deleted successfully

    Raises:
        ResourceNotFoundException: If the induction does not exist
    """
    # Get existing induction (will raise if not found)
    induction = await get_induction(db, workspace_name, induction_id)

    await db.delete(induction)
    await db.commit()

    logger.debug(
        "Induction %s deleted successfully from workspace %s",
        induction_id,
        workspace_name,
    )
    return True


async def list_inductions(
    workspace_name: str,
    observer: str | None = None,
    observed: str | None = None,
    collection_id: str | None = None,
    pattern_type: str | None = None,
    confidence: str | None = None,
    filters: dict[str, Any] | None = None,
) -> Select[tuple[models.Induction]]:
    """
    List inductions with optional filtering.

    Args:
        workspace_name: Name of the workspace
        observer: Filter by observer peer name
        observed: Filter by observed peer name
        collection_id: Filter by collection ID
        pattern_type: Filter by pattern type (temporal | causal | correlational | structural)
        confidence: Filter by confidence (high | medium | low)
        filters: Additional metadata filters

    Returns:
        SQLAlchemy Select statement for the inductions
    """
    stmt = select(models.Induction).where(
        models.Induction.workspace_name == workspace_name
    )

    # Apply optional filters
    if observer is not None:
        stmt = stmt.where(models.Induction.observer == observer)

    if observed is not None:
        stmt = stmt.where(models.Induction.observed == observed)

    if collection_id is not None:
        stmt = stmt.where(models.Induction.collection_id == collection_id)

    if pattern_type is not None:
        stmt = stmt.where(models.Induction.pattern_type == pattern_type)

    if confidence is not None:
        stmt = stmt.where(models.Induction.confidence == confidence)

    # Apply metadata filters
    stmt = apply_filter(stmt, models.Induction, filters)

    # Order by created_at descending (most recent first)
    stmt = stmt.order_by(models.Induction.created_at.desc())

    return stmt
