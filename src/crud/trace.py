"""CRUD operations for FalsificationTrace model.

Note: FalsificationTrace is immutable (append-only). No update or delete operations.
"""

from datetime import datetime
from logging import getLogger
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.workspace import get_or_create_workspace
from src.exceptions import ResourceNotFoundException
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def create_trace(
    db: AsyncSession,
    trace: schemas.FalsificationTraceCreate,
    workspace_name: str,
) -> models.FalsificationTrace:
    """
    Create a new falsification trace.

    Note: Traces are immutable once created.

    Args:
        db: Database session
        trace: Trace creation schema
        workspace_name: Name of the workspace

    Returns:
        Created trace object

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    # Ensure workspace exists
    await get_or_create_workspace(db, schemas.WorkspaceCreate(name=workspace_name))

    # Create trace object
    trace_obj = models.FalsificationTrace(
        prediction_id=trace.prediction_id,
        search_queries=trace.search_queries,
        contradicting_premise_ids=trace.contradicting_premise_ids,
        reasoning_chain=trace.reasoning_chain or {},
        final_status=trace.final_status or "untested",
        search_count=trace.search_count or 0,
        search_efficiency_score=trace.search_efficiency_score,
        workspace_name=workspace_name,
        collection_id=trace.collection_id,
    )

    db.add(trace_obj)
    await db.commit()
    await db.refresh(trace_obj)

    logger.debug(
        "FalsificationTrace %s created successfully in workspace %s",
        trace_obj.id,
        workspace_name,
    )
    return trace_obj


async def get_trace(
    db: AsyncSession,
    workspace_name: str,
    trace_id: str,
) -> models.FalsificationTrace:
    """
    Get a falsification trace by ID.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        trace_id: ID of the trace

    Returns:
        The trace if found

    Raises:
        ResourceNotFoundException: If the trace does not exist
    """
    stmt = (
        select(models.FalsificationTrace)
        .where(models.FalsificationTrace.workspace_name == workspace_name)
        .where(models.FalsificationTrace.id == trace_id)
    )
    result = await db.execute(stmt)
    trace = result.scalar_one_or_none()

    if trace is None:
        raise ResourceNotFoundException(
            f"FalsificationTrace {trace_id} not found in workspace {workspace_name}"
        )

    return trace


async def list_traces(
    workspace_name: str,
    prediction_id: str | None = None,
    collection_id: str | None = None,
    final_status: str | None = None,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    filters: dict[str, Any] | None = None,
) -> Select[tuple[models.FalsificationTrace]]:
    """
    List falsification traces with optional filtering.

    Args:
        workspace_name: Name of the workspace
        prediction_id: Filter by prediction ID
        collection_id: Filter by collection ID
        final_status: Filter by final status (unfalsified | falsified | untested)
        after_date: Return traces created after this datetime
        before_date: Return traces created before this datetime
        filters: Additional metadata filters

    Returns:
        SQLAlchemy Select statement for the traces
    """
    stmt = select(models.FalsificationTrace).where(
        models.FalsificationTrace.workspace_name == workspace_name
    )

    # Apply optional filters
    if prediction_id is not None:
        stmt = stmt.where(models.FalsificationTrace.prediction_id == prediction_id)

    if collection_id is not None:
        stmt = stmt.where(models.FalsificationTrace.collection_id == collection_id)

    if final_status is not None:
        stmt = stmt.where(models.FalsificationTrace.final_status == final_status)

    # Apply date range filters
    if after_date is not None:
        stmt = stmt.where(models.FalsificationTrace.created_at >= after_date)

    if before_date is not None:
        stmt = stmt.where(models.FalsificationTrace.created_at <= before_date)

    # Apply metadata filters
    stmt = apply_filter(stmt, models.FalsificationTrace, filters)

    # Order by created_at descending (most recent first)
    stmt = stmt.order_by(models.FalsificationTrace.created_at.desc())

    return stmt


async def get_traces_by_prediction(
    db: AsyncSession,
    workspace_name: str,
    prediction_id: str,
) -> list[models.FalsificationTrace]:
    """
    Get all traces for a specific prediction.

    This is a convenience function for retrieving the complete falsification
    history of a prediction.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        prediction_id: ID of the prediction

    Returns:
        List of traces ordered by creation time (oldest first)
    """
    stmt = (
        select(models.FalsificationTrace)
        .where(models.FalsificationTrace.workspace_name == workspace_name)
        .where(models.FalsificationTrace.prediction_id == prediction_id)
        .order_by(models.FalsificationTrace.created_at.asc())
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())
