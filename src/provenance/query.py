"""Provenance query interface for retrieving historical traces.

Provides functions for querying provenance traces with various filters
including agent type, date range, outcome, and more.
"""

from datetime import datetime
from logging import getLogger
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models

logger = getLogger(__name__)


async def query_traces_by_agent(
    db: AsyncSession,
    workspace_name: str,
    agent_type: str,
    limit: int = 100,
    offset: int = 0,
) -> list[models.FalsificationTrace]:
    """
    Query traces by agent type.

    Args:
        db: Database session
        workspace_name: Workspace name
        agent_type: Agent type to filter by (abducer | predictor | falsifier | inductor | extractor)
        limit: Maximum number of traces to return
        offset: Number of traces to skip

    Returns:
        List of FalsificationTrace objects
    """
    stmt = (
        select(models.FalsificationTrace)
        .where(models.FalsificationTrace.workspace_name == workspace_name)
        .where(
            models.FalsificationTrace.reasoning_chain["agent_type"].astext == agent_type
        )
        .order_by(models.FalsificationTrace.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    result = await db.execute(stmt)
    traces = list(result.scalars().all())

    logger.debug(
        "Found %d traces for agent %s in workspace %s",
        len(traces),
        agent_type,
        workspace_name,
    )

    return traces


async def query_traces_by_date_range(
    db: AsyncSession,
    workspace_name: str,
    start_date: datetime,
    end_date: datetime,
    agent_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[models.FalsificationTrace]:
    """
    Query traces within a date range.

    Args:
        db: Database session
        workspace_name: Workspace name
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        agent_type: Optional agent type filter
        limit: Maximum number of traces to return
        offset: Number of traces to skip

    Returns:
        List of FalsificationTrace objects
    """
    stmt = (
        select(models.FalsificationTrace)
        .where(models.FalsificationTrace.workspace_name == workspace_name)
        .where(models.FalsificationTrace.created_at >= start_date)
        .where(models.FalsificationTrace.created_at <= end_date)
    )

    if agent_type:
        stmt = stmt.where(
            models.FalsificationTrace.reasoning_chain["agent_type"].astext == agent_type
        )

    stmt = (
        stmt.order_by(models.FalsificationTrace.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    result = await db.execute(stmt)
    traces = list(result.scalars().all())

    logger.debug(
        "Found %d traces in date range %s to %s for workspace %s",
        len(traces),
        start_date,
        end_date,
        workspace_name,
    )

    return traces


async def query_traces_by_outcome(
    db: AsyncSession,
    workspace_name: str,
    success: bool,
    agent_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[models.FalsificationTrace]:
    """
    Query traces by execution outcome (success/failure).

    Args:
        db: Database session
        workspace_name: Workspace name
        success: Whether to filter for successful (True) or failed (False) executions
        agent_type: Optional agent type filter
        limit: Maximum number of traces to return
        offset: Number of traces to skip

    Returns:
        List of FalsificationTrace objects
    """
    stmt = (
        select(models.FalsificationTrace)
        .where(models.FalsificationTrace.workspace_name == workspace_name)
        .where(
            models.FalsificationTrace.reasoning_chain["success"].astext
            == str(success).lower()
        )
    )

    if agent_type:
        stmt = stmt.where(
            models.FalsificationTrace.reasoning_chain["agent_type"].astext == agent_type
        )

    stmt = (
        stmt.order_by(models.FalsificationTrace.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    result = await db.execute(stmt)
    traces = list(result.scalars().all())

    logger.debug(
        "Found %d %s traces for workspace %s",
        len(traces),
        "successful" if success else "failed",
        workspace_name,
    )

    return traces


async def query_traces_by_prediction(
    db: AsyncSession,
    workspace_name: str,
    prediction_id: str,
    limit: int = 100,
) -> list[models.FalsificationTrace]:
    """
    Query all traces associated with a specific prediction.

    Args:
        db: Database session
        workspace_name: Workspace name
        prediction_id: Prediction ID
        limit: Maximum number of traces to return

    Returns:
        List of FalsificationTrace objects ordered by creation time
    """
    stmt = (
        select(models.FalsificationTrace)
        .where(models.FalsificationTrace.workspace_name == workspace_name)
        .where(models.FalsificationTrace.prediction_id == prediction_id)
        .order_by(models.FalsificationTrace.created_at.asc())
        .limit(limit)
    )

    result = await db.execute(stmt)
    traces = list(result.scalars().all())

    logger.debug(
        "Found %d traces for prediction %s in workspace %s",
        len(traces),
        prediction_id,
        workspace_name,
    )

    return traces


async def query_traces_by_metadata(
    db: AsyncSession,
    workspace_name: str,
    metadata_key: str,
    metadata_value: Any,
    agent_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[models.FalsificationTrace]:
    """
    Query traces by metadata key-value pair.

    Args:
        db: Database session
        workspace_name: Workspace name
        metadata_key: Metadata key to filter by
        metadata_value: Metadata value to filter by
        agent_type: Optional agent type filter
        limit: Maximum number of traces to return
        offset: Number of traces to skip

    Returns:
        List of FalsificationTrace objects
    """
    stmt = (
        select(models.FalsificationTrace)
        .where(models.FalsificationTrace.workspace_name == workspace_name)
        .where(
            models.FalsificationTrace.reasoning_chain["metadata"][metadata_key].astext
            == str(metadata_value)
        )
    )

    if agent_type:
        stmt = stmt.where(
            models.FalsificationTrace.reasoning_chain["agent_type"].astext == agent_type
        )

    stmt = (
        stmt.order_by(models.FalsificationTrace.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    result = await db.execute(stmt)
    traces = list(result.scalars().all())

    logger.debug(
        "Found %d traces with metadata %s=%s for workspace %s",
        len(traces),
        metadata_key,
        metadata_value,
        workspace_name,
    )

    return traces


def build_trace_query(
    workspace_name: str,
    agent_type: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    success: bool | None = None,
    prediction_id: str | None = None,
    collection_id: str | None = None,
) -> Select[tuple[models.FalsificationTrace]]:
    """
    Build a flexible trace query with multiple optional filters.

    This is a low-level function that returns a SQLAlchemy Select statement
    for advanced use cases. Most users should use the specific query functions above.

    Args:
        workspace_name: Workspace name (required)
        agent_type: Optional agent type filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        success: Optional success filter
        prediction_id: Optional prediction ID filter
        collection_id: Optional collection ID filter

    Returns:
        SQLAlchemy Select statement
    """
    stmt = select(models.FalsificationTrace).where(
        models.FalsificationTrace.workspace_name == workspace_name
    )

    if agent_type:
        stmt = stmt.where(
            models.FalsificationTrace.reasoning_chain["agent_type"].astext == agent_type
        )

    if start_date:
        stmt = stmt.where(models.FalsificationTrace.created_at >= start_date)

    if end_date:
        stmt = stmt.where(models.FalsificationTrace.created_at <= end_date)

    if success is not None:
        stmt = stmt.where(
            models.FalsificationTrace.reasoning_chain["success"].astext
            == str(success).lower()
        )

    if prediction_id:
        stmt = stmt.where(models.FalsificationTrace.prediction_id == prediction_id)

    if collection_id:
        stmt = stmt.where(models.FalsificationTrace.collection_id == collection_id)

    stmt = stmt.order_by(models.FalsificationTrace.created_at.desc())

    return stmt
