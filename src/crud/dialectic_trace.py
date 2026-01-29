"""CRUD operations for DialecticTrace records."""

import datetime
import re
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.schemas import DialecticTraceCreate

# Patterns that indicate the agent abstained from answering
ABSTENTION_PATTERNS = [
    r"don't have (?:enough )?information",
    r"cannot (?:find|answer|determine)",
    r"no relevant",
    r"not (?:enough|sufficient) (?:information|context|data)",
    r"unable to (?:find|answer|determine)",
    r"no (?:observations|memory|data) (?:found|available)",
]

_ABSTENTION_REGEX = re.compile("|".join(ABSTENTION_PATTERNS), re.IGNORECASE)


def _is_abstention(response: str) -> bool:
    """Check if a response indicates abstention from answering."""
    return bool(_ABSTENTION_REGEX.search(response))


async def create_dialectic_trace(
    db: AsyncSession,
    trace: DialecticTraceCreate,
) -> models.DialecticTrace:
    """
    Create a new DialecticTrace record.

    Args:
        db: Database session
        trace: DialecticTraceCreate schema with trace data

    Returns:
        The created DialecticTrace model instance
    """
    db_trace = models.DialecticTrace(
        workspace_name=trace.workspace_name,
        session_name=trace.session_name,
        observer=trace.observer,
        observed=trace.observed,
        query=trace.query,
        retrieved_doc_ids=trace.retrieved_doc_ids,
        tool_calls=trace.tool_calls,
        response=trace.response,
        reasoning_level=trace.reasoning_level,
        total_duration_ms=trace.total_duration_ms,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
    )
    db.add(db_trace)
    await db.flush()
    return db_trace


async def get_dialectic_traces(
    db: AsyncSession,
    workspace_name: str,
    limit: int = 100,
    offset: int = 0,
) -> list[models.DialecticTrace]:
    """
    Get dialectic traces for a workspace.

    Args:
        db: Database session
        workspace_name: Workspace to query traces for
        limit: Maximum number of traces to return (default 100)
        offset: Number of traces to skip (default 0)

    Returns:
        List of DialecticTrace records, ordered by created_at descending
    """
    stmt = (
        select(models.DialecticTrace)
        .where(models.DialecticTrace.workspace_name == workspace_name)
        .order_by(models.DialecticTrace.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_dialectic_trace_stats(
    db: AsyncSession,
    workspace_name: str,
    since: datetime.datetime | None = None,
) -> dict[str, Any]:
    """
    Get aggregate statistics for dialectic traces.

    Args:
        db: Database session
        workspace_name: Workspace to query stats for
        since: Optional datetime to filter traces created after this time

    Returns:
        Dictionary with:
        - total_queries: Total number of dialectic queries
        - avg_duration_ms: Average duration in milliseconds
        - abstention_count: Number of queries where agent abstained
        - abstention_rate: Ratio of abstentions to total queries
    """
    # Build base query
    base_filter = models.DialecticTrace.workspace_name == workspace_name
    if since is not None:
        base_filter = base_filter & (models.DialecticTrace.created_at >= since)

    # Get aggregate stats
    stmt = select(
        func.count(models.DialecticTrace.id).label("total_queries"),
        func.avg(models.DialecticTrace.total_duration_ms).label("avg_duration_ms"),
    ).where(base_filter)

    result = await db.execute(stmt)
    row = result.one()
    total_queries = row.total_queries or 0
    avg_duration_ms = float(row.avg_duration_ms) if row.avg_duration_ms else 0.0

    # Get all responses to check for abstentions
    # (We need to do this in Python since abstention detection uses regex)
    responses_stmt = select(models.DialecticTrace.response).where(base_filter)
    responses_result = await db.execute(responses_stmt)
    responses = [r[0] for r in responses_result.all()]

    abstention_count = sum(1 for r in responses if _is_abstention(r))
    abstention_rate = abstention_count / total_queries if total_queries > 0 else 0.0

    return {
        "total_queries": total_queries,
        "avg_duration_ms": avg_duration_ms,
        "abstention_count": abstention_count,
        "abstention_rate": abstention_rate,
    }
