from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import Select, case, func, or_, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas

logger = getLogger(__name__)


async def get_queue_status(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None = None,
    *,
    observer: str | None = None,
    observed: str | None = None,
) -> schemas.QueueStatus:
    """
    Get the processing queue status, optionally filtered by observer, sender, and/or session.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Optional session name to filter by
        observer: Optional name of the observer to filter by
        observed: Optional name of the observed (message sender) to filter by
    """
    # Normalize empty strings to None for consistent handling
    normalized_observer = observer if observer else None
    normalized_observed = observed if observed else None
    normalized_session_name = session_name if session_name else None

    stmt = _build_queue_status_query(
        workspace_name,
        normalized_session_name,
        observer=normalized_observer,
        observed=normalized_observed,
    )
    result = await db.execute(stmt)
    rows = result.fetchall()

    counts = _process_queue_rows(rows)
    return _build_status_response(
        normalized_session_name,
        counts,
    )


async def get_deriver_status(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None = None,
    *,
    observer: str | None = None,
    observed: str | None = None,
) -> schemas.QueueStatus:
    """Deprecated: use get_queue_status."""

    return await get_queue_status(
        db=db,
        workspace_name=workspace_name,
        session_name=session_name,
        observer=observer,
        observed=observed,
    )


def _build_queue_status_query(
    workspace_name: str,
    session_name: str | None,
    *,
    observer: str | None = None,
    observed: str | None = None,
) -> Select[Any]:
    """Build SQL query for queue status with validation and aggregation."""
    observer_name_expr = models.QueueItem.payload["observer"].astext
    observed_name_expr = models.QueueItem.payload["observed"].astext

    # Define conditions for cleaner window functions
    is_completed = models.QueueItem.processed
    is_in_progress = (~models.QueueItem.processed) & (
        models.ActiveQueueSession.id.isnot(None)
    )
    is_pending = (~models.QueueItem.processed) & (
        models.ActiveQueueSession.id.is_(None)
    )

    # Use window functions to calculate totals and per-session counts in SQL
    stmt = select(
        models.QueueItem.session_id,
        # Overall totals using window functions
        func.count().over().label("total"),
        func.count(case((is_completed, 1))).over().label("completed"),
        func.count(case((is_in_progress, 1))).over().label("in_progress"),
        func.count(case((is_pending, 1))).over().label("pending"),
        # Per-session totals using partitioned window functions
        func.count()
        .over(partition_by=models.QueueItem.session_id)
        .label("session_total"),
        func.count(case((is_completed, 1)))
        .over(partition_by=models.QueueItem.session_id)
        .label("session_completed"),
        func.count(case((is_in_progress, 1)))
        .over(partition_by=models.QueueItem.session_id)
        .label("session_in_progress"),
        func.count(case((is_pending, 1)))
        .over(partition_by=models.QueueItem.session_id)
        .label("session_pending"),
    ).select_from(models.QueueItem)

    stmt = stmt.outerjoin(
        models.ActiveQueueSession,
        models.QueueItem.work_unit_key == models.ActiveQueueSession.work_unit_key,
    )

    stmt = stmt.where(models.QueueItem.workspace_name == workspace_name)

    if session_name is not None:
        stmt = stmt.join(
            models.Session, models.QueueItem.session_id == models.Session.id
        )
        stmt = stmt.where(models.Session.name == session_name)

    peer_conditions = []
    if observer is not None:
        peer_conditions.append(observer_name_expr == observer)  # pyright: ignore
    if observed is not None:
        peer_conditions.append(observed_name_expr == observed)  # pyright: ignore
    if peer_conditions:
        stmt = stmt.where(or_(*peer_conditions))  # pyright: ignore

    return stmt


def _process_queue_rows(rows: Sequence[Row[Any]]) -> schemas.QueueCounts:
    """Process query results that already contain aggregated counts."""
    if not rows:
        return schemas.QueueCounts(
            total=0,
            completed=0,
            in_progress=0,
            pending=0,
            sessions={},
        )

    # Since we're using window functions, all rows have the same overall totals
    # We just need the first row for overall counts
    first_row = rows[0]

    # Build sessions dictionary from unique session_ids
    sessions: dict[str, schemas.SessionCounts] = {}
    seen_sessions: set[str] = set()

    for row in rows:
        if row.session_id and row.session_id not in seen_sessions:
            sessions[row.session_id] = schemas.SessionCounts(
                completed=row.session_completed,
                in_progress=row.session_in_progress,
                pending=row.session_pending,
            )
            seen_sessions.add(row.session_id)

    return schemas.QueueCounts(
        total=first_row.total,
        completed=first_row.completed,
        in_progress=first_row.in_progress,
        pending=first_row.pending,
        sessions=sessions,
    )


def _build_status_response(
    session_name: str | None,
    counts: schemas.QueueCounts,
) -> schemas.QueueStatus:
    """Build the final response object."""

    if session_name:
        return schemas.QueueStatus(
            total_work_units=counts.total,
            completed_work_units=counts.completed,
            in_progress_work_units=counts.in_progress,
            pending_work_units=counts.pending,
        )

    sessions: dict[str, schemas.SessionQueueStatus] = {}
    for session_id, data in counts.sessions.items():
        total = data.completed + data.in_progress + data.pending
        sessions[session_id] = schemas.SessionQueueStatus(
            session_id=session_id,
            total_work_units=total,
            completed_work_units=data.completed,
            in_progress_work_units=data.in_progress,
            pending_work_units=data.pending,
        )

    return schemas.QueueStatus(
        sessions=sessions if sessions else None,
        total_work_units=counts.total,
        completed_work_units=counts.completed,
        in_progress_work_units=counts.in_progress,
        pending_work_units=counts.pending,
    )
