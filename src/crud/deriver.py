from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import Select, and_, case, false, func, or_, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings

logger = getLogger(__name__)


# Task types surfaced by the queue status endpoint.
_TRACKED_TASK_TYPES = ("representation", "summary", "dream")

# Only this task type is gated by DERIVER_REPRESENTATION_BATCH_MAX_TOKENS.
_THRESHOLD_GATED_TASK_TYPE = "representation"


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

    Only tracks user-facing task types: representation, summary, and dream.
    Internal infrastructure tasks (reconciler, webhook, deletion) are excluded.

    Pending work units are further split into "stalled" (representation work
    units below DERIVER_REPRESENTATION_BATCH_MAX_TOKENS) vs "ready" (everything
    else). When DERIVER_FLUSH_ENABLED is true, the threshold is bypassed and
    nothing is stalled.

    Note: completed_work_units reflects items since the last periodic queue
    cleanup, not lifetime totals.

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


async def get_queue_work_units(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None = None,
    *,
    observer: str | None = None,
    observed: str | None = None,
) -> schemas.QueueWorkUnitsResponse:
    """
    Return one row per unprocessed work unit in the queue, with token totals,
    in-progress flag, and threshold classification.

    Useful for debugging "why isn't this work unit advancing?" — distinguishes
    work units stalled below DERIVER_REPRESENTATION_BATCH_MAX_TOKENS from those
    that are claimed by a worker or eligible to be claimed.

    Same filter semantics as get_queue_status: observer/observed match the
    queue item payload, session_name filters via the sessions table.
    """
    normalized_observer = observer if observer else None
    normalized_observed = observed if observed else None
    normalized_session_name = session_name if session_name else None

    batch_max_tokens = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
    flush_enabled = settings.DERIVER.FLUSH_ENABLED

    stmt = _build_queue_work_units_query(
        workspace_name,
        normalized_session_name,
        observer=normalized_observer,
        observed=normalized_observed,
    )
    result = await db.execute(stmt)
    rows = result.fetchall()

    work_units: list[schemas.QueueWorkUnit] = []
    for row in rows:
        threshold_gated = (
            row.task_type == _THRESHOLD_GATED_TASK_TYPE
            and not flush_enabled
            and batch_max_tokens > 0
        )
        if threshold_gated:
            hit_threshold = row.pending_tokens >= batch_max_tokens
            tokens_until_threshold = max(batch_max_tokens - row.pending_tokens, 0)
        else:
            hit_threshold = True
            tokens_until_threshold = 0

        work_units.append(
            schemas.QueueWorkUnit(
                work_unit_key=row.work_unit_key,
                task_type=row.task_type,
                session_id=row.session_id,
                session_name=row.session_name,
                observer=row.observer,
                observed=row.observed,
                pending_items=row.pending_items,
                pending_tokens=row.pending_tokens,
                tokens_until_threshold=tokens_until_threshold,
                hit_threshold=hit_threshold,
                in_progress=bool(row.in_progress),
                oldest_item_at=row.oldest_item_at,
                newest_item_at=row.newest_item_at,
            )
        )

    return schemas.QueueWorkUnitsResponse(
        representation_batch_max_tokens=batch_max_tokens,
        flush_enabled=flush_enabled,
        work_units=work_units,
    )


def _build_queue_status_query(
    workspace_name: str,
    session_name: str | None,
    *,
    observer: str | None = None,
    observed: str | None = None,
) -> Select[Any]:
    """Build SQL query for queue status with validation and aggregation.

    Two-layer structure: an inner per-queue-item subquery joins messages to
    compute the per-work_unit_key pending-token sum (a window function), then
    the outer query classifies each row and tallies overall + per-session
    counts via additional window functions.
    """
    observer_name_expr = models.QueueItem.payload["observer"].astext
    observed_name_expr = models.QueueItem.payload["observed"].astext

    inner_is_in_progress = models.ActiveQueueSession.id.isnot(None)
    inner_is_pending = (~models.QueueItem.processed) & (
        models.ActiveQueueSession.id.is_(None)
    )

    # Per-work_unit_key sum of token_count, restricted to pending items.
    # Computed in the inner subquery because window functions cannot reference
    # each other directly in the outer SELECT.
    pending_tokens_per_wuk = func.sum(
        case(
            (inner_is_pending, func.coalesce(models.Message.token_count, 0)),
            else_=0,
        )
    ).over(partition_by=models.QueueItem.work_unit_key)

    inner = (
        select(
            models.QueueItem.session_id.label("session_id"),
            models.QueueItem.task_type.label("task_type"),
            models.QueueItem.processed.label("processed"),
            inner_is_in_progress.label("is_in_progress_flag"),
            pending_tokens_per_wuk.label("wuk_pending_tokens"),
        )
        .select_from(models.QueueItem)
        .outerjoin(
            models.ActiveQueueSession,
            models.QueueItem.work_unit_key == models.ActiveQueueSession.work_unit_key,
        )
        .outerjoin(
            models.Message,
            models.QueueItem.message_id == models.Message.id,
        )
    )

    inner = inner.where(models.QueueItem.workspace_name == workspace_name)
    inner = inner.where(models.QueueItem.task_type.in_(_TRACKED_TASK_TYPES))

    if session_name is not None:
        inner = inner.join(
            models.Session, models.QueueItem.session_id == models.Session.id
        )
        inner = inner.where(models.Session.name == session_name)

    peer_conditions = []
    if observer is not None:
        peer_conditions.append(observer_name_expr == observer)  # pyright: ignore
    if observed is not None:
        peer_conditions.append(observed_name_expr == observed)  # pyright: ignore
    if peer_conditions:
        inner = inner.where(or_(*peer_conditions))  # pyright: ignore

    inner_subq = inner.subquery()

    # Outer classification
    is_completed = inner_subq.c.processed
    is_in_progress = (~inner_subq.c.processed) & inner_subq.c.is_in_progress_flag
    is_pending = (~inner_subq.c.processed) & ~inner_subq.c.is_in_progress_flag

    batch_max_tokens = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
    flush_enabled = settings.DERIVER.FLUSH_ENABLED

    if not flush_enabled and batch_max_tokens > 0:
        is_stalled = and_(
            is_pending,
            inner_subq.c.task_type == _THRESHOLD_GATED_TASK_TYPE,
            inner_subq.c.wuk_pending_tokens < batch_max_tokens,
        )
    else:
        is_stalled = false()

    is_pending_ready = and_(is_pending, ~is_stalled)

    stmt = select(
        inner_subq.c.session_id,
        # Overall totals
        func.count().over().label("total"),
        func.count(case((is_completed, 1))).over().label("completed"),
        func.count(case((is_in_progress, 1))).over().label("in_progress"),
        func.count(case((is_pending, 1))).over().label("pending"),
        func.count(case((is_stalled, 1))).over().label("pending_stalled"),
        func.count(case((is_pending_ready, 1))).over().label("pending_ready"),
        # Per-session totals
        func.count().over(partition_by=inner_subq.c.session_id).label("session_total"),
        func.count(case((is_completed, 1)))
        .over(partition_by=inner_subq.c.session_id)
        .label("session_completed"),
        func.count(case((is_in_progress, 1)))
        .over(partition_by=inner_subq.c.session_id)
        .label("session_in_progress"),
        func.count(case((is_pending, 1)))
        .over(partition_by=inner_subq.c.session_id)
        .label("session_pending"),
        func.count(case((is_stalled, 1)))
        .over(partition_by=inner_subq.c.session_id)
        .label("session_pending_stalled"),
        func.count(case((is_pending_ready, 1)))
        .over(partition_by=inner_subq.c.session_id)
        .label("session_pending_ready"),
    )

    return stmt


def _build_queue_work_units_query(
    workspace_name: str,
    session_name: str | None,
    *,
    observer: str | None = None,
    observed: str | None = None,
) -> Select[Any]:
    """One row per unprocessed work_unit_key, aggregating queue items + tokens."""
    observer_name_expr = models.QueueItem.payload["observer"].astext
    observed_name_expr = models.QueueItem.payload["observed"].astext

    stmt = (
        select(
            models.QueueItem.work_unit_key.label("work_unit_key"),
            models.QueueItem.task_type.label("task_type"),
            models.QueueItem.session_id.label("session_id"),
            models.Session.name.label("session_name"),
            func.min(observer_name_expr).label("observer"),
            func.min(observed_name_expr).label("observed"),
            func.count().label("pending_items"),
            func.coalesce(
                func.sum(func.coalesce(models.Message.token_count, 0)), 0
            ).label("pending_tokens"),
            func.min(models.QueueItem.created_at).label("oldest_item_at"),
            func.max(models.QueueItem.created_at).label("newest_item_at"),
            func.bool_or(models.ActiveQueueSession.id.isnot(None)).label("in_progress"),
        )
        .select_from(models.QueueItem)
        .outerjoin(
            models.ActiveQueueSession,
            models.QueueItem.work_unit_key == models.ActiveQueueSession.work_unit_key,
        )
        .outerjoin(
            models.Message,
            models.QueueItem.message_id == models.Message.id,
        )
        .outerjoin(
            models.Session,
            models.QueueItem.session_id == models.Session.id,
        )
        .where(models.QueueItem.workspace_name == workspace_name)
        .where(models.QueueItem.task_type.in_(_TRACKED_TASK_TYPES))
        .where(~models.QueueItem.processed)
        .group_by(
            models.QueueItem.work_unit_key,
            models.QueueItem.task_type,
            models.QueueItem.session_id,
            models.Session.name,
        )
        .order_by(
            # Stalled-first ordering would require a HAVING-style classification;
            # instead order by oldest pending so debugging surfaces oldest stuck
            # work first.
            func.min(models.QueueItem.created_at)
        )
    )

    if session_name is not None:
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
            pending_stalled=0,
            pending_ready=0,
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
                pending_stalled=row.session_pending_stalled,
                pending_ready=row.session_pending_ready,
            )
            seen_sessions.add(row.session_id)

    return schemas.QueueCounts(
        total=first_row.total,
        completed=first_row.completed,
        in_progress=first_row.in_progress,
        pending=first_row.pending,
        pending_stalled=first_row.pending_stalled,
        pending_ready=first_row.pending_ready,
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
            pending_stalled_work_units=counts.pending_stalled,
            pending_ready_work_units=counts.pending_ready,
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
            pending_stalled_work_units=data.pending_stalled,
            pending_ready_work_units=data.pending_ready,
        )

    return schemas.QueueStatus(
        sessions=sessions if sessions else None,
        total_work_units=counts.total,
        completed_work_units=counts.completed,
        in_progress_work_units=counts.in_progress,
        pending_work_units=counts.pending,
        pending_stalled_work_units=counts.pending_stalled,
        pending_ready_work_units=counts.pending_ready,
    )
