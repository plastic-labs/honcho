from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any

from sqlalchemy import ColumnElement, Select, case, func, or_, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings

logger = getLogger(__name__)

REPRESENTATION_WORK_UNIT_PREFIX = "representation:"


def representation_batch_threshold_clause(
    *,
    work_unit_key: ColumnElement[str],
    total_tokens: ColumnElement[Any],
    oldest_created_at: ColumnElement[Any],
) -> ColumnElement[bool] | None:
    """The batch gate a representation work unit passes before it is claimable, or None when no gate applies."""
    if settings.DERIVER.FLUSH_ENABLED:
        return None

    target_tokens = settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS
    if target_tokens <= 0:
        return None

    threshold: ColumnElement[bool] = func.coalesce(total_tokens, 0) >= target_tokens

    max_age_seconds = settings.DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS
    if max_age_seconds > 0:
        threshold = or_(
            threshold,
            oldest_created_at <= func.now() - timedelta(seconds=max_age_seconds),
        )

    return or_(
        ~work_unit_key.startswith(REPRESENTATION_WORK_UNIT_PREFIX),
        threshold,
    )


def unclaimed_work_unit_clause(
    work_unit_key: ColumnElement[str],
) -> ColumnElement[bool]:
    """No claim row exists for this work unit, stale ones included."""
    return (
        ~select(models.ActiveQueueSession.id)
        .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
        .exists()
    )


def stale_claim_cutoff() -> datetime:
    return datetime.now(UTC) - timedelta(
        minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES
    )


def not_live_claimed_work_unit_clause(
    work_unit_key: ColumnElement[str],
) -> ColumnElement[bool]:
    """No claim refreshed inside the stale timeout exists, so a stale claim leaves its work unit claimable."""
    return (
        ~select(models.ActiveQueueSession.id)
        .where(
            models.ActiveQueueSession.work_unit_key == work_unit_key,
            models.ActiveQueueSession.last_updated >= stale_claim_cutoff(),
        )
        .exists()
    )


async def get_deriver_metrics(db: AsyncSession) -> schemas.DeriverMetrics:
    """Count the outstanding deriver work in the whole database, read-only."""
    from src.reconciler.sync_vectors import backoff_eligible  # noqa: PLC0415

    token_stats = (
        select(
            models.QueueItem.work_unit_key,
            func.sum(models.Message.token_count).label("total_tokens"),
            func.min(models.QueueItem.created_at).label("oldest_created_at"),
        )
        .join(models.Message, models.QueueItem.message_id == models.Message.id)
        .where(~models.QueueItem.processed)
        .where(
            models.QueueItem.work_unit_key.startswith(REPRESENTATION_WORK_UNIT_PREFIX)
        )
        .group_by(models.QueueItem.work_unit_key)
        .subquery()
    )

    work_units = (
        select(models.QueueItem.work_unit_key)
        .where(~models.QueueItem.processed)
        .group_by(models.QueueItem.work_unit_key)
        .subquery()
    )

    eligible = (
        select(func.count())
        .select_from(work_units)
        .outerjoin(
            token_stats,
            work_units.c.work_unit_key == token_stats.c.work_unit_key,
        )
        .where(not_live_claimed_work_unit_clause(work_units.c.work_unit_key))
    )

    threshold_clause = representation_batch_threshold_clause(
        work_unit_key=work_units.c.work_unit_key,
        total_tokens=token_stats.c.total_tokens,
        oldest_created_at=token_stats.c.oldest_created_at,
    )
    if threshold_clause is not None:
        eligible = eligible.where(threshold_clause)

    claimed = (
        select(func.count())
        .select_from(models.ActiveQueueSession)
        .where(models.ActiveQueueSession.last_updated >= stale_claim_cutoff())
    )

    pending = select(
        func.count(models.QueueItem.id),
        func.coalesce(
            func.extract("epoch", func.now() - func.min(models.QueueItem.created_at)),
            0,
        ),
    ).where(~models.QueueItem.processed)

    embeddings = select(
        func.count(),
        func.coalesce(
            func.sum(
                case(
                    (backoff_eligible(models.MessageEmbedding.last_sync_at), 1),
                    else_=0,
                )
            ),
            0,
        ),
    ).where(models.MessageEmbedding.sync_state == "pending")

    eligible_count = (await db.execute(eligible)).scalar_one()
    claimed_count = (await db.execute(claimed)).scalar_one()
    pending_count, oldest_age = (await db.execute(pending)).one()
    embeddings_pending, embeddings_due = (await db.execute(embeddings)).one()

    return schemas.DeriverMetrics(
        eligible_work_units=int(eligible_count),
        claimed_work_units=int(claimed_count),
        pending_items=int(pending_count),
        oldest_pending_age_seconds=float(oldest_age),
        embeddings_pending=int(embeddings_pending),
        embeddings_pending_due=int(embeddings_due),
    )


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


# Task types surfaced by the queue status endpoint.
_TRACKED_TASK_TYPES = ("representation", "summary", "dream")


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

    # Only include user-facing task types
    stmt = stmt.where(models.QueueItem.task_type.in_(_TRACKED_TASK_TYPES))

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
