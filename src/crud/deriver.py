from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any

from sqlalchemy import (
    ColumnElement,
    Select,
    SQLColumnExpression,
    and_,
    case,
    delete,
    func,
    or_,
    select,
)
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.db import tenant_context

logger = getLogger(__name__)


def batch_threshold_clause() -> ColumnElement[bool] | None:
    """The token-or-age gate a batch row passes before its unit is claimable, or None when no gate applies."""
    # region ai
    # Typed by task_type on the batch row, never by key prefix — a
    # startswith("representation:") test here silently disabled the gate for
    # every tenant-prefixed key. Non-representation units are always eligible;
    # representation units wait for the token target or the age flush.
    # endregion
    if settings.DERIVER.FLUSH_ENABLED:
        return None

    target_tokens = settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS
    if target_tokens <= 0:
        return None

    threshold: ColumnElement[bool] = models.QueueItemBatch.total_tokens >= target_tokens

    max_age_seconds = settings.DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS
    if max_age_seconds > 0:
        threshold = or_(
            threshold,
            models.QueueItemBatch.oldest_created_at
            <= func.now() - timedelta(seconds=max_age_seconds),
        )

    return or_(models.QueueItemBatch.task_type != "representation", threshold)


def claim_excluded_tenant_ids() -> Sequence[str] | None:
    """Tenant ids the claim must skip, or None when no exclusion applies."""
    # region ai
    # The suspension seam: excluding a tenant means filtering its batch rows
    # out of the claim's eligible set before ranking, so an excluded whale
    # contributes nothing to any round. No exclusion source exists yet, hence
    # None; the seam returns ids rather than a clause so wiring a source in
    # cannot reintroduce the NULL footgun — the claim composes `tenant_id IS
    # NULL OR tenant_id NOT IN (ids)` itself, keeping the tenant-less
    # (reconciler) lane in rotation by construction (a bare NOT IN is
    # NULL-false and would silently starve it).
    # endregion
    return None


def unclaimed_work_unit_clause(
    work_unit_key: SQLColumnExpression[str],
) -> ColumnElement[bool]:
    """No claim row exists for this work unit, stale ones included."""
    return (
        ~select(models.ActiveQueueSession.id)
        .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
        .exists()
    )


def active_queue_session_match(
    workspace_name: str,
    session_name: str | None = None,
) -> ColumnElement[bool]:
    """Match claim rows whose work unit belongs to this workspace (and session), flag-aware."""
    # region ai
    # Flag-off keys are {task_type}:{workspace}:{session?}:... — workspace at
    # split position 2, session at position 3. Under MULTI_TENANT tenant-scoped
    # keys gain a {tenant_id}: prefix, shifting both positions by one; rows are
    # additionally pinned to the ambient tenant via the tenant_id attribution
    # column (a position match alone would hit other tenants' rows for a
    # same-named workspace), and tenant-less (reconciler) rows are left to the
    # stale-claim GC that owns them. Fail closed flag-on with no ambient tenant —
    # every caller (API routes via require_auth, the deriver via
    # process_work_unit) is tenant-bound when the flag is on.
    # Known, pre-existing wrinkle both branches inherit: dream keys carry
    # workspace one position deeper ({task}:{dream_type}:{workspace}:...) and
    # scope keys carry session at the peer's position + 1, so those units are
    # missed here and swept by the stale-claim GC instead — unchanged behavior.
    # endregion
    if settings.MULTI_TENANT:
        tenant = tenant_context.get()
        if not tenant:
            raise ValueError(
                f"cannot match claim rows for workspace {workspace_name!r} "
                + "without a tenant when MULTI_TENANT is on"
            )
        workspace_position, session_position = 3, 4
        tenant_match: ColumnElement[bool] | None = (
            models.ActiveQueueSession.tenant_id == tenant
        )
    else:
        workspace_position, session_position = 2, 3
        tenant_match = None

    match: ColumnElement[bool] = (
        func.split_part(
            models.ActiveQueueSession.work_unit_key, ":", workspace_position
        )
        == workspace_name
    )
    if session_name is not None:
        match = and_(
            match,
            func.split_part(
                models.ActiveQueueSession.work_unit_key, ":", session_position
            )
            == session_name,
        )
    if tenant_match is not None:
        match = and_(tenant_match, match)
    return match


def queue_item_tenant_match() -> ColumnElement[bool] | None:
    """Pin a QueueItem row to the ambient tenant, flag-aware; None flag-off (compose conditionally, as claim_rows_query does with batch_threshold_clause, so flag-off SQL is unchanged)."""
    # region ai
    # tenant_id is the authoritative tenant-scoped column for queue rows, not
    # workspace_name or message_id: workspace_name is only unique per tenant
    # (every tenant has a "default" workspace) and message ids are drawn from
    # one global sequence, so either can collide across tenants. tenant_id
    # does not — the queue_item_batches backfill (migration b7d2f4a81c39,
    # the `UPDATE {schema}.queue SET tenant_id = split_part(work_unit_key,
    # ':', 1) ...` step) derived it from the work_unit_key's tenant prefix for
    # every row that predated the column, and every tenant-bound writer since
    # (src/deriver/enqueue.py's _stamp_tenant_id, the reconciler scheduler,
    # webhooks/events.py) sets it explicitly on insert. Rows with tenant_id
    # IS NULL flag-on are the tenant-less reconciler lane (workspace_name is
    # NULL there too, per the workspace_null_iff_reconciler check) and are
    # left alone here, exactly as active_queue_session_match above leaves
    # reconciler claim rows to the stale-claim GC that owns them. Fail closed
    # flag-on with no ambient tenant, same shape as active_queue_session_match:
    # every caller (API routes via require_auth, the deriver via
    # process_work_unit) is tenant-bound when the flag is on.
    # endregion
    if not settings.MULTI_TENANT:
        return None
    tenant = tenant_context.get()
    if not tenant:
        raise ValueError(
            "cannot match queue rows without a tenant when MULTI_TENANT is on"
        )
    return models.QueueItem.tenant_id == tenant


def claim_rows_query(limit: int) -> Select[Any]:
    """The claim's locked candidate SELECT: the tenants holding least work in flight first, skipping rows a concurrent claimer holds."""
    # region ai
    # Fairness = round-robin over tenant_id, weighted by the concurrency a
    # tenant already holds. Ranks number each tenant's ELIGIBLE units
    # oldest-first, and the tenant's LIVE claim count is ADDED to that rank, so
    # the ordering key reads "how many units deep is this for its tenant,
    # counting what it is already running". A tenant with two units in flight
    # starts its next one at effective rank 3 and yields to every idle tenant's
    # rank 1 — fairness conserves granted concurrency, not queue position,
    # which is what stops a whale from holding every worker.
    #
    # Ranking over eligible rows ALONE lets a tenant already being processed
    # re-enter at rank 1 every round (it is charged nothing for the unit it
    # holds); moving eligibility OUTSIDE the subquery instead over-charges it,
    # because a claimed rank-1 unit would shadow its tenant's rank-2 and sink
    # it behind every other tenant regardless of how little that tenant holds.
    # The offset is the middle: a tenant is charged for what it holds and
    # nothing else. Stale claims are deliberately not counted — a crashed
    # worker's abandoned claim would otherwise penalize its tenant until the
    # GC reaps it.
    #
    # NULLs group as one partition, so the tenant-less reconciler lane is a
    # bucket in the rotation (its offset joins NULL-safely, hence IS NOT
    # DISTINCT FROM) and a flag-off deployment (every tenant_id NULL) adds one
    # constant offset to every row, leaving plain oldest-first — the ordering
    # is identical to the pre-fairness one. The window function cannot combine
    # with FOR UPDATE, hence the rank-then-join shape.
    #
    # One locking statement on purpose: FOR UPDATE SKIP LOCKED locks rows in
    # output order below the LIMIT, so a concurrent claimer's locked rows are
    # skipped and BACKFILLED from the sorted stream — both claimers fill their
    # batch, disjointly, with zero wasted claims. A two-step select-then-lock
    # variant loses that backfill (the loser picks the same blind candidates,
    # skips them all, and claims nothing for the poll). The cost of locking in
    # scheduling order is that a rare lock-order inversion against the enqueue
    # trigger's batch upserts can deadlock; Postgres's detector breaks it
    # and both sides retry — the enqueue in _insert_queue_records, the claim
    # on its next poll (the polling loop's catch-all backs off and continues).
    # endregion
    # The concurrency each tenant already holds. Sized by the fleet's live
    # workers, not by queue depth, so it stays a small aggregate.
    inflight = (
        select(
            models.ActiveQueueSession.tenant_id.label("tenant_id"),
            func.count().label("inflight_units"),
        )
        .where(models.ActiveQueueSession.last_updated >= stale_claim_cutoff())
        .group_by(models.ActiveQueueSession.tenant_id)
        .subquery()
    )

    eligible = (
        select(
            models.QueueItemBatch.work_unit_key,
            models.QueueItemBatch.oldest_created_at,
            (
                func.coalesce(inflight.c.inflight_units, 0)
                + func.row_number().over(
                    partition_by=models.QueueItemBatch.tenant_id,
                    order_by=(
                        models.QueueItemBatch.oldest_created_at.asc(),
                        models.QueueItemBatch.work_unit_key.asc(),
                    ),
                )
            ).label("tenant_fair_rank"),
        )
        .outerjoin(
            inflight,
            models.QueueItemBatch.tenant_id.is_not_distinct_from(inflight.c.tenant_id),
        )
        .where(unclaimed_work_unit_clause(models.QueueItemBatch.work_unit_key))
    )

    threshold_clause = batch_threshold_clause()
    if threshold_clause is not None:
        eligible = eligible.where(threshold_clause)
    excluded_tenant_ids = claim_excluded_tenant_ids()
    if excluded_tenant_ids:
        eligible = eligible.where(
            or_(
                models.QueueItemBatch.tenant_id.is_(None),
                models.QueueItemBatch.tenant_id.notin_(excluded_tenant_ids),
            )
        )
    eligible_subq = eligible.subquery()

    return (
        select(
            models.QueueItemBatch.work_unit_key,
            models.QueueItemBatch.task_type,
            models.QueueItemBatch.total_tokens,
            models.QueueItemBatch.oldest_created_at,
        )
        .join(
            eligible_subq,
            models.QueueItemBatch.work_unit_key == eligible_subq.c.work_unit_key,
        )
        .order_by(
            eligible_subq.c.tenant_fair_rank.asc(),
            eligible_subq.c.oldest_created_at.asc(),
            models.QueueItemBatch.work_unit_key.asc(),
        )
        .limit(limit)
        .with_for_update(skip_locked=True, of=models.QueueItemBatch)
    )


def stale_claim_cutoff() -> datetime:
    return datetime.now(UTC) - timedelta(
        minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES
    )


def not_live_claimed_work_unit_clause(
    work_unit_key: SQLColumnExpression[str],
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


async def cleanup_stale_work_units(db: AsyncSession) -> None:
    """Delete claim rows whose last update is older than the stale timeout."""
    stale_ids = (
        (
            await db.execute(
                select(models.ActiveQueueSession.id)
                .where(models.ActiveQueueSession.last_updated < stale_claim_cutoff())
                .order_by(models.ActiveQueueSession.last_updated)
                .with_for_update(skip_locked=True)
            )
        )
        .scalars()
        .all()
    )

    if stale_ids:
        await db.execute(
            delete(models.ActiveQueueSession).where(
                models.ActiveQueueSession.id.in_(stale_ids)
            )
        )
    await db.commit()


async def get_deriver_metrics(db: AsyncSession) -> schemas.DeriverMetrics:
    """Count the outstanding deriver work in the whole database, read-only."""
    from src.reconciler.sync_vectors import backoff_eligible  # noqa: PLC0415

    # region ai
    # Reads the trigger-maintained queue_item_batches, not the queue: this poller
    # ran the same two GROUP BYs as the old claim path on every backlog-metrics
    # poll, the identical ~O(depth²) cost. sum(pending_count) equals the old
    # per-item count and min(oldest_created_at) the old per-item min by
    # construction of the triggers (see the queue_item_batches migration).
    # endregion
    eligible = (
        select(func.count())
        .select_from(models.QueueItemBatch)
        .where(not_live_claimed_work_unit_clause(models.QueueItemBatch.work_unit_key))
    )

    threshold_clause = batch_threshold_clause()
    if threshold_clause is not None:
        eligible = eligible.where(threshold_clause)

    # region ai
    # An excluded tenant's rows are filtered out of the claim, so nothing will
    # pick them up and they must not read as work waiting for a worker. Every
    # gauge that answers "is there anything to do" therefore drops them: KEDA
    # scales the deriver fleet off eligible_work_units, and outstanding-work
    # falls back to the pending count and the oldest pending age when eligible
    # and claimed are both zero — so leaving them in pending would hold the
    # fleet up for a backlog no worker can take, one level down from the same
    # bug. They are counted on their own gauge rather than dropped, so
    # suspended depth stays visible.
    #
    # Same NULL-safe shape as the claim's filter: a bare NOT IN is NULL-false
    # and would silently drop the tenant-less reconciler lane from both counts.
    # endregion
    excluded_tenant_ids = claim_excluded_tenant_ids()
    excluded: Select[Any] | None = None
    claimable_tenant_clause: ColumnElement[bool] | None = None
    if excluded_tenant_ids:
        claimable_tenant_clause = or_(
            models.QueueItemBatch.tenant_id.is_(None),
            models.QueueItemBatch.tenant_id.notin_(excluded_tenant_ids),
        )
        excluded = (
            select(func.count())
            .select_from(models.QueueItemBatch)
            .where(
                not_live_claimed_work_unit_clause(models.QueueItemBatch.work_unit_key),
                models.QueueItemBatch.tenant_id.in_(excluded_tenant_ids),
            )
        )
        if threshold_clause is not None:
            excluded = excluded.where(threshold_clause)
        eligible = eligible.where(claimable_tenant_clause)

    claimed = (
        select(func.count())
        .select_from(models.ActiveQueueSession)
        .where(models.ActiveQueueSession.last_updated >= stale_claim_cutoff())
    )

    pending = select(
        func.coalesce(func.sum(models.QueueItemBatch.pending_count), 0),
        func.coalesce(
            func.extract(
                "epoch",
                func.now() - func.min(models.QueueItemBatch.oldest_created_at),
            ),
            0,
        ),
    )
    if claimable_tenant_clause is not None:
        pending = pending.where(claimable_tenant_clause)

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
    excluded_count = (
        (await db.execute(excluded)).scalar_one() if excluded is not None else 0
    )
    claimed_count = (await db.execute(claimed)).scalar_one()
    pending_count, oldest_age = (await db.execute(pending)).one()
    embeddings_pending, embeddings_due = (await db.execute(embeddings)).one()

    return schemas.DeriverMetrics(
        eligible_work_units=int(eligible_count),
        excluded_work_units=int(excluded_count),
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
