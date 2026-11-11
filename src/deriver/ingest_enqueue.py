"""The ingest enqueuer: turns `pending` ingest records into queue work.

The second transaction of the ingest write path (src/crud/ingest.py is the
first). Per session, in batches, each its own transaction:

1. Try the ingest fence (src/crud/ingest_fence.py): the workspace lock shared,
   then the session lock. A held lock means an ingest batch or a deletion is on
   this session right now; the session is skipped and the reconciler's sweep
   comes back to it.
2. Read the session's `pending` records with their messages, without row locks.
3. Build the same queue work the message route builds for those messages
   (`generate_queue_records`), resolving configuration and observers now, as the
   message route's enqueue does, and insert it in work_unit_key order.
4. Mark the records `queued`, delete any whose message no longer exists, commit.

Its lock order is fence, then queue, then ingest rows: the same as deletion's,
so the two can't cycle, and holding the session fence means it never queues work
for a session mid-deletion or races another enqueuer on the same session.
"""

import asyncio
import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, and_, delete, exists, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud.ingest_fence import take_session_fences, take_workspace_fence
from src.dependencies import service_db, tracked_db
from src.deriver.enqueue import (
    generate_queue_records,
    get_peers_with_configuration,
    insert_queue_records_in_transaction,
)
from src.schemas import MessageConfiguration
from src.utils.config_helpers import get_configuration
from src.utils.retryable_errors import is_retryable_error

logger = logging.getLogger(__name__)

# Records per transaction, so one large backfilled session doesn't hold its
# fence and a huge queue insert in a single transaction.
_BATCH_SIZE = 500
_ATTEMPTS = 3

# The sweep leaves records this new alone: the request that wrote them is
# enqueueing them right after its own commit.
SWEEP_MIN_AGE = datetime.timedelta(seconds=30)
# Pending records younger than this count as deriver work, so a fleet scaled to
# zero wakes up to sweep them. Older ones (a session that keeps failing to
# enqueue) stay in the pending gauge and are still swept whenever a deriver
# runs, but can't keep the fleet awake on their own.
SWEEP_WAKE_WINDOW = datetime.timedelta(hours=1)
# Sessions one sweep run enqueues; the next run continues after the last one.
_SWEEP_SESSIONS = 200

# Where the next sweep run in this process starts, in (tenant, workspace,
# session) order. Advancing past every session a run visited, whether it was
# queued, fenced or failed, keeps sessions that can't be queued from taking every
# run's slots; at the end of the order it wraps around.
_sweep_cursor: tuple[str, str, str] | None = None


@dataclass
class EnqueueOutcome:
    """What enqueueing one session did."""

    queued: int = 0
    # Records deleted because their message no longer exists.
    orphaned: int = 0
    # The fence was held, so pending records may be left for the sweep.
    skipped: bool = False


async def enqueue_session(
    db: AsyncSession, workspace_name: str, session_name: str
) -> EnqueueOutcome:
    """Queue every `pending` record of one session, batch by batch.

    `db` must be a tenant-bound write session with no transaction in progress.
    A transient failure (deadlock, serialization failure, lost connection) is
    retried per batch; one that persists is raised, and the records stay
    `pending` for the sweep.
    """
    outcome = EnqueueOutcome()
    while True:
        batch = await _with_retries(db, workspace_name, session_name)
        if batch is None:
            outcome.skipped = True
            return outcome
        queued, orphaned = batch
        outcome.queued += queued
        outcome.orphaned += orphaned
        if queued + orphaned < _BATCH_SIZE:
            return outcome


async def _with_retries(
    db: AsyncSession, workspace_name: str, session_name: str
) -> tuple[int, int] | None:
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return await _enqueue_batch(db, workspace_name, session_name)
        except Exception as error:
            await db.rollback()
            if attempt == _ATTEMPTS or not is_retryable_error(error):
                raise
            await asyncio.sleep(0.1 * attempt)
    raise AssertionError("unreachable")


async def _enqueue_batch(
    db: AsyncSession, workspace_name: str, session_name: str
) -> tuple[int, int] | None:
    """One transaction: queue up to a batch of records. None when fenced out."""
    if not await take_workspace_fence(
        db, workspace_name, mode="shared", try_lock=True
    ) or not await take_session_fences(
        db, workspace_name, [session_name], try_lock=True
    ):
        await db.rollback()
        return None

    tenant_id = models.scoped_tenant_id()
    record = models.IngestRecord
    message = models.Message
    rows = (
        await db.execute(
            select(
                record.id,
                record.configuration,
                message.id,
                message.public_id,
                message.content,
                message.peer_name,
                message.created_at,
                message.seq_in_session,
            )
            .outerjoin(
                message,
                (message.tenant_id == record.tenant_id)
                & (message.public_id == record.message_public_id),
            )
            .where(
                record.tenant_id == tenant_id,
                record.workspace_name == workspace_name,
                record.session_name == session_name,
                record.derivation == "pending",
            )
            .order_by(message.seq_in_session.nulls_last(), record.id)
            .limit(_BATCH_SIZE)
        )
    ).all()
    if not rows:
        await db.rollback()
        return 0, 0

    live = [row for row in rows if row[2] is not None]
    orphaned_ids = [row[0] for row in rows if row[2] is None]

    session = await db.scalar(
        select(models.Session).where(
            models.Session.tenant_id == tenant_id,
            models.Session.workspace_name == workspace_name,
            models.Session.name == session_name,
        )
    )
    if session is None:
        # Deletion removes a session's ingest records under this fence, so this
        # only happens if a session goes some other way; nothing can be queued.
        orphaned_ids = [row[0] for row in rows]
        live = []
    else:
        workspace = await db.scalar(
            select(models.Workspace).where(
                models.Workspace.tenant_id == tenant_id,
                models.Workspace.name == workspace_name,
            )
        )
        queue_records = await _queue_records(
            db, workspace_name, session, workspace, live
        )
        await insert_queue_records_in_transaction(db, queue_records)

    if live:
        await db.execute(
            update(record)
            .where(
                record.tenant_id == tenant_id, record.id.in_([row[0] for row in live])
            )
            .values(derivation="queued")
        )
    if orphaned_ids:
        await db.execute(
            delete(record).where(
                record.tenant_id == tenant_id, record.id.in_(orphaned_ids)
            )
        )
    await db.commit()
    return len(live), len(orphaned_ids)


async def _queue_records(
    db: AsyncSession,
    workspace_name: str,
    session: models.Session,
    workspace: models.Workspace | None,
    rows: Sequence[Any],
) -> list[dict[str, Any]]:
    """The queue work the message route would build for these messages."""
    if not rows:
        return []
    peers = await get_peers_with_configuration(db, workspace_name, session.name)
    queue_records: list[dict[str, Any]] = []
    for (
        _record_id,
        stored_configuration,
        message_id,
        public_id,
        content,
        peer_name,
        created_at,
        seq_in_session,
    ) in rows:
        configuration = (
            MessageConfiguration.model_validate(stored_configuration)
            if stored_configuration
            else None
        )
        payload = {
            "workspace_name": workspace_name,
            "session_name": session.name,
            "message_id": message_id,
            "content": content,
            "peer_name": peer_name,
            "created_at": created_at,
            "message_public_id": public_id,
            "seq_in_session": seq_in_session,
            "configuration": configuration,
        }
        queue_records.extend(
            await generate_queue_records(
                db,
                payload,
                peers,
                session.id,
                get_configuration(configuration, session, workspace),
            )
        )
    return queue_records


# ---------------------------------------------------------------------------
# The reconciler's sweep
# ---------------------------------------------------------------------------


def sweep_due_clause() -> ColumnElement[bool]:
    """Pending records a sweep should be scheduled, and a deriver woken, for.

    Old enough that the sweep, not their own request, queues them; young enough
    (SWEEP_WAKE_WINDOW) that a session failing forever can't keep scheduling
    sweeps or holding derivers awake. Older pending records are still found by
    any sweep that runs, since its discovery has no upper bound.
    """
    now = datetime.datetime.now(datetime.UTC)
    record = models.IngestRecord
    return and_(
        record.derivation == "pending",
        record.created_at < now - SWEEP_MIN_AGE,
        record.created_at >= now - SWEEP_WAKE_WINDOW,
    )


async def has_pending_ingest(db: AsyncSession) -> bool:
    """Whether a sweep is worth scheduling (see `sweep_due_clause`)."""
    return bool(await db.scalar(select(exists().where(sweep_due_clause()))))


@dataclass
class SweepResult:
    """What one sweep run did."""

    sessions: int = 0
    queued: int = 0
    orphaned: int = 0
    skipped: int = 0
    failed: int = 0


async def sweep_pending_ingest() -> SweepResult:
    """Enqueue sessions whose records were left `pending`.

    Records are left `pending` when the request that wrote them died before
    enqueueing, failed to, or found the session fenced. Sessions are found
    across tenants on the service session (read only), then each is enqueued
    on a session bound to its own tenant, so the writes run under RLS. One
    failing session is logged and doesn't stop the others.
    """
    global _sweep_cursor
    cutoff = datetime.datetime.now(datetime.UTC) - SWEEP_MIN_AGE
    record = models.IngestRecord
    key = (record.tenant_id, record.workspace_name, record.session_name)
    statement = (
        select(*key)
        .where(record.derivation == "pending", record.created_at < cutoff)
        .group_by(*key)
        .order_by(*key)
        .limit(_SWEEP_SESSIONS)
    )
    if _sweep_cursor is not None:
        statement = statement.where(tuple_(*key) > _sweep_cursor)
    async with service_db("ingest_sweep.discover", read_only=True) as db:
        sessions = [
            (str(tenant), str(workspace), str(session))
            for tenant, workspace, session in (await db.execute(statement)).all()
        ]
    _sweep_cursor = sessions[-1] if len(sessions) == _SWEEP_SESSIONS else None

    result = SweepResult()
    for tenant_id, workspace_name, session_name in sessions:
        result.sessions += 1
        try:
            async with tracked_db(
                "ingest_sweep.enqueue",
                tenant_id=tenant_id if settings.MULTI_TENANT else None,
            ) as db:
                outcome = await enqueue_session(db, workspace_name, session_name)
        except Exception:
            result.failed += 1
            logger.exception(
                "Couldn't enqueue pending ingest records for %s/%s",
                workspace_name,
                session_name,
            )
            continue
        result.queued += outcome.queued
        result.orphaned += outcome.orphaned
        result.skipped += int(outcome.skipped)
    return result
