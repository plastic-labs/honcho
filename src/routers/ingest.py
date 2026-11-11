"""The ingest route: source records in, messages out, mapped and deduplicated.

`POST /v3/workspaces/{workspace_id}/ingest` takes up to 100 records and answers
one outcome per record. The write runs in src/crud/ingest.py; this module
authorizes it, retries it on transient lock failures, and does the post-commit
work the message route does (events, metrics, immediate embedding), then queues
the new records' derivation (src/deriver/ingest_enqueue.py).
"""

import logging
from collections import Counter

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.config import settings
from src.crud.ingest import IngestRaced, IngestResult, after_commit, ingest_batch
from src.dependencies import db
from src.deriver.ingest_enqueue import enqueue_session
from src.dreamer.dream_scheduler import get_dream_scheduler
from src.reconciler.embed_now import embed_task_gate
from src.security import require_auth
from src.telemetry import prometheus_metrics
from src.telemetry.events import MessageCreatedEvent, emit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces/{workspace_id}/ingest", tags=["ingest"])

# Deadlock, serialization failure, lock not available (lock_timeout).
_RETRYABLE_SQLSTATES = frozenset({"40P01", "40001", "55P03"})
_ATTEMPTS = 3


def _sqlstate(error: DBAPIError) -> str | None:
    original = getattr(error, "orig", None)
    return getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)


async def _enqueue(
    db: AsyncSession, workspace_name: str, outcome: IngestResult
) -> set[str]:
    """Queue the new records' work now; return the sessions fully queued.

    A session whose fence is held (another batch or a deletion on it) or whose
    enqueue fails keeps its records `pending`, and the reconciler's sweep queues
    them; the batch itself has committed either way.
    """
    queued: set[str] = set()
    for session_name in sorted(outcome.created_messages):
        try:
            enqueued = await enqueue_session(db, workspace_name, session_name)
        except Exception:
            await db.rollback()
            logger.exception(
                "Couldn't enqueue ingested records for %s/%s; the sweep will",
                workspace_name,
                session_name,
            )
            continue
        if not enqueued.skipped:
            queued.add(session_name)
    return queued


def _record_outcomes(results: list[schemas.IngestRecordResult]) -> None:
    counts = Counter((result.outcome, result.reason or "") for result in results)
    for (outcome, reason), count in counts.items():
        prometheus_metrics.record_ingest_records(
            outcome=outcome, reason=reason, count=count
        )


async def _cancel_dreams(workspace_name: str, authors: set[str]) -> None:
    """Cancel the authors' pending dreams: they're active again. Best-effort."""
    scheduler = get_dream_scheduler()
    if scheduler is None:
        return
    for peer_name in sorted(authors):
        try:
            await scheduler.cancel_dreams_for_observed(workspace_name, peer_name)
        except Exception:
            logger.exception(
                "Couldn't cancel pending dreams for %s in %s", peer_name, workspace_name
            )


@router.post(
    "",
    response_model=schemas.IngestResponse,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Workspace not found"},
        409: {
            "model": schemas.ErrorResponse,
            "description": "The workspace isn't an auto workspace",
        },
        503: {
            "model": schemas.ErrorResponse,
            "description": "Contention outlasted the retries; nothing was written."
            + " Resending the same batch is safe.",
        },
    },
)
async def ingest(
    background_tasks: BackgroundTasks,
    workspace_id: str = Path(..., description="ID of the workspace"),
    batch: schemas.IngestBatch = Body(..., description="The records to ingest"),
    db: AsyncSession = db,
) -> schemas.IngestResponse:
    """Ingest source records as messages.

    Honcho maps each record to a session and peers from its source context and
    deduplicates it on its source identity (`source_id`, or a hash of its
    content), so resending a batch is always safe. The response has one outcome
    per record, in request order: `created`, `duplicate`, or `rejected` with a
    reason. The workspace must be an auto workspace (created with `auto: true`).
    Only workspace and admin keys may ingest: one batch can create sessions and
    peers anywhere in the workspace.
    """
    outcome: IngestResult | None = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            outcome = await ingest_batch(db, workspace_id, batch)
            break
        except DBAPIError as error:
            await db.rollback()
            if _sqlstate(error) not in _RETRYABLE_SQLSTATES:
                raise
            reason = _sqlstate(error)
        except IngestRaced:
            await db.rollback()
            reason = "a concurrent batch bound the same source"
        logger.info(
            "Ingest attempt %d/%d in workspace %s hit contention (%s)",
            attempt,
            _ATTEMPTS,
            workspace_id,
            reason,
        )
    if outcome is None:
        raise HTTPException(
            status_code=503,
            detail="Ingest could not finish because of lock contention; nothing was"
            + " written. Resend the same batch.",
            headers={"Retry-After": "1"},
        )

    await after_commit(workspace_id, outcome)
    await _cancel_dreams(workspace_id, outcome.authors)
    queued: set[str] = set()
    if batch.derive == "now":
        queued = await _enqueue(db, workspace_id, outcome)
    # Everything pending in a fully queued session is queued now, including a
    # duplicate of a record this batch or an earlier one left pending.
    results = [
        result.model_copy(update={"derivation": "queued"})
        if result.derivation == "pending" and result.session_id in queued
        else result
        for result in outcome.results
    ]
    if settings.METRICS.ENABLED:
        _record_outcomes(results)

    for session_name, message_ids in outcome.created_messages.items():
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_messages_created(
                count=len(message_ids), workspace_name=workspace_id
            )
        emit(
            MessageCreatedEvent(
                workspace_name=workspace_id,
                session_name=session_name,
                message_count=len(message_ids),
                total_tokens=outcome.created_token_counts[session_name],
                source="ingest",
                last_message_id=message_ids[-1],
            )
        )
        # Embed immediately so ingested messages are searchable within seconds;
        # the reconciler is the fallback, as on the message route.
        if settings.EMBED_MESSAGES:
            embed_task_gate.try_schedule(background_tasks, message_ids)

    return schemas.IngestResponse(records=results)
