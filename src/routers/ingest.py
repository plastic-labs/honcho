"""The ingest route: source records in, messages out, mapped and deduplicated.

`POST /v3/workspaces/{workspace_id}/ingest` takes up to 100 records and answers
one outcome per record. The write runs in src/crud/ingest.py; this module
authorizes it, retries it on transient lock failures, and does the post-commit
work the message route does (events, metrics, immediate embedding), then queues
the new records' derivation (src/deriver/ingest_enqueue.py).

`POST /v3/workspaces/{workspace_id}/ingest/derive` releases records ingested with
`derive: "later"` and queues them the same way.
"""

import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from functools import partial
from typing import Any, TypeVar

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path
from fastapi_pagination.bases import AbstractParams
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.cache.client import safe_cache_delete
from src.config import settings
from src.crud.ingest import (
    IngestRaced,
    after_commit,
    count_deferred,
    ingest_batch,
    release_deferred,
)
from src.crud.ingest_link import link_peers, reset_import, unlink_account
from src.crud.ingest_status import (
    get_ingest_status,
    people_query,
    require_auto_workspace,
    session_summaries,
    sessions_query,
)
from src.crud.peer import peer_cache_key
from src.crud.session import session_cache_key
from src.dependencies import db, read_db
from src.deriver.ingest_enqueue import enqueue_session
from src.dreamer.dream_scheduler import get_dream_scheduler
from src.reconciler.embed_now import embed_task_gate
from src.security import require_auth
from src.telemetry import prometheus_metrics
from src.telemetry.events import MessageCreatedEvent, emit
from src.utils.pagination import (
    CursorPage,
    Page,
    paginate_offset_or_cursor,
    pagination_params,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces/{workspace_id}/ingest", tags=["ingest"])

# Deadlock, serialization failure, lock not available (lock_timeout).
_RETRYABLE_SQLSTATES = frozenset({"40P01", "40001", "55P03"})
_ATTEMPTS = 3

_T = TypeVar("_T")


def _sqlstate(error: DBAPIError) -> str | None:
    original = getattr(error, "orig", None)
    return getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)


async def _with_retries(
    db: AsyncSession, workspace_name: str, write: Callable[[], Awaitable[_T]]
) -> _T:
    """Run a write transaction, retrying it whole on transient lock failures.

    Raises:
        HTTPException: 503 once the retries are spent; nothing was committed.
    """
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return await write()
        except DBAPIError as error:
            await db.rollback()
            if _sqlstate(error) not in _RETRYABLE_SQLSTATES:
                raise
            reason = _sqlstate(error)
        except IngestRaced:
            await db.rollback()
            reason = "a concurrent batch or reset changed what it read"
        logger.info(
            "Ingest attempt %d/%d in workspace %s hit contention (%s)",
            attempt,
            _ATTEMPTS,
            workspace_name,
            reason,
        )
    raise HTTPException(
        status_code=503,
        detail="Ingest could not finish because of lock contention; nothing was"
        + " written. Resend the same request.",
        headers={"Retry-After": "1"},
    )


async def _enqueue(
    db: AsyncSession, workspace_name: str, session_names: Iterable[str]
) -> set[str]:
    """Queue the sessions' pending records now; return the sessions fully queued.

    A session whose fence is held (another batch or a deletion on it) or whose
    enqueue fails keeps its records `pending`, and the reconciler's sweep queues
    them; the records' own transaction has committed either way.
    """
    queued: set[str] = set()
    for session_name in sorted(set(session_names)):
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
    outcome = await _with_retries(
        db, workspace_id, lambda: ingest_batch(db, workspace_id, batch)
    )

    await after_commit(workspace_id, outcome)
    await _cancel_dreams(workspace_id, outcome.authors)
    queued: set[str] = set()
    if batch.derive == "now":
        queued = await _enqueue(db, workspace_id, outcome.created_messages)
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


@router.post(
    "/derive",
    response_model=schemas.IngestDeriveResponse,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Workspace not found"},
        503: {
            "model": schemas.ErrorResponse,
            "description": "Contention outlasted the retries; nothing was released."
            + " Calling again is safe.",
        },
    },
)
async def derive(
    workspace_id: str = Path(..., description="ID of the workspace"),
    db: AsyncSession = db,
) -> schemas.IngestDeriveResponse:
    """Derive records that were ingested with `derive: "later"`.

    Each call releases up to 1,000 deferred records, a session at a time, and
    queues their derivation. Call again until `remaining` is 0. Configuration
    and observers are resolved when a record is queued, not when it was
    ingested. Only workspace and admin keys may call it.
    """
    released = await _with_retries(
        db, workspace_id, lambda: release_deferred(db, workspace_id)
    )
    await _enqueue(db, workspace_id, released.sessions)
    remaining = await count_deferred(db, workspace_id)
    await db.commit()
    return schemas.IngestDeriveResponse(pending=released.pending, remaining=remaining)


@router.get(
    "/status",
    response_model=schemas.IngestStatus,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Workspace not found"},
        409: {"model": schemas.ErrorResponse, "description": "Not an auto workspace"},
    },
)
async def status(
    workspace_id: str = Path(..., description="ID of the workspace"),
    db: AsyncSession = read_db,
) -> schemas.IngestStatus:
    """What the workspace's imports produced, for review before deriving.

    Records by derivation state, the tokens still deferred, and how many people
    and sessions the records map to, with how many of each are not modeled.
    Only workspace and admin keys may call it.
    """
    workspace = await require_auto_workspace(db, workspace_id)
    return await get_ingest_status(db, workspace)


@router.get(
    "/people",
    response_model=Page[schemas.IngestPerson] | CursorPage[schemas.IngestPerson],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Workspace not found"},
        409: {"model": schemas.ErrorResponse, "description": "Not an auto workspace"},
    },
)
async def people(
    workspace_id: str = Path(..., description="ID of the workspace"),
    params: AbstractParams = Depends(pagination_params),
    db: AsyncSession = read_db,
):
    """The peers created for imported accounts, most records first.

    Each lists its accounts (source, ID and display name), how many records
    they wrote, and whether the peer is modeled. Only workspace and admin keys
    may call it.
    """
    await require_auto_workspace(db, workspace_id)
    # Rows, not entities: there is nothing to deduplicate, and the accounts
    # column holds JSON, which can't be hashed.
    return await paginate_offset_or_cursor(
        db, people_query(workspace_id), params, unique=False
    )


@router.get(
    "/sessions",
    response_model=Page[schemas.IngestSessionSummary]
    | CursorPage[schemas.IngestSessionSummary],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Workspace not found"},
        409: {"model": schemas.ErrorResponse, "description": "Not an auto workspace"},
    },
)
async def sessions(
    workspace_id: str = Path(..., description="ID of the workspace"),
    params: AbstractParams = Depends(pagination_params),
    db: AsyncSession = read_db,
):
    """The sessions created for imported containers, most records first.

    Each gives its container's display name and source, how many records it
    holds, and whether reasoning is on for it. Only workspace and admin keys may
    call it.
    """
    workspace = await require_auto_workspace(db, workspace_id)
    # Rows, not entities: there is nothing to deduplicate, and their JSON
    # columns can't be hashed. `modeled` is resolved per row as the deriver
    # resolves a session's reasoning setting.
    return await paginate_offset_or_cursor(
        db,
        sessions_query(workspace_id),
        params,
        unique=False,
        transformer=partial(session_summaries, workspace=workspace),
    )


_LINK_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": schemas.ErrorResponse, "description": "Workspace or peer not found"},
    409: {
        "model": schemas.ErrorResponse,
        "description": "Not an auto workspace, or the people involved already have"
        + " imported messages: reset the import first",
    },
    503: {
        "model": schemas.ErrorResponse,
        "description": "Contention outlasted the retries; nothing changed. Calling"
        + " again is safe.",
    },
}


@router.post(
    "/link",
    response_model=schemas.IngestLinkResult,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
    responses=_LINK_RESPONSES,
)
async def link(
    workspace_id: str = Path(..., description="ID of the workspace"),
    body: schemas.IngestLink = Body(...),
    db: AsyncSession = db,
) -> schemas.IngestLinkResult:
    """Link imported people into one, from the next import on.

    Every account of the other people maps to the first from now on, so the
    next import puts their records on one person. Nothing already imported
    moves: the people being merged must have no imported messages yet, so
    reset the import (`POST /ingest/reset`), link, then import again. Only
    workspace and admin keys may call it.
    """
    await require_auto_workspace(db, workspace_id)
    result = await _with_retries(
        db, workspace_id, lambda: link_peers(db, workspace_id, body.peers)
    )
    await db.commit()
    for name in body.peers:
        await safe_cache_delete(peer_cache_key(workspace_id, name))
    return result


@router.post(
    "/unlink",
    response_model=schemas.IngestUnlinkResult,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
    responses=_LINK_RESPONSES,
)
async def unlink(
    workspace_id: str = Path(..., description="ID of the workspace"),
    body: schemas.IngestUnlink = Body(...),
    db: AsyncSession = db,
) -> schemas.IngestUnlinkResult:
    """Take a linked account back out, from the next import on.

    The account maps to the person it was linked from again. Its records must
    not be imported yet: reset the import first. Only workspace and admin keys
    may call it.
    """
    await require_auto_workspace(db, workspace_id)
    result = await _with_retries(
        db,
        workspace_id,
        lambda: unlink_account(db, workspace_id, body.peer, body.account),
    )
    await db.commit()
    for name in (body.peer, result.peer_id):
        await safe_cache_delete(peer_cache_key(workspace_id, name))
    return result


@router.post(
    "/reset",
    response_model=schemas.IngestResetResponse,
    status_code=202,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
    responses=_LINK_RESPONSES,
)
async def reset(
    workspace_id: str = Path(..., description="ID of the workspace"),
    db: AsyncSession = db,
) -> schemas.IngestResetResponse:
    """Delete every session holding imported records, to import again.

    Each session is marked inactive at once and deleted in the background, as
    `DELETE /sessions/{id}` does, with its messages, memberships and ingest
    records. People, their configuration and links stay; a session's own
    configuration goes with it. `GET /ingest/status` counts only active
    sessions, so it shows the reset straight away. Only workspace and admin
    keys may call it.
    """
    await require_auto_workspace(db, workspace_id)
    sessions = await _with_retries(
        db, workspace_id, lambda: reset_import(db, workspace_id)
    )
    await db.commit()
    for name in sessions:
        await safe_cache_delete(session_cache_key(workspace_id, name))
    return schemas.IngestResetResponse(sessions=sessions)
