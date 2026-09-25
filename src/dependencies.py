import asyncio
import contextlib
import logging
import uuid
from collections.abc import Coroutine
from typing import Any

from fastapi import Depends
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import ReadSessionLocal, SessionLocal, request_context

logger = logging.getLogger(__name__)

# Upper bound on how long shielded cleanup may run before we stop waiting on it.
# Healthy ROLLBACK/close take milliseconds; this only bites a wedged connection
# (e.g. a black-holed socket with no libpq timeout) so it can't pin the task
# forever. Cleanup is abandoned (not cancelled) past this — the pool/OS reclaims
# the connection — since a shielded task cannot be interrupted anyway.
_CLEANUP_TIMEOUT_SECONDS = 10.0


async def _run_to_completion(coro: Coroutine[Any, Any, None]) -> None:
    """Run ``coro`` to completion even if the current task is cancelled (DEV-1861).

    Runs the coro as a detached task and waits on it via ``asyncio.wait`` (which
    never cancels the inner task), re-raising the cancellation only afterward.
    Survives repeated re-delivery. NB: ``anyio.CancelScope(shield=True)`` does
    NOT work here — it ignores a native ``asyncio.Task.cancel()``, which is how
    Starlette and the deriver cancel.
    """
    fut = asyncio.ensure_future(coro)
    cancelled = False
    loop = asyncio.get_running_loop()
    # One monotonic deadline for the whole loop, not a per-wait timeout: anyio
    # (Starlette's BaseHTTPMiddleware wraps every request in a task group)
    # re-delivers cancellation on every event-loop tick, so a per-wait timeout
    # re-arms each tick and never elapses.
    deadline = loop.time() + _CLEANUP_TIMEOUT_SECONDS
    while not fut.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            # Wedged cleanup (dead socket). Stop pinning this task; leave fut to
            # error out / be reclaimed rather than block indefinitely.
            logger.error(
                "DB session cleanup exceeded %.0fs; abandoning to avoid pinning the task",
                _CLEANUP_TIMEOUT_SECONDS,
            )
            break
        try:
            # wait() shields fut from our cancellation but still bounds the wait.
            await asyncio.wait({fut}, timeout=remaining)
        except asyncio.CancelledError:
            cancelled = True  # keep waiting so ROLLBACK/close reach Postgres
    if fut.cancelled():
        cancelled = True  # cleanup itself was cancelled (close() still ran)
    elif fut.done() and (exc := fut.exception()) is not None:
        logger.error("session cleanup task failed", exc_info=exc)
    if cancelled:
        raise asyncio.CancelledError()


async def _finalize_session(db: AsyncSession) -> None:
    """Roll back and close a session, releasing its pooled connection.

    close() is guaranteed by the inner ``finally`` even if the rollback await is
    interrupted — and returning the connection to the pool triggers a DBAPI-level
    ROLLBACK on reset, so the transaction closes regardless. A broken-mid-protocol
    connection is invalidated so the pool discards it instead of reusing it.
    """
    try:
        await db.rollback()
    except (InterfaceError, OperationalError, DBAPIError) as e:
        logger.warning("cleanup ROLLBACK failed; invalidating connection: %s", e)
        with contextlib.suppress(Exception):
            await db.invalidate()
    except Exception:
        logger.exception("unexpected error during cleanup ROLLBACK")
    finally:
        with contextlib.suppress(Exception):
            await db.close()


async def get_db():
    """FastAPI Dependency Generator for Database.

    The session is lazy: it does NOT check out a pooled connection here. The
    AsyncSession checks one out on the first DB-touching call, so a handler doing
    non-DB work (embedding/file/LLM) before its first query does not pin a
    connection across it.
    """
    db: AsyncSession = SessionLocal()
    try:
        yield db
    finally:
        # Shielded so a cancelled request can't leak an open BEGIN (DEV-1861).
        await _run_to_completion(_finalize_session(db))


async def get_read_db():
    """FastAPI Dependency Generator for SELECT-only handlers.

    Same lazy-checkout semantics as get_db, but the session is bound to the
    AUTOCOMMIT read engine: no BEGIN is ever emitted, so the connection can not
    sit 'idle in transaction' between the query and this teardown — a delayed
    finally here is harmless (the backend is plain 'idle'). close() is still
    required to release the connection itself back to the pool.

    MUST only be used by handlers that never mutate; see ReadSessionLocal.
    """
    db: AsyncSession = ReadSessionLocal()
    try:
        yield db
    finally:
        await _run_to_completion(_finalize_session(db))


@contextlib.asynccontextmanager
async def tracked_db(operation_name: str | None = None, *, read_only: bool = False):
    """Context manager for tracked database sessions.

    Sets a task-scoped request_context so the lazy session picks it up for
    tracing/attribution, then yields a lazy session (see get_db).

    Pass read_only=True for SELECT-only windows: the session is then bound to
    the AUTOCOMMIT read engine, so the work inside the block never holds an
    open transaction (no idle-in-transaction parking; the pooler can reclaim
    the backend between statements). Never use read_only=True on a path that
    mutates — see ReadSessionLocal.
    """
    # Get request ID if available, or create operation-specific one
    context = request_context.get()
    token = None

    if not context and operation_name:
        context = f"task:{operation_name}:{str(uuid.uuid4())[:8]}"
        token = request_context.set(context)

    db = (ReadSessionLocal if read_only else SessionLocal)()
    try:
        yield db
    finally:
        try:
            await _run_to_completion(_finalize_session(db))  # shielded — see get_db
        finally:
            # Must run even when cleanup re-raises CancelledError, or a reused
            # long-lived task (e.g. the deriver) leaks this contextvar.
            if token:  # Only reset if we set it
                request_context.reset(token)


db: AsyncSession = Depends(get_db)
read_db: AsyncSession = Depends(get_read_db)
