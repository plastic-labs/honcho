import asyncio
import logging
import uuid
from collections.abc import Coroutine
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import ReadSessionLocal, SessionLocal, request_context

logger = logging.getLogger(__name__)


async def _run_to_completion(coro: Coroutine[Any, Any, None]) -> None:
    """Run ``coro`` to completion even if the current task is cancelled.

    This is the cancellation shield for DB-session cleanup (DEV-1861). A client
    disconnect cancels the request task, and the cancellation can be re-delivered
    onto the ROLLBACK/close awaits — leaving the backend parked 'idle in
    transaction' with an open BEGIN that Supavisor v2 does not reset.

    NOTE: ``anyio.CancelScope(shield=True)`` does NOT cover this: it only defers
    anyio-scoped cancellation, not a native ``asyncio.Task.cancel()`` (which is
    how the deriver's uvloop and the ASGI server actually cancel). We instead run
    cleanup as a detached task and keep awaiting it through ``asyncio.shield`` —
    which never cancels the inner task — re-raising the cancellation only after
    cleanup has fully finished. Survives repeated re-delivery (a cancel storm).
    """
    fut = asyncio.ensure_future(coro)
    cancelled = False
    while not fut.done():
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            # Our task was cancelled; the shielded cleanup keeps running. Note it
            # and keep waiting so ROLLBACK/close actually reach Postgres.
            cancelled = True
    exc = fut.exception()
    if exc is not None:  # _finalize_session swallows its own errors; be safe.
        logger.exception("session cleanup task failed", exc_info=exc)
    if cancelled:
        raise asyncio.CancelledError()


async def _finalize_session(db: AsyncSession) -> None:
    """Roll back and close a session, releasing its pooled connection.

    Runs via ``_run_to_completion`` (see callers) so it cannot be interrupted by
    task cancellation. Structurally, close() is ALSO guaranteed by an inner
    ``finally`` even if the rollback await is cut off — a second line of defense
    the shield backs up: if ROLLBACK is interrupted before reaching Postgres and
    close() never runs, the pooled connection is orphaned with an open BEGIN and
    the backend parks 'idle in transaction' indefinitely (DEV-1861). Supavisor
    v2 does NOT reset such orphaned transactions on client disconnect in
    transaction-pooling mode. (close() returning the connection to the pool
    triggers a DBAPI-level ROLLBACK on reset, so the transaction is closed even
    if the explicit rollback below was interrupted.)

    Each step is guarded independently: a failed ROLLBACK (e.g. the connection
    broke mid-protocol) must not prevent close(), and a broken connection is
    invalidated so the pool discards it rather than handing out a poisoned
    connection on the next checkout. ROLLBACK is unconditional (a wire-level
    no-op under read_only/AUTOCOMMIT, and cheap if the lazy session never
    checked out a connection).
    """
    try:
        await db.rollback()
    except (InterfaceError, OperationalError, DBAPIError) as e:
        # The connection is broken mid-protocol; the ROLLBACK never reached
        # Postgres. Invalidate so the pool discards the dead connection instead
        # of returning it. Do not let this mask the original error path.
        logger.warning("cleanup ROLLBACK failed; invalidating connection: %s", e)
        try:
            await db.invalidate()
        except Exception:
            logger.exception("failed to invalidate broken connection during cleanup")
    except Exception:
        logger.exception("unexpected error during cleanup ROLLBACK")
    finally:
        try:
            await db.close()
        except Exception:
            logger.exception("failed to close session during cleanup")


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
        # Cleanup must survive a re-delivered cancellation on the rollback/close
        # awaits, or the connection leaks with an open BEGIN (DEV-1861).
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


@asynccontextmanager
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
        # Run cleanup to completion despite (re-delivered) cancellation — see
        # get_db / _run_to_completion.
        await _run_to_completion(_finalize_session(db))
        if token:  # Only reset if we set it
            request_context.reset(token)


db: AsyncSession = Depends(get_db)
read_db: AsyncSession = Depends(get_read_db)
