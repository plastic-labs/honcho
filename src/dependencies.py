import uuid
from contextlib import asynccontextmanager

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import ReadSessionLocal, SessionLocal, request_context


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
    except Exception:
        await db.rollback()
        raise
    finally:
        # Always send ROLLBACK unconditionally so the wire-level transaction
        # is closed before the TCP connection drops.  Supavisor v2 does NOT
        # clean up orphaned transactions on client disconnect in transaction-
        # pooling mode, so relying on `in_transaction()` (Python-side state)
        # can leave the backend pinned with an open BEGIN. (Cheap no-op if the
        # lazy session never checked out a connection.)
        await db.rollback()
        await db.close()


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
        # rollback is a wire-level no-op under AUTOCOMMIT; kept to reset any
        # Python-side session state before close, mirroring get_db.
        await db.rollback()
        await db.close()


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
    except Exception:
        await db.rollback()
        raise
    finally:
        # Always send ROLLBACK unconditionally — see get_db() comment. (Under
        # read_only/AUTOCOMMIT it is a wire-level no-op.)
        await db.rollback()
        await db.close()
        if token:  # Only reset if we set it
            request_context.reset(token)


db: AsyncSession = Depends(get_db)
read_db: AsyncSession = Depends(get_read_db)
