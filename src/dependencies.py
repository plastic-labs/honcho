import uuid
from contextlib import asynccontextmanager

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import SessionLocal, request_context


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


@asynccontextmanager
async def tracked_db(operation_name: str | None = None):
    """Context manager for tracked database sessions.

    Sets a task-scoped request_context so the lazy session picks it up for
    tracing/attribution, then yields a lazy session (see get_db).
    """
    # Get request ID if available, or create operation-specific one
    context = request_context.get()
    token = None

    if not context and operation_name:
        context = f"task:{operation_name}:{str(uuid.uuid4())[:8]}"
        token = request_context.set(context)

    db = SessionLocal()
    try:
        yield db
    except Exception:
        await db.rollback()
        raise
    finally:
        # Always send ROLLBACK unconditionally — see get_db() comment.
        await db.rollback()
        await db.close()
        if token:  # Only reset if we set it
            request_context.reset(token)


db: AsyncSession = Depends(get_db)
