from contextlib import asynccontextmanager

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .db import SessionLocal, request_context


async def get_db():
    """FastAPI Dependency Generator for Database"""

    context = request_context.get() or "unknown"

    db: AsyncSession = SessionLocal()
    try:
        await db.execute(text(f"SET application_name = '{context}'"))
        await db.commit()
        yield db
    except Exception:
        await db.rollback()
        raise
    finally:
        if db.in_transaction():
            await db.rollback()
        await db.close()


@asynccontextmanager
async def tracked_db(operation_name=None):
    """Context manager for tracked database sessions"""
    # Generate a unique ID for this DB session
    token = None

    # Always prioritize the operation_name if provided
    if operation_name:
        app_name = f"task:{operation_name}"
        # Only modify the context var if we're not in a request context
        if not request_context.get():
            token = request_context.set(app_name)
    else:
        # Fallback to request context if no operation name provided
        context = request_context.get()
        app_name = context if context else "task:unspecified"

    # Create session with tracking info
    db = SessionLocal()

    try:
        await db.execute(text(f"SET application_name = '{app_name}'"))
        await db.commit()

        yield db
        # Explicitly end transaction if still open
        if db.in_transaction():
            await db.rollback()  # Or commit if needed for write operations
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()
        if token:  # Only reset if we set it
            request_context.reset(token)


db: AsyncSession = Depends(get_db)
