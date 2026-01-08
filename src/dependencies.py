import uuid
from contextlib import asynccontextmanager

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db import SessionLocal, request_context


async def get_db():
    """FastAPI Dependency Generator for Database"""

    context = request_context.get() or "unknown"

    db: AsyncSession = SessionLocal()
    try:
        if settings.DB.TRACING:
            await db.execute(text(f"SET application_name = '{context}'"))
        yield db
    except Exception:
        await db.rollback()
        raise
    finally:
        if db.in_transaction():
            await db.rollback()
        await db.close()


@asynccontextmanager
async def tracked_db(operation_name: str | None = None):
    """Context manager for tracked database sessions"""
    # Get request ID if available, or create operation-specific one
    context = request_context.get()
    token = None

    if not context and operation_name:
        context = f"task:{operation_name}:{str(uuid.uuid4())[:8]}"
        token = request_context.set(context)

    # Create session with tracking info
    db = SessionLocal()

    try:
        if settings.DB.TRACING:
            await db.execute(
                text(f"SET application_name = '{context or f'task:{operation_name}'}'")
            )

        yield db
    except Exception:
        await db.rollback()
        raise
    finally:
        if db.in_transaction():
            await db.rollback()
        await db.close()
        if token:  # Only reset if we set it
            request_context.reset(token)


db: AsyncSession = Depends(get_db)
