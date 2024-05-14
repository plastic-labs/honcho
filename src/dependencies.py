from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from .db import SessionLocal


async def get_db():
    """FastAPI Dependency Generator for Database"""
    db: AsyncSession = SessionLocal()
    try:
        yield db
    finally:
        await db.close()


db: AsyncSession = Depends(get_db)
