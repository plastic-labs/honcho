from fastapi import Depends
from .db import SessionLocal
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db():
    """FastAPI Dependency Generator for Database"""
    db: AsyncSession = SessionLocal()
    try:
        yield db
    finally:
        await db.close()


db: AsyncSession = Depends(get_db)
