import contextvars

from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from src.config import settings

connect_args = {"prepare_threshold": None}

# Context variable to store request context
request_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_context", default=None
)

engine_kwargs = {}

if settings.DB.POOL_CLASS == "null":
    engine_kwargs["poolclass"] = NullPool
else:
    # Only add pool-related kwargs for pooled connections
    engine_kwargs.update(  # pyright: ignore
        {
            "pool_pre_ping": settings.DB.POOL_PRE_PING,
            "pool_size": settings.DB.POOL_SIZE,
            "max_overflow": settings.DB.MAX_OVERFLOW,
            "pool_timeout": settings.DB.POOL_TIMEOUT,
            "pool_recycle": settings.DB.POOL_RECYCLE,
            "pool_use_lifo": settings.DB.POOL_USE_LIFO,
        }
    )

engine = create_async_engine(
    settings.DB.CONNECTION_URI,
    connect_args=connect_args,
    echo=settings.DB.SQL_DEBUG,
    **engine_kwargs,
)

SessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)

table_schema = settings.DB.SCHEMA
meta = MetaData()
meta.schema = table_schema
Base = declarative_base(metadata=meta)


async def init_db():
    """Initialize the database using Alembic migrations"""
    from alembic import command
    from alembic.config import Config

    async with engine.connect() as connection:
        # Create schema if it doesn't exist
        await connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{table_schema}"'))
        # Install pgvector extension if it doesn't exist
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.commit()

    # Run Alembic migrations
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
