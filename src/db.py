import contextvars
import os
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

load_dotenv()

connect_args = {"prepare_threshold": None}

# Context variable to store request context
request_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_context", default=None
)

engine = create_async_engine(
    os.environ["CONNECTION_URI"],
    connect_args=connect_args,
    echo=os.getenv("SQL_DEBUG", "false").lower() == "true",  # Only enable in debug mode
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=300,  # Recycle connections after 5 minutes
    pool_use_lifo=True,  # Use last-in-first-out (LIFO) to prevent connection spread
)

SessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)

table_schema = os.getenv("DATABASE_SCHEMA", "public")
meta = MetaData()
meta.schema = table_schema
Base = declarative_base(metadata=meta)


def init_db():
    """Initialize the database using Alembic migrations"""
    from alembic import command
    from alembic.config import Config

    # Create a sync engine for schema operations
    sync_engine = create_engine(
        os.environ["CONNECTION_URI"],
        pool_pre_ping=True,
        echo=os.getenv("SQL_DEBUG", "false").lower() == "true",
    )

    with sync_engine.connect() as connection:
        # Create schema if it doesn't exist
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{table_schema}"'))
        # Install pgvector extension if it doesn't exist
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()

    # Run Alembic migrations
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    # Clean up
    sync_engine.dispose()
