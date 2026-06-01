import contextvars

import sentry_sdk
from sqlalchemy import MetaData, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

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

# Errors worth retrying when acquiring a pooled connection: SQLAlchemy's local
# pool-checkout timeout, and OperationalError (how a saturated transaction
# pooler surfaces "too many clients" / connection refusals).
RETRYABLE_DB_CONNECTION_ERRORS = (SQLAlchemyTimeoutError, OperationalError)


def get_pool_stats() -> dict[str, int]:
    """Return live connection-pool stats for this process.

    ``engine.pool`` is the AsyncEngine's pool (the same object as
    ``engine.sync_engine.pool``); its stat methods are synchronous counter
    reads with no I/O, so they are safe to call without ``await``. Returns
    zeros for pools that do not track connections (e.g. ``NullPool``).
    """
    zeros = {"checked_out": 0, "checked_in": 0, "size": 0, "overflow": 0}
    pool = engine.pool
    # Only QueuePool (and its AsyncAdaptedQueuePool subclass) tracks connection
    # counts; NullPool and others have no meaningful stats.
    if not isinstance(pool, QueuePool):
        return zeros
    try:
        return {
            "checked_out": pool.checkedout(),
            "checked_in": pool.checkedin(),
            "size": pool.size(),
            "overflow": pool.overflow(),
        }
    except Exception:
        return zeros


async def acquire_connection_with_retry(db: AsyncSession, context: str) -> None:
    """Force pool checkout (which ``SessionLocal()`` defers) with bounded backoff.

    ``SessionLocal()`` is lazy: the pool checkout — and any pooler rejection —
    happens on the first query. We force it here inside a retry block so that
    transient saturation of the transaction pooler is retried with exponential
    backoff rather than surfacing as an immediate error. The checkout is wrapped
    in a Sentry span so wait time is visible in traces; on budget exhaustion the
    original error is reraised after capturing live pool stats to Sentry.

    Retrying the same session is safe: no connection is bound until checkout
    succeeds, so each attempt re-attempts the checkout cleanly.
    """
    with sentry_sdk.start_span(op="db.pool.acquire", name=context):
        if not settings.DB.CONNECTION_RETRY_ENABLED:
            await db.connection()
            return
        try:
            async for attempt in AsyncRetrying(
                wait=wait_exponential_jitter(
                    initial=settings.DB.CONNECTION_RETRY_BACKOFF_INITIAL_SECONDS,
                    max=settings.DB.CONNECTION_RETRY_BACKOFF_MAX_SECONDS,
                ),
                stop=stop_after_delay(settings.DB.CONNECTION_RETRY_MAX_DELAY_SECONDS),
                retry=retry_if_exception_type(RETRYABLE_DB_CONNECTION_ERRORS),
                reraise=True,
            ):
                with attempt:
                    await db.connection()
        except RETRYABLE_DB_CONNECTION_ERRORS as e:
            if settings.SENTRY.ENABLED:
                sentry_sdk.set_context("db_pool", get_pool_stats())
                sentry_sdk.capture_exception(e)
            raise


# Define your naming convention
convention = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",  # Index - supports multi-column
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",  # Unique constraint - supports multi-column
    "ck": "ck_%(table_name)s_%(constraint_name)s",  # Check constraint
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",  # Foreign key - supports composite keys
    "pk": "pk_%(table_name)s",  # Primary key
}

table_schema = settings.DB.SCHEMA
# Note: column_0_N_name expands to include all columns in multi-column constraints
# e.g., "workspace_id_tenant_id" for a composite constraint on both columns
meta = MetaData(naming_convention=convention)
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
