import contextvars
import logging
from typing import Any

import sentry_sdk
from sqlalchemy import MetaData, event, text
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
from src.telemetry.prometheus.metrics import (
    db_queries_in_flight_gauge,
    prometheus_metrics,
)

logger = logging.getLogger(__name__)

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

# Identifies this process ("api" | "deriver") on DB metrics. Set once at startup
# by register_db_query_instrumentation; stays "unknown" if metrics are disabled.
_db_instance_type: str = "unknown"


def _record_acquisition_outcome(outcome: str) -> None:
    """Record a connection-acquisition outcome (no-op when metrics disabled)."""
    if settings.METRICS.ENABLED:
        prometheus_metrics.record_db_connection_acquisition(
            instance_type=_db_instance_type, outcome=outcome
        )


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

    Each attempt rolls the session back on a retryable failure before retrying:
    a failed checkout can leave the autobegun transaction in a pending-rollback
    state, which would make the next ``db.connection()`` raise instead of
    re-checking-out cleanly. The rollback is pure Python-side state cleanup when
    no connection was bound, so it is cheap and safe.
    """
    with sentry_sdk.start_span(op="db.pool.acquire", name=context):
        if not settings.DB.CONNECTION_RETRY_ENABLED:
            await db.connection()
            return
        attempts = 0
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
                    attempts += 1
                    try:
                        await db.connection()
                    except RETRYABLE_DB_CONNECTION_ERRORS:
                        # Reset session state so the next attempt starts clean.
                        try:
                            await db.rollback()
                        except Exception:
                            logger.debug(
                                "rollback after failed checkout failed",
                                exc_info=True,
                            )
                        raise
        except RETRYABLE_DB_CONNECTION_ERRORS as e:
            _record_acquisition_outcome("exhausted")
            if settings.SENTRY.ENABLED:
                sentry_sdk.set_context("db_pool", get_pool_stats())
                sentry_sdk.capture_exception(e)
            raise
        # "ok" on first try, "retried" if backoff was needed before success.
        _record_acquisition_outcome("ok" if attempts <= 1 else "retried")


class DBQueryInflightTracker:
    """Tracks statements executing on the wire via SQLAlchemy cursor events.

    Drift-proof: marks ``Connection.info`` when a statement starts and clears it
    on completion OR error, so the gauge can't leak upward (an errored statement
    skips ``after_cursor_execute``) or go negative (a connect-time error has no
    matching start). Bound to a pre-resolved labeled gauge child so the
    per-statement hot path does no label resolution.
    """

    # Marker on Connection.info recording that we incremented for the current
    # statement, so we decrement exactly once on completion or error.
    INFLIGHT_KEY: str = "_honcho_inflight"

    def __init__(self, gauge_child: Any) -> None:
        self._child: Any = gauge_child

    def on_before(self, conn: Any, *_: Any) -> None:
        try:
            conn.info[self.INFLIGHT_KEY] = True
            self._child.inc()
        except Exception:
            logger.debug("in-flight gauge inc failed", exc_info=True)

    def on_after(self, conn: Any, *_: Any) -> None:
        try:
            if conn.info.pop(self.INFLIGHT_KEY, False):
                self._child.dec()
        except Exception:
            logger.debug("in-flight gauge dec failed", exc_info=True)

    def on_error(self, exception_context: Any) -> None:
        try:
            conn = exception_context.connection
            if conn is not None and conn.info.pop(self.INFLIGHT_KEY, False):
                self._child.dec()
        except Exception:
            logger.debug("in-flight gauge error-path dec failed", exc_info=True)


# Process-wide tracker, created at registration (None until then / if metrics off).
_inflight_tracker: DBQueryInflightTracker | None = None


def register_db_query_instrumentation(instance_type: str) -> None:
    """Attach per-statement in-flight tracking to the engine (no-op if off).

    Gated on METRICS.ENABLED so there is zero overhead — not even attached event
    listeners — when metrics are disabled. Safe to call once per process.
    """
    global _db_instance_type, _inflight_tracker
    _db_instance_type = instance_type
    if not settings.METRICS.ENABLED:
        return
    child = db_queries_in_flight_gauge.labels(instance_type=instance_type)
    _inflight_tracker = DBQueryInflightTracker(child)
    sync_engine = engine.sync_engine
    event.listen(sync_engine, "before_cursor_execute", _inflight_tracker.on_before)
    event.listen(sync_engine, "after_cursor_execute", _inflight_tracker.on_after)
    event.listen(sync_engine, "handle_error", _inflight_tracker.on_error)


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
