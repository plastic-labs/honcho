import contextvars
import logging
from typing import Any

from sqlalchemy import MetaData, event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool

from src.config import settings
from src.telemetry.prometheus.metrics import db_queries_in_flight_gauge

logger = logging.getLogger(__name__)

connect_args = {
    "prepare_threshold": None,
    # Bound a single connection attempt so it fails fast instead of hanging when
    # the server/pooler is unreachable or stalled (psycopg, seconds).
    "connect_timeout": settings.DB.CONNECT_TIMEOUT_SECONDS,
}

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

# A vanilla AsyncSession is lazy: it checks out a pooled connection on the first
# DB-touching call (not at construction) and couples the checkout to the
# statement, so a handler doing non-DB work (embedding/file/LLM) before its
# first query does not pin a connection across it. Connection acquisition is a
# single attempt with no retry — callers handle a saturated/unreachable DB (the
# API surfaces the error; the deriver backs off and retries on a later poll).
SessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
    class_=AsyncSession,
)

# Read-only engine: shares `engine`'s pool, but checks connections out in DBAPI
# AUTOCOMMIT mode, so psycopg emits NO BEGIN — a SELECT never autobegins a
# transaction. The backend therefore returns to state 'idle' (not 'idle in
# transaction') the moment a statement completes.
read_engine = engine.execution_options(isolation_level="AUTOCOMMIT")

# Sessions for SELECT-only work (same lazy-checkout semantics as SessionLocal).
# MUST NOT be used for writes: with no enclosing transaction, begin_nested()
# savepoints (see the crud get-or-create paths) break, and every flush would
# commit immediately. Use SessionLocal for anything that mutates.
ReadSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=read_engine,
    class_=AsyncSession,
)


def _set_application_name_on_checkout(
    dbapi_connection: Any, _connection_record: Any, _connection_proxy: Any
) -> None:
    """Tag each checked-out connection with the current request context.

    Registered only when ``DB.TRACING`` is on. Fires on every pool checkout (so a
    reused pooled connection is re-tagged for the new caller), reading the
    per-task ``request_context`` the request/task scope has already set.
    Best-effort: a failure here must never break the checkout.

    Runs in autocommit so it never leaves the connection 'idle in transaction'
    at checkout: this hook fires BEFORE the dialect applies execution-option
    isolation levels, and psycopg refuses to switch a connection into AUTOCOMMIT
    (which the read engine does) while a transaction opened by this statement is
    still in progress. set_config(..., is_local=false) is session-scoped, so it
    persists past the autocommit boundary.
    """
    context = request_context.get() or "unknown"
    try:
        previous_autocommit = dbapi_connection.autocommit
        if not previous_autocommit:
            dbapi_connection.autocommit = True
        try:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute(
                    "SELECT set_config('application_name', %s, false)", (context,)
                )
            finally:
                cursor.close()
        finally:
            if not previous_autocommit:
                dbapi_connection.autocommit = False
    except Exception:
        logger.debug("setting application_name on checkout failed", exc_info=True)


if settings.DB.TRACING:
    event.listen(engine.sync_engine, "checkout", _set_application_name_on_checkout)


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
        # overflow() is negative until the base pool fills (it starts at
        # -pool_size); clamp to the count of overflow connections actually open.
        return {
            "checked_out": pool.checkedout(),
            "checked_in": pool.checkedin(),
            "size": pool.size(),
            "overflow": max(0, pool.overflow()),
        }
    except Exception:
        return zeros


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


_db_query_instrumentation_registered = False


def register_db_query_instrumentation(instance_type: str) -> None:
    """Attach per-statement in-flight tracking to the engine (no-op if off).

    Gated on METRICS.ENABLED so there is zero overhead — not even attached event
    listeners — when metrics are disabled. Idempotent: repeated calls (e.g. a
    re-run lifespan or test startup) won't attach duplicate listeners, which
    would double-count in-flight statements.
    """
    global _inflight_tracker, _db_query_instrumentation_registered
    if not settings.METRICS.ENABLED or _db_query_instrumentation_registered:
        return
    child = db_queries_in_flight_gauge.labels(instance_type=instance_type)
    _inflight_tracker = DBQueryInflightTracker(child)
    sync_engine = engine.sync_engine
    event.listen(sync_engine, "before_cursor_execute", _inflight_tracker.on_before)
    event.listen(sync_engine, "after_cursor_execute", _inflight_tracker.on_after)
    event.listen(sync_engine, "handle_error", _inflight_tracker.on_error)
    _db_query_instrumentation_registered = True


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
