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
from src.telemetry.prometheus.metrics import (
    db_connections_established_counter,
    db_connections_open_gauge,
    db_queries_in_flight_gauge,
)

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

# Context variable holding the current request's tenant. Read at connection
# checkout to bind the session-scoped `app.tenant` GUC when MULTI_TENANT is on.
# Set at the request/auth boundary or explicitly via tracked_db(tenant_id=...);
# None outside any tenant scope.
tenant_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tenant_context", default=None
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


def _set_session_gucs_on_checkout(
    dbapi_connection: Any, _connection_record: Any, _connection_proxy: Any
) -> None:
    """Set session-scoped GUCs on each checked-out connection.

    Registered when ``DB.TRACING`` or ``MULTI_TENANT`` is on; fires on every pool
    checkout (so a reused pooled connection is re-tagged for the new caller). Two
    independent GUCs, each gated on its own setting, applied in one autocommit pass:

    - ``application_name`` (when ``DB.TRACING``): the per-task ``request_context``,
      for observability. Best-effort — a failure only mislabels a backend.
    - ``app.tenant`` (when ``MULTI_TENANT``): the request's tenant, so row-level
      security policies resolve. This is the ONLY place it can be bound on the
      AUTOCOMMIT read path, which has no transaction for a ``SET LOCAL`` to attach
      to, so it is set session-scoped (``is_local=false``) and persists across the
      whole checkout. Set on EVERY checkout — to the current tenant, or to '' when
      none is in scope — so a reused connection never carries a prior tenant's
      value; an empty/unset tenant makes the policy match zero rows (fails CLOSED,
      never cross-tenant). Session-scoped safety assumes the connection is not
      multiplexed across transactions below the session (no transaction-mode
      pooler); the startup checks enforce that.

    Runs in autocommit so it never leaves the connection 'idle in transaction' at
    checkout: this hook fires BEFORE the dialect applies execution-option isolation
    levels, and psycopg refuses to switch a connection into AUTOCOMMIT (which the
    read engine does) while a transaction opened here is still in progress.
    """
    set_tracing = settings.DB.TRACING
    set_tenant = settings.MULTI_TENANT
    if not set_tracing and not set_tenant:
        return
    try:
        previous_autocommit = dbapi_connection.autocommit
        if not previous_autocommit:
            dbapi_connection.autocommit = True
        try:
            cursor = dbapi_connection.cursor()
            try:
                if set_tracing:
                    context = request_context.get() or "unknown"
                    cursor.execute(
                        "SELECT set_config('application_name', %s, false)", (context,)
                    )
                if set_tenant:
                    tenant = tenant_context.get() or ""
                    cursor.execute(
                        "SELECT set_config('app.tenant', %s, false)", (tenant,)
                    )
            finally:
                cursor.close()
        finally:
            if not previous_autocommit:
                dbapi_connection.autocommit = False
    except Exception:
        logger.debug("setting session GUCs on checkout failed", exc_info=True)


if settings.DB.TRACING or settings.MULTI_TENANT:
    event.listen(engine.sync_engine, "checkout", _set_session_gucs_on_checkout)


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


class DBConnectionTracker:
    """Tracks physical DB connections open on this engine via pool lifecycle events.

    Drift-proof, mirroring ``DBQueryInflightTracker``: marks the ``ConnectionRecord``
    on ``connect`` and decrements only if that mark is still present on
    ``close``/``invalidate``, so each physical connection increments the gauge
    exactly once and decrements at most once — it can't leak upward or go negative
    when both events fire during invalidation cleanup. Works for every pool class,
    including ``NullPool`` (whose pool keeps no records, so the scrape-time
    ``db_pool_connections`` collector reads zero).
    """

    # Marker on ConnectionRecord.info recording that we incremented for this
    # connection, so we decrement exactly once across close/invalidate.
    OPEN_KEY: str = "_honcho_conn_open"

    def __init__(self, open_child: Any, established_child: Any) -> None:
        self._open: Any = open_child
        self._established: Any = established_child

    def on_connect(self, _dbapi_connection: Any, connection_record: Any) -> None:
        try:
            connection_record.info[self.OPEN_KEY] = True
            self._established.inc()
            self._open.inc()
        except Exception:
            logger.debug("db-connection gauge inc failed", exc_info=True)

    def on_close(self, _dbapi_connection: Any, connection_record: Any, *_: Any) -> None:
        try:
            if connection_record is not None and connection_record.info.pop(
                self.OPEN_KEY, False
            ):
                self._open.dec()
        except Exception:
            logger.debug("db-connection gauge dec failed", exc_info=True)


# Process-wide tracker, created at registration (None until then / if metrics off).
_connection_tracker: DBConnectionTracker | None = None


_db_connection_instrumentation_registered = False


def register_db_connection_instrumentation(instance_type: str) -> None:
    """Attach physical-connection tracking to the engine (no-op if metrics off).

    Counts connections via pool lifecycle events, so it reports real numbers under
    any pool class — unlike the pool-object collector, which reads zero under
    ``NullPool``. Pre-resolving the labeled children materializes both series at 0,
    so an absent series signals a broken scrape rather than "no connections" (the
    zero-init convention). Idempotent: repeated calls won't attach duplicate
    listeners, which would double-count connections.
    """
    global _connection_tracker, _db_connection_instrumentation_registered
    if not settings.METRICS.ENABLED or _db_connection_instrumentation_registered:
        return
    open_child = db_connections_open_gauge.labels(instance_type=instance_type)
    established_child = db_connections_established_counter.labels(
        instance_type=instance_type
    )
    _connection_tracker = DBConnectionTracker(open_child, established_child)
    sync_engine = engine.sync_engine
    event.listen(sync_engine, "connect", _connection_tracker.on_connect)
    # A connection is torn down by close (normal return / recycle discard),
    # invalidate (broken connection), or detach — the last fires on GC-cleanup of an
    # abandoned async connection, where NullPool's close is a no-op so `close` never
    # fires. All three carry the ConnectionRecord; the marker dedupes if more than
    # one fires for the same connection.
    for teardown_event in ("close", "invalidate", "detach"):
        event.listen(sync_engine, teardown_event, _connection_tracker.on_close)
    _db_connection_instrumentation_registered = True


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
