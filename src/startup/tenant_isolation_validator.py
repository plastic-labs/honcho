"""Startup validator for the multi-tenant isolation binding.

Gates boot (API and deriver) when ``MULTI_TENANT`` is on, converting two silent
half-states — where isolation looks enabled but isn't — into a hard boot failure:

1. Pooler vs read-path strategy. The read-path binding is a session-scoped
   ``app.tenant`` set at checkout; it is safe under NullPool / session-mode, but a
   transaction/statement-mode pooler multiplexes backends below the SQLAlchemy
   session, so tenant A's GUC can be read by tenant B's next transaction — a
   cross-tenant read that fails OPEN (the dangerous direction). Refuse to boot in
   that combination.

2. Flag vs policies. ``MULTI_TENANT`` on but the data tables lack RLS
   enabled + forced means the binding is set but nothing enforces it — no
   isolation, no error. Refuse to boot unless ``MULTI_TENANT_SKIP_RLS_ASSERT`` is
   set (migration window only).

No-op when ``MULTI_TENANT`` is off: self-host runs on plain, RLS-free Postgres.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from src.config import AppSettings, settings
from src.startup.embedding_validator import StartupValidationError

logger = logging.getLogger(__name__)

# The tenant-scoped data tables that MUST carry RLS (enabled + forced) when
# MULTI_TENANT is on. The service tables (queue, active_queue_sessions) are
# deliberately excluded — they hold no tenant data behind RLS.
_RLS_REQUIRED_TABLES: tuple[str, ...] = (
    "workspaces",
    "peers",
    "sessions",
    "messages",
    "message_embeddings",
    "collections",
    "documents",
    "session_peers",
    "webhook_endpoints",
)

# Pooler modes that multiplex backends below the SQLAlchemy session, which the
# session-scoped read-path binding cannot survive.
_UNSAFE_POOLER_MODES: frozenset[str] = frozenset({"transaction", "statement"})

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0


async def validate_tenant_isolation(
    engine: AsyncEngine,
    *,
    app_settings: AppSettings | None = None,
) -> None:
    """Fail boot on a multi-tenant half-state. No-op unless MULTI_TENANT is on.

    Run after the DB pool is initialized and before serving traffic / processing
    the queue — the same placement as ``validate_embedding_schema``.
    """
    s = app_settings if app_settings is not None else settings
    if not s.MULTI_TENANT:
        return

    _assert_pooler_mode_safe(s.DB.POOLER_MODE)

    if s.MULTI_TENANT_SKIP_RLS_ASSERT:
        logger.warning(
            "MULTI_TENANT_SKIP_RLS_ASSERT is set: skipping the RLS-enforced"
            + " assertion. Intended for the migration window only — the tenant"
            + " isolation guarantee is NOT verified while this is on."
        )
        return

    rls = await _introspect_rls_with_retry(engine, s.DB.SCHEMA)
    _assert_rls_enforced(rls, schema=s.DB.SCHEMA)


def _assert_pooler_mode_safe(pooler_mode: str) -> None:
    if pooler_mode in _UNSAFE_POOLER_MODES:
        raise StartupValidationError(
            f"DB_POOLER_MODE={pooler_mode!r} is incompatible with MULTI_TENANT:"
            + " the session-scoped tenant binding leaks across tenants under a"
            + " transaction/statement-mode pooler. Run the shared deploy on"
            + " NullPool or a session-mode pooler (DB_POOLER_MODE is hand-set,"
            + " not probed)."
        )


async def _introspect_rls_with_retry(
    engine: AsyncEngine, schema: str
) -> dict[str, tuple[bool, bool]]:
    """Return {table -> (relrowsecurity, relforcerowsecurity)} for the data
    tables. Fails closed on the last attempt — uncertainty is not a green light."""
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_fixed(_RETRY_BACKOFF_SECONDS),
            retry=retry_if_exception_type(SQLAlchemyError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        ):
            with attempt:
                return await _introspect_rls_once(engine, schema)
    except RetryError as e:
        underlying = e.last_attempt.exception()
        raise StartupValidationError(
            f"could not validate tenant-isolation RLS: {underlying}"
        ) from underlying
    # Unreachable: AsyncRetrying either returns from inside the loop or raises.
    raise StartupValidationError("tenant-isolation RLS introspection did not run")


async def _introspect_rls_once(
    engine: AsyncEngine, schema: str
) -> dict[str, tuple[bool, bool]]:
    """Schema-qualified pg_class read of the RLS flags for the data tables.

    On the shared (partitioned) schema these are the partitioned parents; ENABLE
    /FORCE ROW LEVEL SECURITY on a parent cascades to its partitions, and the
    parent's pg_class row carries the flags — so reading the parent is correct.
    """
    query = text(
        """
        SELECT c.relname AS table_name,
               c.relrowsecurity AS row_security,
               c.relforcerowsecurity AS force_row_security
        FROM pg_class c
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = :schema
          AND c.relname = ANY(:tables)
        """
    )
    async with engine.connect() as conn:
        result = await conn.execute(
            query,
            {"schema": schema, "tables": list(_RLS_REQUIRED_TABLES)},
        )
        return {
            row.table_name: (row.row_security, row.force_row_security) for row in result
        }


def _assert_rls_enforced(rls: dict[str, tuple[bool, bool]], *, schema: str) -> None:
    expected = set(_RLS_REQUIRED_TABLES)
    missing = expected - rls.keys()
    if missing:
        listing = ", ".join(sorted(f"{schema}.{t}" for t in missing))
        raise StartupValidationError(
            f"MULTI_TENANT is on but required tables are absent: {listing}."
            + " Run `alembic upgrade head` first."
        )
    unenforced: list[str] = []
    for table in sorted(expected):
        row_security, force_row_security = rls[table]
        if not row_security or not force_row_security:
            unenforced.append(
                f"{schema}.{table}"
                + f" (rowsecurity={row_security}, force={force_row_security})"
            )
    if unenforced:
        raise StartupValidationError(
            "MULTI_TENANT is on but RLS is not enabled+forced on: "
            + ", ".join(unenforced)
            + ". Apply the tenant-isolation policies (ENABLE + FORCE ROW LEVEL"
            + " SECURITY) before enabling the flag, or set"
            + " MULTI_TENANT_SKIP_RLS_ASSERT for the migration window."
        )
