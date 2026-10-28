"""Startup validator for the multi-tenant isolation binding.

Gates boot (API and deriver) when ``MULTI_TENANT`` is on, converting five silent
half-states — where isolation looks enabled but cannot hold, or cannot serve — into
a hard boot failure. Listed in check order: the first three are settings-only and need
no connection; (4) and (5) introspect the database (with retry).

1. Flag vs auth (API processes only). ``MULTI_TENANT`` on with
   ``AUTH_USE_AUTH`` off means every request authenticates as a tenant-less admin,
   so no tenant is ever bound and every tenant-scoped session fails closed — not a
   leak, but a uniform outage on every data route that is hard to read from the
   500s. Refuse to boot with one clear error instead. A deriver takes its tenant
   from the claimed work unit's key, not a JWT, so a deriver process skips this
   check. Each entrypoint names its own process type when it calls the validator.

2. Pooler vs read-path strategy. The read-path binding is a session-scoped
   ``app.tenant`` set at checkout; it is safe under NullPool / session-mode, but a
   transaction/statement-mode pooler multiplexes backends below the SQLAlchemy
   session, so tenant A's GUC can be read by tenant B's next transaction — a
   cross-tenant read that fails OPEN (the dangerous direction). Refuse to boot in
   that combination.

3. Flag vs service role. The cross-tenant service paths (deriver claim,
   reconciler, dreamer) must run on a role that bypasses RLS; with
   ``DB_SERVICE_CONNECTION_URI`` unset they would run on the RLS-enforced app role
   and silently read zero rows.

4. Flag vs policies. ``MULTI_TENANT`` on but the data tables lack RLS
   enabled + forced means the binding is set but nothing enforces it — no
   isolation, no error.

5. Role vs policies. Postgres does not enforce row-level security for a
   superuser or a role with BYPASSRLS, so on such a role every check in (4)
   passes vacuously while isolation is actually off. Refuse to boot if the
   TENANT engine's connecting role (``DB_CONNECTION_URI``) is either.

No-op when ``MULTI_TENANT`` is off: self-host runs on plain, RLS-free Postgres.

This is a permanent isolation guard, not a migration-scoped check: it stays in
OSS so a self-hoster running one instance across multiple tenants gets the same
fail-closed guarantee, and it defends against a later misconfiguration turning the
flag on without RLS in place.
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

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
from src.models import MESSAGE_TEXT_MATCHES
from src.startup.embedding_validator import StartupValidationError

logger = logging.getLogger(__name__)

# The tenant-scoped data tables that MUST carry RLS (enabled + forced) when
# MULTI_TENANT is on.
# region ai
# The service tables (queue, active_queue_sessions) are deliberately excluded —
# they hold no tenant data behind RLS.
# endregion
_RLS_REQUIRED_TABLES: tuple[str, ...] = (
    "workspaces",
    "peers",
    "sessions",
    "messages",
    "message_embeddings",
    "collections",
    "documents",
    "document_sources",
    "session_peers",
    "webhook_endpoints",
)

# region ai
# Pooler modes that multiplex backends below the SQLAlchemy session, which the
# session-scoped read-path binding cannot survive.
# endregion
_UNSAFE_POOLER_MODES: frozenset[str] = frozenset({"transaction", "statement"})

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0


async def validate_tenant_isolation(
    engine: AsyncEngine,
    *,
    instance_type: Literal["api", "deriver"],
    app_settings: AppSettings | None = None,
) -> None:
    """Fail boot on a multi-tenant half-state. No-op unless MULTI_TENANT is on.

    Run after the DB pool is initialized and before serving traffic / processing
    the queue — the same placement as ``validate_embedding_schema``. The caller
    passes its own ``instance_type`` (the API lifespan ``"api"``, the deriver
    entrypoint ``"deriver"``), the same way both label their metrics.
    """
    s = app_settings if app_settings is not None else settings
    if not s.MULTI_TENANT:
        return

    # Every settings-only check runs first: a misconfiguration that needs no
    # database should be reported immediately, not after the introspection retries
    # have spent themselves failing to reach a database that may also be down.
    _assert_auth_enabled(s, instance_type)
    _assert_pooler_mode_safe(s.DB.POOLER_MODE)
    _assert_service_role_configured(s)

    rls = await _introspect_rls_with_retry(engine, s.DB.SCHEMA)
    _assert_rls_enforced(rls, schema=s.DB.SCHEMA)
    rolname, rolsuper, rolbypassrls = await _introspect_tenant_role_with_retry(engine)
    _assert_tenant_role_cannot_bypass_rls(rolname, rolsuper, rolbypassrls)
    search_function = await _introspect_search_function_with_retry(engine, s.DB.SCHEMA)
    _assert_search_function_bypasses_rls(search_function, schema=s.DB.SCHEMA)


def _assert_service_role_configured(s: AppSettings) -> None:
    """Require a distinct service connection for the cross-tenant paths."""
    # region ai
    # The cross-tenant service paths (deriver claim, reconciler, dreamer, enqueue)
    # read tenant data across tenants and must run on a role that BYPASSES RLS.
    # With DB_SERVICE_CONNECTION_URI unset, service_db falls back to the
    # RLS-enforced app role, so those reads match zero rows and the deriver
    # silently processes nothing.
    # endregion
    if not s.DB.SERVICE_CONNECTION_URI:
        raise StartupValidationError(
            "MULTI_TENANT is on with RLS enforced but DB_SERVICE_CONNECTION_URI is"
            + " unset: the cross-tenant service paths would run on the RLS-enforced"
            + " app role and read zero rows. Configure a service connection whose"
            + " role bypasses RLS."
        )


def _assert_auth_enabled(
    s: AppSettings, instance_type: Literal["api", "deriver"]
) -> None:
    """API instances: require JWT auth under the flag; the claim is the tenant source."""
    # region ai
    # With AUTH_USE_AUTH off, auth() short-circuits every request to a tenant-less
    # admin JWTParams and never reaches the tenant gate, so nothing binds
    # tenant_context and the fail-closed tracked_db guard 500s every data route.
    # No cross-tenant read is possible in that state — operability, not a security
    # gap — but one boot error beats a storm of identical runtime errors. The
    # deriver binds its tenant from the claimed work unit's key and verifies no
    # JWT, so it is exempt rather than forced to carry the API's auth config.
    # endregion
    if instance_type != "api":
        return
    if not s.AUTH.USE_AUTH:
        raise StartupValidationError(
            "MULTI_TENANT is on but AUTH_USE_AUTH is off: with auth disabled no"
            + " request carries a tenant claim, so no tenant is ever bound and every"
            + " tenant-scoped request fails closed. Enable AUTH_USE_AUTH (with"
            + " AUTH_JWT_SECRET) and issue tenant-bearing tokens."
        )


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
    """Return {table -> (relrowsecurity, relforcerowsecurity)} for the data tables."""
    # ai: fails closed on the last attempt — uncertainty is not a green light.
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
    # ai: unreachable — AsyncRetrying either returns from inside the loop or raises.
    raise StartupValidationError("tenant-isolation RLS introspection did not run")


async def _introspect_rls_once(
    engine: AsyncEngine, schema: str
) -> dict[str, tuple[bool, bool]]:
    """Schema-qualified pg_class read of the RLS flags for the data tables."""
    # region ai
    # On the shared (partitioned) schema these are the partitioned parents;
    # ENABLE/FORCE ROW LEVEL SECURITY on a parent cascades to its partitions, and
    # the parent's pg_class row carries the flags — so reading the parent is correct.
    # endregion
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
            + " SECURITY) before enabling the flag."
        )


async def _introspect_tenant_role_with_retry(
    engine: AsyncEngine,
) -> tuple[str, bool, bool]:
    """Return (rolname, rolsuper, rolbypassrls) for the TENANT engine's connecting role."""
    # ai: fails closed on the last attempt — uncertainty is not a green light.
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_fixed(_RETRY_BACKOFF_SECONDS),
            retry=retry_if_exception_type(SQLAlchemyError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        ):
            with attempt:
                return await _introspect_tenant_role_once(engine)
    except RetryError as e:
        underlying = e.last_attempt.exception()
        raise StartupValidationError(
            f"could not validate the tenant engine's connecting role: {underlying}"
        ) from underlying
    # ai: unreachable — AsyncRetrying either returns from inside the loop or raises.
    raise StartupValidationError("tenant engine role introspection did not run")


async def _introspect_tenant_role_once(engine: AsyncEngine) -> tuple[str, bool, bool]:
    """Read whether the role the TENANT engine connects as can bypass RLS.

    ``current_user`` (not ``session_user``) so this reflects the role RLS is
    actually evaluated against; the app never issues ``SET ROLE``, so the two
    coincide for every connection this engine hands out. Deliberately run on
    ``engine`` (the TENANT engine — the one ``tracked_db``/request/deriver
    sessions bind ``app.tenant`` on and RLS policies are meant to constrain),
    never on ``service_engine``: the service role is REQUIRED to bypass RLS
    (see ``_assert_service_role_configured``) and must not trip this check.
    """
    query = text(
        """
        SELECT rolname, rolsuper, rolbypassrls
        FROM pg_roles
        WHERE rolname = current_user
        """
    )
    async with engine.connect() as conn:
        row = (await conn.execute(query)).one()
        return row.rolname, row.rolsuper, row.rolbypassrls


def _assert_tenant_role_cannot_bypass_rls(
    rolname: str, rolsuper: bool, rolbypassrls: bool
) -> None:
    # region ai
    # Postgres bypasses row-level security entirely for a superuser or any role
    # with BYPASSRLS, regardless of relrowsecurity/relforcerowsecurity on the
    # tables (the checks _assert_rls_enforced just ran) — so on such a role
    # every one of those checks passes vacuously while isolation is actually
    # off. Do not special-case or soften this: there is no "skip if superuser"
    # escape hatch.
    # endregion
    reasons: list[str] = []
    if rolsuper:
        reasons.append("is a superuser")
    if rolbypassrls:
        reasons.append("has BYPASSRLS")
    if reasons:
        raise StartupValidationError(
            "MULTI_TENANT is on but the TENANT engine's connecting role"
            + f" {rolname!r} (DB_CONNECTION_URI) "
            + " and ".join(reasons)
            + ": Postgres does not enforce row-level security for a superuser or"
            + " a BYPASSRLS role, so the RLS checks above pass vacuously while"
            + " isolation is actually off for every query this role runs. Point"
            + " DB_CONNECTION_URI at an ordinary role with RLS enforced;"
            + " superuser/BYPASSRLS is reserved for the cross-tenant service role"
            + " (DB_SERVICE_CONNECTION_URI)."
        )


class SearchFunction(NamedTuple):
    """What the boot check reads about ``message_text_matches``."""

    security_definer: bool
    owner: str
    owner_is_superuser: bool
    owner_bypasses_rls: bool


async def _introspect_search_function_with_retry(
    engine: AsyncEngine, schema: str
) -> SearchFunction | None:
    """Return the search function's definer flag and owner, or None if it is absent."""
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_fixed(_RETRY_BACKOFF_SECONDS),
            retry=retry_if_exception_type(SQLAlchemyError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        ):
            with attempt:
                return await _introspect_search_function_once(engine, schema)
    except RetryError as e:
        underlying = e.last_attempt.exception()
        raise StartupValidationError(
            f"could not validate the message search function: {underlying}"
        ) from underlying
    raise StartupValidationError("message search function introspection did not run")


async def _introspect_search_function_once(
    engine: AsyncEngine, schema: str
) -> SearchFunction | None:
    query = text(
        """
        SELECT p.prosecdef, r.rolname, r.rolsuper, r.rolbypassrls
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        JOIN pg_roles r ON r.oid = p.proowner
        WHERE n.nspname = :schema AND p.proname = :name
        """
    )
    async with engine.connect() as conn:
        row = (
            await conn.execute(query, {"schema": schema, "name": MESSAGE_TEXT_MATCHES})
        ).first()
    if row is None:
        return None
    return SearchFunction(row.prosecdef, row.rolname, row.rolsuper, row.rolbypassrls)


def _assert_search_function_bypasses_rls(
    found: SearchFunction | None, *, schema: str
) -> None:
    name = f"{schema}.{MESSAGE_TEXT_MATCHES}"
    if found is None:
        raise StartupValidationError(
            f"MULTI_TENANT is on but the message search function {name} is absent."
            + " Run `alembic upgrade head` first."
        )
    if not found.security_definer:
        raise StartupValidationError(
            f"MULTI_TENANT is on but {name} is not SECURITY DEFINER, so message"
            + " search runs under RLS and cannot use the content indexes."
            + " Recreate it with `alembic upgrade head`."
        )
    if not (found.owner_is_superuser or found.owner_bypasses_rls):
        raise StartupValidationError(
            f"MULTI_TENANT is on but {name} is owned by {found.owner!r}, which"
            + " cannot bypass RLS: the function runs under the policy like any"
            + " other statement and message search cannot use the content"
            + " indexes. Run migrations as a BYPASSRLS role, or"
            + f" `ALTER FUNCTION {name}(text, text, text, text, boolean)"
            + " OWNER TO <a BYPASSRLS role>`."
        )
