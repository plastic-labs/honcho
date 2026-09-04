import uuid
from contextlib import asynccontextmanager

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db import (
    ReadSessionLocal,
    ServiceReadSessionLocal,
    ServiceSessionLocal,
    SessionLocal,
    request_context,
    tenant_context,
)


async def get_db():
    """FastAPI Dependency Generator for Database.

    The session is lazy: it does NOT check out a pooled connection here. The
    AsyncSession checks one out on the first DB-touching call, so a handler doing
    non-DB work (embedding/file/LLM) before its first query does not pin a
    connection across it.
    """
    db: AsyncSession = SessionLocal()
    try:
        yield db
    except Exception:
        await db.rollback()
        raise
    finally:
        # Always send ROLLBACK unconditionally so the wire-level transaction
        # is closed before the TCP connection drops.  Supavisor v2 does NOT
        # clean up orphaned transactions on client disconnect in transaction-
        # pooling mode, so relying on `in_transaction()` (Python-side state)
        # can leave the backend pinned with an open BEGIN. (Cheap no-op if the
        # lazy session never checked out a connection.)
        await db.rollback()
        await db.close()


async def get_read_db():
    """FastAPI Dependency Generator for SELECT-only handlers.

    Same lazy-checkout semantics as get_db, but the session is bound to the
    AUTOCOMMIT read engine: no BEGIN is ever emitted, so the connection can not
    sit 'idle in transaction' between the query and this teardown — a delayed
    finally here is harmless (the backend is plain 'idle'). close() is still
    required to release the connection itself back to the pool.

    MUST only be used by handlers that never mutate; see ReadSessionLocal.
    """
    db: AsyncSession = ReadSessionLocal()
    try:
        yield db
    finally:
        # rollback is a wire-level no-op under AUTOCOMMIT; kept to reset any
        # Python-side session state before close, mirroring get_db.
        await db.rollback()
        await db.close()


@asynccontextmanager
async def tracked_db(
    operation_name: str | None = None,
    *,
    read_only: bool = False,
    tenant_id: str | None = None,
):
    """Context manager for tracked database sessions.

    Sets a task-scoped request_context so the lazy session picks it up for
    tracing/attribution, then yields a lazy session (see get_db).

    Pass read_only=True for SELECT-only windows: the session is then bound to
    the AUTOCOMMIT read engine, so the work inside the block never holds an
    open transaction (no idle-in-transaction parking; the pooler can reclaim
    the backend between statements). Never use read_only=True on a path that
    mutates — see ReadSessionLocal.

    tenant_id binds this session to a tenant: it is set into tenant_context for
    the duration so the connection-checkout hook applies the `app.tenant` GUC (see
    src/db.py) and the RLS policies resolve. A tenant may instead be inherited from
    an ambient tenant_context already set by an outer scope (today, the deriver
    binding a claimed work unit's tenant; a per-request API binding is not yet
    wired), so nested sessions need not re-pass it. When MULTI_TENANT is enabled a tenant is REQUIRED
    from one of those two sources — if neither is present we raise here, before any
    query executes, so a tenant-scoped session can never run unbound (fail-closed).
    Legitimately cross-tenant work must use service_db(), not a null tenant here.
    """
    if settings.MULTI_TENANT and not tenant_id and not tenant_context.get():
        raise ValueError(
            "tracked_db requires a tenant when MULTI_TENANT is enabled "
            + "(pass tenant_id or set tenant_context); cross-tenant paths must use "
            + "service_db()"
        )

    # Get request ID if available, or create operation-specific one
    context = request_context.get()
    token = None

    if not context and operation_name:
        context = f"task:{operation_name}:{str(uuid.uuid4())[:8]}"
        token = request_context.set(context)

    tenant_token = None
    if tenant_id is not None:
        tenant_token = tenant_context.set(tenant_id)

    db = (ReadSessionLocal if read_only else SessionLocal)()
    try:
        yield db
    except Exception:
        await db.rollback()
        raise
    finally:
        # Always send ROLLBACK unconditionally — see get_db() comment. (Under
        # read_only/AUTOCOMMIT it is a wire-level no-op.)
        await db.rollback()
        await db.close()
        if token:  # Only reset if we set it
            request_context.reset(token)
        if tenant_token:  # Only reset if we set it
            tenant_context.reset(tenant_token)


@asynccontextmanager
async def service_db(operation_name: str | None = None, *, read_only: bool = False):
    """RLS-bypassing session for the legitimately cross-tenant service paths.

    For work that spans all tenants by design — the deriver's queue claim, the
    reconciler's vector scans, the dreamer's scheduling, enqueue — which read and
    write across tenants. Bound to the service engine, which in the cloud deploy
    connects as a role that BYPASSES row-level security. This is deliberate: a
    session that merely omits `app.tenant` would hit the fail-closed policy and see
    ZERO rows, silently processing nothing. No tenant is bound here.

    ⚠️ Claim here, work per-tenant: a work unit claimed cross-tenant must be
    *processed* through tracked_db(tenant_id=...) — the unit carries its tenant —
    so its writes are RLS-checked (WITH CHECK). Do not do per-tenant writes on
    this session.

    Pass read_only=True for SELECT-only cross-tenant windows (AUTOCOMMIT; see
    tracked_db). When DB.SERVICE_CONNECTION_URI is unset this is the ordinary
    engine (single-role — correct when MULTI_TENANT is off).
    """
    context = request_context.get()
    token = None

    if not context and operation_name:
        context = f"task:{operation_name}:{str(uuid.uuid4())[:8]}"
        token = request_context.set(context)

    db = (ServiceReadSessionLocal if read_only else ServiceSessionLocal)()
    try:
        yield db
    except Exception:
        await db.rollback()
        raise
    finally:
        # Always send ROLLBACK unconditionally — see get_db() comment. (Under
        # read_only/AUTOCOMMIT it is a wire-level no-op.)
        await db.rollback()
        await db.close()
        if token:  # Only reset if we set it
            request_context.reset(token)


db: AsyncSession = Depends(get_db)
read_db: AsyncSession = Depends(get_read_db)
