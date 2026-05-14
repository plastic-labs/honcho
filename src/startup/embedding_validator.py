"""Startup validator for the embedding pipeline.

Crashes the process at boot if the configured EMBEDDING_VECTOR_DIMENSIONS does
not match the physical pgvector schema. Replaces an earlier config-time guard
that forbade non-1536 dims unless the operator asserted a VECTOR_STORE.MIGRATED
flag — the schema introspection here is more accurate because it inspects
actual state instead of operator-asserted state.

For external stores (turbopuffer, lancedb) the check is best-effort: namespaces
are per-workspace and lazy-created, so this validator can only sample existing
ones. Full enumeration is available via `uv run python scripts/configure_embeddings.py --report`.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, text
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
from src.exceptions import HonchoException
from src.models import Collection, Workspace
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Embedding tables that must exist with matching dim.
_EMBEDDING_TABLES: tuple[str, ...] = ("documents", "message_embeddings")

# Retry budget for transient introspection failures. Total wall time is
# bounded so a sick DB does not hang readiness; fail-closed after exhaustion.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0

# Best-effort external sampler bounds.
_EXTERNAL_SAMPLE_LIMIT = 10


class StartupValidationError(HonchoException):
    """Raised when the embedding configuration cannot be reconciled with the
    physical schema. Always surfaced before any HTTP route is served or any
    queue task is processed.

    Inherits from ``HonchoException`` (status_code=500) so the project's
    exception handlers recognize it consistently. Startup-time failure, not
    a per-request validation error — ``ValidationException``'s 422 semantics
    would be misleading.
    """


async def validate_embedding_schema(
    engine: AsyncEngine,
    *,
    app_settings: AppSettings | None = None,
) -> None:
    """Validate that the embedding schema matches the configured dimension.

    Run after the DB pool is initialized and before the embedding client is
    constructed. Fails closed: any unrecoverable introspection error raises
    rather than letting the process serve traffic with an unknown state.
    """
    s = app_settings if app_settings is not None else settings
    target_dim = s.EMBEDDING.VECTOR_DIMENSIONS
    schema = s.DB.SCHEMA

    dims = await _introspect_pgvector_dims_with_retry(engine, schema)
    _assert_pgvector_dims_match(dims, schema=schema, target_dim=target_dim)

    if s.VECTOR_STORE.TYPE in ("turbopuffer", "lancedb"):
        await _sample_external_namespaces(engine, target_dim=target_dim)


async def _introspect_pgvector_dims_with_retry(
    engine: AsyncEngine, schema: str
) -> dict[str, int]:
    """Schema-qualified pg_attribute introspection with bounded retries.

    Returns a mapping of table name -> raw ``atttypmod`` for the embedding
    columns. Fails closed on the last attempt — uncertainty is not a green
    light to serve traffic.
    """
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_fixed(_RETRY_BACKOFF_SECONDS),
            retry=retry_if_exception_type(SQLAlchemyError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        ):
            with attempt:
                return await _introspect_pgvector_dims_once(engine, schema)
    except RetryError as e:
        underlying = e.last_attempt.exception()
        raise StartupValidationError(
            f"could not validate embedding schema: {underlying}"
        ) from underlying
    # Unreachable: AsyncRetrying either returns from inside the loop or raises.
    raise StartupValidationError("embedding schema introspection did not run")


async def _introspect_pgvector_dims_once(
    engine: AsyncEngine, schema: str
) -> dict[str, int]:
    """Single-shot schema-qualified pg_attribute lookup.

    The join through ``pg_class``/``pg_namespace`` lets us respect
    ``DB.SCHEMA`` rather than relying on the ambient search_path.
    """
    query = text(
        """
        SELECT c.relname AS table_name, a.atttypmod AS typmod
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = :schema
          AND c.relname = ANY(:tables)
          AND a.attname = 'embedding'
        """
    )
    async with engine.connect() as conn:
        result = await conn.execute(
            query,
            {"schema": schema, "tables": list(_EMBEDDING_TABLES)},
        )
        return {row.table_name: row.typmod for row in result}


def _assert_pgvector_dims_match(
    dims: dict[str, int], *, schema: str, target_dim: int
) -> None:
    expected = set(_EMBEDDING_TABLES)
    missing = expected - dims.keys()
    if missing:
        listing = ", ".join(sorted(f"{schema}.{t}.embedding" for t in missing))
        raise StartupValidationError(
            f"Required vector columns missing: {listing}."
            + " Run `alembic upgrade head` first."
        )
    for table in sorted(expected):
        atttypmod = dims[table]
        if atttypmod == -1:
            raise StartupValidationError(
                f"{schema}.{table}.embedding has no declared vector dimension"
                + " (unbounded typmod). Run"
                + " `uv run python scripts/configure_embeddings.py`."
            )
        # pgvector stores the declared dim directly in atttypmod (no VARHDRSZ).
        actual = atttypmod
        if actual != target_dim:
            raise StartupValidationError(
                f"{schema}.{table}.embedding dim ({actual}) does not match"
                + f" EMBEDDING_VECTOR_DIMENSIONS ({target_dim}). Run"
                + " `uv run python scripts/configure_embeddings.py`"
                + " or fix EMBEDDING_VECTOR_DIMENSIONS."
            )


async def _sample_external_namespaces(engine: AsyncEngine, *, target_dim: int) -> None:
    """Best-effort dim check across existing external-store namespaces.

    External stores in this codebase are per-workspace and lazy-created on
    first write (see ``src.vector_store.get_vector_namespace``), so there is
    no canonical deployment-wide namespace to introspect. We enumerate up to
    ``_EXTERNAL_SAMPLE_LIMIT`` of each namespace category from the application
    DB and probe each:

    - Message namespaces — one per workspace.
    - Document namespaces — one per existing ``(workspace, observer, observed)``
      collection triple.

    Missing namespaces are OK; mismatched dims crash startup. Run
    ``configure_embeddings --report`` for full enumeration when a hard
    guarantee is needed.
    """
    workspace_names = await _sample_workspace_names(engine, _EXTERNAL_SAMPLE_LIMIT)
    collection_keys = await _sample_collection_keys(engine, _EXTERNAL_SAMPLE_LIMIT)
    if not workspace_names and not collection_keys:
        logger.info(
            "External-store validator: no workspaces or collections exist yet,"
            + " skipping sample"
        )
        return

    # Import lazily to avoid pulling in vector store deps when not configured.
    from src.vector_store import get_external_vector_store

    store = get_external_vector_store()
    if store is None:
        # Settings said TYPE != pgvector but the store could not be created.
        # That is its own problem and not for this validator to swallow.
        return

    candidates: list[str] = []
    for workspace_name in workspace_names:
        candidates.append(store.get_vector_namespace("message", workspace_name))
    for workspace_name, observer, observed in collection_keys:
        candidates.append(
            store.get_vector_namespace(
                "document",
                workspace_name,
                observer=observer,
                observed=observed,
            )
        )

    mismatches: list[tuple[str, int]] = []
    for namespace in candidates:
        actual_dim = await _probe_namespace_dim(store, namespace)
        if actual_dim is not None and actual_dim != target_dim:
            mismatches.append((namespace, actual_dim))

    if mismatches:
        formatted = ", ".join(f"{ns} (dim={d})" for ns, d in mismatches)
        raise StartupValidationError(
            f"Existing external-store namespaces have dim != {target_dim}:"
            + f" {formatted}. Run"
            + " `uv run python scripts/configure_embeddings.py --report`."
        )


async def _sample_workspace_names(engine: AsyncEngine, limit: int) -> list[str]:
    """Pull up to ``limit`` workspace names ordered by creation time.

    Uses the ORM ``Workspace`` model so ``Base.metadata.schema`` (configured
    from ``settings.DB.SCHEMA`` in ``src/db.py``) is honored automatically —
    a non-public schema deployment must not silently sample the wrong table.
    """
    stmt = select(Workspace.name).order_by(Workspace.created_at.desc()).limit(limit)
    async with engine.connect() as conn:
        result = await conn.execute(stmt)
        return [row[0] for row in result]


async def _sample_collection_keys(
    engine: AsyncEngine, limit: int
) -> list[tuple[str, str, str]]:
    """Pull up to ``limit`` ``(workspace_name, observer, observed)`` triples,
    one per existing collection row. Each triple corresponds to a document
    namespace that may exist in the external store."""
    stmt = (
        select(Collection.workspace_name, Collection.observer, Collection.observed)
        .order_by(Collection.created_at.desc())
        .limit(limit)
    )
    async with engine.connect() as conn:
        result = await conn.execute(stmt)
        return [(row[0], row[1], row[2]) for row in result]


async def _probe_namespace_dim(store: VectorStore, namespace: str) -> int | None:
    """Return the namespace's declared dim, or ``None`` if not present.

    Delegates to the store's own ``probe_namespace_dim`` implementation
    (lancedb opens the table, turbopuffer reads the schema). ``None`` means
    "lazy-create namespace, nothing to validate against."
    """
    return await store.probe_namespace_dim(namespace)
