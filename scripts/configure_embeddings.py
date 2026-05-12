"""Configure pgvector schema dim to match EMBEDDING_VECTOR_DIMENSIONS.

Usage::

    uv run python scripts/configure_embeddings.py              # interactive
    uv run python scripts/configure_embeddings.py --dry-run    # print intent, no DB write
    uv run python scripts/configure_embeddings.py --yes        # apply without prompt
    uv run python scripts/configure_embeddings.py --report     # full external-store inventory

The bootstrap sequence for a self-hosted install is:

    1. alembic upgrade head                              # creates default vector(1536) schema
    2. uv run python scripts/configure_embeddings.py     # ALTER columns to target dim
    3. start the API and deriver                         # validators refuse to start on mismatch

Existing 1536 deployments need no action — step 2 is a no-op when settings
already match the schema.

This script never creates or modifies external-store namespaces. Turbopuffer
and LanceDB namespaces are per-workspace and lazy-created on first write by
application code; their dim is implicitly pinned at that point. Use
``--report`` to enumerate existing namespaces against the configured dim.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass

# Match the path-shim convention used by the other scripts in this directory
# so `src.*` imports resolve when the script is run directly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine  # noqa: E402

from src.config import settings  # noqa: E402
from src.db import engine  # noqa: E402

logger = logging.getLogger(__name__)

_EMBEDDING_TABLES: tuple[str, ...] = ("documents", "message_embeddings")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PgvectorPlan:
    target_dim: int
    schema: str
    current_dims: dict[str, int]
    needs_alter: bool


@dataclass(frozen=True)
class _NamespaceRecord:
    """A single row in the --report output."""

    namespace: str
    status: str  # one of: "ok", "missing", "mismatch", "unknown"
    actual_dim: int | None
    target_dim: int


# ---------------------------------------------------------------------------
# pgvector phase
# ---------------------------------------------------------------------------


async def _introspect_pgvector(conn: AsyncConnection, schema: str) -> dict[str, int]:
    """Return ``{table_name: atttypmod}`` for embedding columns in ``schema``.

    Tables not present in the result dict are absent from the schema.
    pgvector stores the declared dim directly in ``atttypmod`` (no VARHDRSZ).
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
    result = await conn.execute(
        query,
        {"schema": schema, "tables": list(_EMBEDDING_TABLES)},
    )
    return {row.table_name: row.typmod for row in result}


async def _build_pgvector_plan(
    engine: AsyncEngine, target_dim: int, schema: str
) -> _PgvectorPlan:
    """Build a plan describing what (if anything) the script will change."""
    async with engine.connect() as conn:
        current = await _introspect_pgvector(conn, schema)

    missing = set(_EMBEDDING_TABLES) - current.keys()
    if missing:
        listing = ", ".join(sorted(f"{schema}.{t}.embedding" for t in missing))
        raise SystemExit(
            f"error: required vector columns missing: {listing}."
            + " Run `alembic upgrade head` first."
        )
    for table, typmod in current.items():
        if typmod == -1:
            raise SystemExit(
                f"error: {schema}.{table}.embedding has no declared vector"
                + " dimension (unbounded typmod). Drop and recreate the column"
                + " or restore from a versioned migration before re-running."
            )

    needs_alter = any(typmod != target_dim for typmod in current.values())
    return _PgvectorPlan(
        target_dim=target_dim,
        schema=schema,
        current_dims=current,
        needs_alter=needs_alter,
    )


async def _count_non_null_embeddings(
    conn: AsyncConnection, schema: str, table: str
) -> int:
    query = text(
        f'SELECT COUNT(*) AS n FROM "{schema}"."{table}" WHERE embedding IS NOT NULL'
    )
    result = await conn.execute(query)
    row = result.first()
    return int(row.n) if row is not None else 0


async def _fetch_hnsw_index_defs(
    conn: AsyncConnection, schema: str
) -> list[tuple[str, str]]:
    """Return ``(index_name, CREATE INDEX ...)`` for HNSW indices on the
    embedding columns. We re-CREATE them after the ALTER using these exact
    definitions, preserving operator-set params (m, ef_construction, etc.)."""
    query = text(
        """
        SELECT indexname AS name, indexdef AS ddl
        FROM pg_indexes
        WHERE schemaname = :schema
          AND tablename = ANY(:tables)
          AND indexdef ILIKE '%USING hnsw%'
        """
    )
    result = await conn.execute(
        query, {"schema": schema, "tables": list(_EMBEDDING_TABLES)}
    )
    return [(row.name, row.ddl) for row in result]


async def _apply_pgvector_alter(engine: AsyncEngine, plan: _PgvectorPlan) -> None:
    """ALTER the embedding columns to ``plan.target_dim`` in a single
    transaction. Refuses to proceed if any non-null embeddings exist.

    Sequence (inside the transaction):
      1. LOCK TABLE ... IN ACCESS EXCLUSIVE MODE  — closes the TOCTOU window
         between the population check and the ALTER.
      2. SELECT COUNT(embedding IS NOT NULL) per table — refuse if any > 0.
      3. Save HNSW index definitions, then DROP them (cannot ALTER under HNSW).
      4. ALTER ... ALTER COLUMN embedding TYPE vector(N) USING NULL.
      5. Recreate HNSW indices from saved definitions.
    """
    async with engine.begin() as conn:
        # Step 1: lock both tables for the duration of the transaction.
        for table in _EMBEDDING_TABLES:
            await conn.execute(
                text(f'LOCK TABLE "{plan.schema}"."{table}" IN ACCESS EXCLUSIVE MODE')
            )

        # Step 2: population check.
        counts: dict[str, int] = {}
        for table in _EMBEDDING_TABLES:
            counts[table] = await _count_non_null_embeddings(conn, plan.schema, table)
        populated = {t: n for t, n in counts.items() if n > 0}
        if populated:
            detail = ", ".join(f"{t}: {n} rows" for t, n in sorted(populated.items()))
            raise SystemExit(
                f"error: refusing to ALTER populated embedding tables ({detail})."
                + " This script only configures empty tables. Re-embed out-of-band"
                + " into a fresh deployment, then cut over."
            )

        # Step 3: snapshot + drop HNSW indices.
        index_defs = await _fetch_hnsw_index_defs(conn, plan.schema)
        for index_name, _ddl in index_defs:
            logger.info("dropping HNSW index %s", index_name)
            await conn.execute(text(f'DROP INDEX "{plan.schema}"."{index_name}"'))

        # Step 4: ALTER columns.
        for table in _EMBEDDING_TABLES:
            logger.info(
                "altering %s.%s.embedding to vector(%d)",
                plan.schema,
                table,
                plan.target_dim,
            )
            await conn.execute(
                text(
                    f'ALTER TABLE "{plan.schema}"."{table}"'
                    + f" ALTER COLUMN embedding TYPE vector({plan.target_dim})"
                    + " USING NULL"
                )
            )

        # Step 5: recreate HNSW indices from the saved definitions.
        for index_name, ddl in index_defs:
            logger.info("recreating HNSW index %s", index_name)
            await conn.execute(text(ddl))


# ---------------------------------------------------------------------------
# External-store report
# ---------------------------------------------------------------------------


async def _enumerate_workspaces(conn: AsyncConnection) -> list[str]:
    result = await conn.execute(text("SELECT name FROM workspaces ORDER BY created_at"))
    return [row.name for row in result]


async def _enumerate_collections(
    conn: AsyncConnection,
) -> list[tuple[str, str, str]]:
    result = await conn.execute(
        text("SELECT workspace_name, observer, observed FROM collections")
    )
    return [(row.workspace_name, row.observer, row.observed) for row in result]


async def _build_external_namespace_inventory(
    engine: AsyncEngine,
) -> list[tuple[str, str]]:
    """Return ``(namespace_type, namespace_name)`` pairs for every namespace
    that should exist based on the application DB. Message namespaces are
    derived per workspace, document namespaces per collection row.
    """
    from src.vector_store import get_external_vector_store

    store = get_external_vector_store()
    if store is None:
        return []

    async with engine.connect() as conn:
        workspace_names = await _enumerate_workspaces(conn)
        collection_keys = await _enumerate_collections(conn)

    pairs: list[tuple[str, str]] = []
    for workspace_name in workspace_names:
        pairs.append(("message", store.get_vector_namespace("message", workspace_name)))
    for workspace_name, observer, observed in collection_keys:
        pairs.append(
            (
                "document",
                store.get_vector_namespace(
                    "document", workspace_name, observer=observer, observed=observed
                ),
            )
        )
    return pairs


async def _probe_namespace_dim(store: object, namespace: str) -> int | None:
    """Best-effort: return the namespace's declared dim if introspectable.

    Returns ``None`` if the SDK does not expose a uniform dim accessor for
    the configured store. Future work can specialize per store; today the
    pgvector validator is the load-bearing dim safety.
    """
    _ = (store, namespace)
    return None


async def _emit_report(engine: AsyncEngine, target_dim: int) -> int:
    """Print the per-namespace inventory and return an exit code. 0 on a
    clean report (all matching or missing); non-zero on any mismatch.
    """
    if settings.VECTOR_STORE.TYPE == "pgvector":
        print("--report has no effect with VECTOR_STORE_TYPE=pgvector")
        return 0

    inventory = await _build_external_namespace_inventory(engine)
    if not inventory:
        print(
            "no external namespaces to inventory"
            + " (no workspaces/collections exist yet, or no external store configured)"
        )
        return 0

    from src.vector_store import get_external_vector_store

    store = get_external_vector_store()

    records: list[_NamespaceRecord] = []
    for _ns_type, namespace in inventory:
        actual = await _probe_namespace_dim(store, namespace) if store else None
        if actual is None:
            status = "unknown"
        elif actual == target_dim:
            status = "ok"
        else:
            status = "mismatch"
        records.append(
            _NamespaceRecord(
                namespace=namespace,
                status=status,
                actual_dim=actual,
                target_dim=target_dim,
            )
        )

    width = max(len(r.namespace) for r in records)
    print(f"{'namespace'.ljust(width)}  status     dim")
    print(f"{'-' * width}  ---------  ------")
    for r in records:
        dim_str = "?" if r.actual_dim is None else str(r.actual_dim)
        print(f"{r.namespace.ljust(width)}  {r.status:<9}  {dim_str}")

    mismatches = [r for r in records if r.status == "mismatch"]
    if mismatches:
        print(
            f"\nerror: {len(mismatches)} namespace(s) have dim != {target_dim}",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="configure_embeddings",
        description=(
            "Configure pgvector schema dim to match EMBEDDING_VECTOR_DIMENSIONS."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print intended changes and exit without touching the DB",
    )
    mode.add_argument(
        "--yes",
        action="store_true",
        help="apply changes without an interactive prompt",
    )
    mode.add_argument(
        "--report",
        action="store_true",
        help="print external-store namespace inventory and exit",
    )
    return parser


def _confirm(prompt: str) -> bool:
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


async def _async_main(args: argparse.Namespace) -> int:
    target_dim = settings.EMBEDDING.VECTOR_DIMENSIONS
    schema = settings.DB.SCHEMA

    if args.report:
        return await _emit_report(engine, target_dim)

    plan = await _build_pgvector_plan(engine, target_dim, schema)
    if not plan.needs_alter:
        print(
            f"pgvector: {schema}.documents.embedding and"
            + f" {schema}.message_embeddings.embedding already at dim {target_dim},"
            + " skipping ALTER"
        )
        return await _emit_report(engine, target_dim)

    current_summary = ", ".join(
        f"{schema}.{t}.embedding={plan.current_dims[t]}" for t in _EMBEDDING_TABLES
    )
    print(f"target dim: {target_dim}")
    print(f"current:    {current_summary}")
    print("planned operations (single transaction):")
    print(f"  - LOCK TABLE {schema}.documents IN ACCESS EXCLUSIVE MODE")
    print(f"  - LOCK TABLE {schema}.message_embeddings IN ACCESS EXCLUSIVE MODE")
    print("  - refuse if any non-null embeddings exist")
    print("  - DROP existing HNSW indices on the embedding columns")
    print(
        f"  - ALTER COLUMN embedding TYPE vector({target_dim}) USING NULL"
        + " on both tables"
    )
    print("  - CREATE HNSW indices from snapshotted definitions")

    if args.dry_run:
        print("\n--dry-run: no changes applied")
        return 0

    if not args.yes and not _confirm("apply?"):
        print("aborted")
        return 1

    await _apply_pgvector_alter(engine, plan)
    print(f"\npgvector schema is now at dim {target_dim}")
    return await _emit_report(engine, target_dim)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    finally:
        # Best-effort cleanup; the script is short-lived so a leak here is
        # harmless but keeping the dispose explicit avoids "engine not
        # disposed" warnings during tests.
        asyncio.run(engine.dispose())


if __name__ == "__main__":
    raise SystemExit(main())
