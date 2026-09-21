"""Drain legacy JSONB source linkage into the document_sources table.

Before document_sources existed, a derived document's parents lived in one of
three JSONB locations on ``documents``: the ``source_ids`` column, or the
``internal_metadata`` keys ``source_ids`` / ``premise_ids``. This sweep copies
each row's linkage into the edge table and clears the JSONB so the row is not
revisited. It runs off the deriver queue in bounded batches, so a large tenant
never blocks the api pod's migration step, and the ORM fallback in
``Document.source_ids`` keeps undrained rows readable in the meantime.

Every batch is a single server-side statement; no row content is loaded into
Python. A partial index over the pending predicate keeps both the enqueue
check and each batch proportional to the rows still pending rather than the
table, so the check is free once the drain completes. The follow-up migration
drops the column, the index, and this task.
"""

import logging
import time
from typing import Any, cast

from sqlalchemy import CursorResult, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import Base
from src.dependencies import service_db

logger = logging.getLogger(__name__)

BACKFILL_BATCH_SIZE = 500
BACKFILL_TIME_BUDGET_SECONDS = 240

# Any legacy location still populated. Also matches rows whose column holds a
# non-array JSON value, which the drain clears without emitting edges. Must
# stay textually identical to the ix_documents_legacy_sources_pending
# predicate so the planner can use that partial index.
_PENDING_PREDICATE = """
    source_ids IS NOT NULL
    OR internal_metadata ?| ARRAY['source_ids', 'premise_ids']
"""


def _qualified(table: str) -> str:
    schema = Base.metadata.schema
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def _drain_batch_sql() -> str:
    documents = _qualified("documents")
    document_sources = _qualified("document_sources")
    # Locking the batch in (tenant_id, id) order first (SKIP LOCKED, so never
    # waiting) keeps the UPDATE from cycling with id-ordered document writers.
    # Every join and the INSERT carry tenant_id: documents' key is
    # (tenant_id, id) and document_sources partitions on tenant_id.
    # Interpolates only the schema-qualified table names and a constant
    # predicate; batch_size is a bound parameter.
    return f"""
        WITH batch AS (
            SELECT tenant_id, id
            FROM {documents}
            WHERE {_PENDING_PREDICATE}
            ORDER BY tenant_id, id
            LIMIT :batch_size
            FOR UPDATE SKIP LOCKED
        ),
        copied AS (
            INSERT INTO {document_sources}
                (tenant_id, derived_id, source_id, position, workspace_name)
            SELECT DISTINCT ON (d.tenant_id, d.id, s.value)
                d.tenant_id, d.id, s.value, s.ord - 1, d.workspace_name
            FROM {documents} d
            JOIN batch b ON b.tenant_id = d.tenant_id AND b.id = d.id
            CROSS JOIN LATERAL jsonb_array_elements_text(
                CASE
                    WHEN jsonb_typeof(d.source_ids) = 'array'
                        THEN d.source_ids
                    WHEN jsonb_typeof(d.internal_metadata->'source_ids') = 'array'
                        THEN d.internal_metadata->'source_ids'
                    WHEN jsonb_typeof(d.internal_metadata->'premise_ids') = 'array'
                        THEN d.internal_metadata->'premise_ids'
                    ELSE '[]'::jsonb
                END
            ) WITH ORDINALITY AS s(value, ord)
            WHERE s.value ~ '^[A-Za-z0-9_-]{{21}}$'
            ORDER BY d.tenant_id, d.id, s.value, s.ord
            ON CONFLICT DO NOTHING
        )
        UPDATE {documents} d
        SET source_ids = NULL,
            internal_metadata = d.internal_metadata - 'source_ids' - 'premise_ids'
        FROM batch b
        WHERE d.tenant_id = b.tenant_id AND d.id = b.id
    """  # nosec B608


async def has_pending_document_sources(db: AsyncSession) -> bool:
    """Whether any document still carries legacy JSONB linkage."""
    row = await db.execute(
        text(
            f"SELECT 1 FROM {_qualified('documents')} WHERE {_PENDING_PREDICATE} LIMIT 1"  # nosec B608
        )
    )
    return row.first() is not None


async def drain_document_sources_batch(
    db: AsyncSession, batch_size: int = BACKFILL_BATCH_SIZE
) -> int:
    """Drain one id-ordered batch. Returns the number of documents cleared."""
    result = cast(
        CursorResult[Any],
        await db.execute(text(_drain_batch_sql()), {"batch_size": batch_size}),
    )
    return result.rowcount


async def run_document_sources_backfill_cycle() -> int:
    """Drain batches until none remain or the time budget is spent.

    Each batch commits in its own short-lived session so deriver work can
    interleave and lock footprint stays at one batch.
    """
    deadline = time.monotonic() + BACKFILL_TIME_BUDGET_SECONDS
    drained = 0
    while time.monotonic() < deadline:
        # ai: cross-tenant drain — a reconciler path, so it runs on the RLS-bypass
        # service session like sync_vectors; tracked_db would fail closed here
        # with no tenant in scope under MULTI_TENANT.
        async with service_db("reconciliation_document_sources") as db:
            count = await drain_document_sources_batch(db)
            await db.commit()
        if count == 0:
            break
        drained += count
    if drained:
        logger.info("Drained legacy source linkage for %d documents", drained)
    return drained
