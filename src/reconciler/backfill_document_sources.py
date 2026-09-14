"""Drain legacy JSONB source linkage into the document_sources table.

Before document_sources existed, a derived document's parents lived in one of
three JSONB locations on ``documents``: the ``source_ids`` column, or the
``internal_metadata`` keys ``source_ids`` / ``premise_ids``. This sweep copies
each row's linkage into the edge table and clears the JSONB so the row is not
revisited. It runs off the deriver queue in bounded batches, so a large tenant
never blocks the api pod's migration step, and the ORM fallback in
``Document.source_ids`` keeps undrained rows readable in the meantime.

Every batch is a single server-side statement; no row content is loaded into
Python. Once every row is drained the follow-up migration drops the column and
this task is removed.
"""

import logging
import time
from typing import Any, cast

from sqlalchemy import CursorResult, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import Base
from src.dependencies import tracked_db

logger = logging.getLogger(__name__)

BACKFILL_BATCH_SIZE = 500
BACKFILL_TIME_BUDGET_SECONDS = 240

# Any legacy location still populated. Also matches rows whose column holds a
# non-array JSON value, which the drain clears without emitting edges.
_PENDING_PREDICATE = """
    source_ids IS NOT NULL
    OR jsonb_exists_any(internal_metadata, ARRAY['source_ids', 'premise_ids'])
"""


def _qualified(table: str) -> str:
    schema = Base.metadata.schema
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def _drain_batch_sql() -> str:
    documents = _qualified("documents")
    document_sources = _qualified("document_sources")
    # Locking the batch in id order first (SKIP LOCKED, so never waiting)
    # keeps the UPDATE from cycling with id-ordered document writers.
    # Interpolates only the schema-qualified table names and a constant
    # predicate; batch_size is a bound parameter.
    return f"""
        WITH batch AS (
            SELECT id
            FROM {documents}
            WHERE {_PENDING_PREDICATE}
            ORDER BY id
            LIMIT :batch_size
            FOR UPDATE SKIP LOCKED
        ),
        copied AS (
            INSERT INTO {document_sources}
                (derived_id, source_id, position, workspace_name)
            SELECT DISTINCT ON (d.id, s.value)
                d.id, s.value, s.ord - 1, d.workspace_name
            FROM {documents} d
            JOIN batch b ON b.id = d.id
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
            ORDER BY d.id, s.value, s.ord
            ON CONFLICT DO NOTHING
        )
        UPDATE {documents} d
        SET source_ids = NULL,
            internal_metadata = d.internal_metadata - 'source_ids' - 'premise_ids'
        FROM batch b
        WHERE d.id = b.id
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
        async with tracked_db("reconciliation_document_sources") as db:
            count = await drain_document_sources_batch(db)
            await db.commit()
        if count == 0:
            break
        drained += count
    if drained:
        logger.info("Drained legacy source linkage for %d documents", drained)
    return drained
