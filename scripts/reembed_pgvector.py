"""Re-embed pgvector rows after changing the configured embedding model/dimension.

This is intentionally boring and resumable: it only selects rows with
sync_state='pending' and NULL embedding, writes the new embedding into Postgres,
and marks the row synced. If interrupted, run it again.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.config import settings
from src.db import SessionLocal
from src.embedding_client import embedding_client

logger = logging.getLogger("reembed_pgvector")

TABLES = ("documents", "message_embeddings")


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.9g}" for x in vector) + "]"


async def _count_remaining(table: str) -> int:
    async with SessionLocal() as db:
        result = await db.execute(
            text(
                f"""
                select count(*)
                from {table}
                where sync_state = 'pending' and embedding is null
                """
            )
        )
        return int(result.scalar_one())


async def _select_batch(table: str, batch_size: int) -> list[tuple[str, str]]:
    id_expr = "id::text"
    async with SessionLocal() as db:
        result = await db.execute(
            text(
                f"""
                select {id_expr} as id, content
                from {table}
                where sync_state = 'pending' and embedding is null
                order by created_at nulls first, id
                limit :batch_size
                for update skip locked
                """
            ),
            {"batch_size": batch_size},
        )
        return [(row.id, row.content) for row in result]


async def _write_batch(table: str, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
    if len(ids) != len(vectors):
        raise RuntimeError(f"embedding count mismatch for {table}: {len(ids)} ids, {len(vectors)} vectors")

    expected_dim = settings.EMBEDDING.VECTOR_DIMENSIONS
    async with SessionLocal() as db:
        for row_id, vector in zip(ids, vectors, strict=True):
            if len(vector) != expected_dim:
                raise RuntimeError(
                    f"{table} row {row_id} returned dim {len(vector)}, expected {expected_dim}"
                )
            if table == "documents":
                stmt = text(
                    """
                    update documents
                    set embedding = cast(:embedding as vector),
                        sync_state = 'synced',
                        sync_attempts = 0,
                        last_sync_at = now()
                    where id = :id and embedding is null
                    """
                )
                params = {"id": row_id, "embedding": _vector_literal(vector)}
            else:
                stmt = text(
                    """
                    update message_embeddings
                    set embedding = cast(:embedding as vector),
                        sync_state = 'synced',
                        sync_attempts = 0,
                        last_sync_at = now()
                    where id = cast(:id as integer) and embedding is null
                    """
                )
                params = {"id": row_id, "embedding": _vector_literal(vector)}
            await db.execute(stmt, params)
        await db.commit()


async def reembed_table(table: str, batch_size: int, limit: int | None = None) -> int:
    processed = 0
    started = time.monotonic()
    while True:
        if limit is not None and processed >= limit:
            break
        this_batch_size = batch_size if limit is None else min(batch_size, limit - processed)
        rows = await _select_batch(table, this_batch_size)
        if not rows:
            break
        ids = [row_id for row_id, _content in rows]
        contents = [content for _row_id, content in rows]
        vectors = await embedding_client.simple_batch_embed(contents)
        await _write_batch(table, ids, vectors)
        processed += len(rows)
        remaining = await _count_remaining(table)
        elapsed = time.monotonic() - started
        rate = processed / elapsed if elapsed else 0
        logger.info(
            "%s: processed=%s remaining=%s rate=%.2f rows/s",
            table,
            processed,
            remaining,
            rate,
        )
    return processed


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=TABLES + ("all",), default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    tables = TABLES if args.table == "all" else (args.table,)
    total = 0
    for table in tables:
        before = await _count_remaining(table)
        logger.info("%s: starting remaining=%s", table, before)
        total += await reembed_table(table, args.batch_size, args.limit)
        after = await _count_remaining(table)
        logger.info("%s: done processed_now=%s remaining=%s", table, before - after, after)

    logger.info("total processed this run=%s", total)


if __name__ == "__main__":
    asyncio.run(main())
