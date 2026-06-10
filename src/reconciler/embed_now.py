"""
Immediate message-embedding fast path.

``create_messages`` writes ``MessageEmbedding`` rows as ``sync_state='pending'``
with no vector and defers embedding to the reconciler, which runs on a fixed
interval. To keep freshly created messages searchable within seconds (not
minutes), the message routers schedule ``embed_messages_now`` as a FastAPI
background task right after the response is sent. The reconciler remains the
fallback for anything this path leaves pending (failures, process restarts, or
rows it could not claim).

The fast path never holds a DB session across a network call (embedding or
external vector store): it claims and leases rows in one short transaction,
embeds with no session open, then persists in short transactions with any
external-store upserts running between them, not inside them. Running concurrently with the
reconciler is safe because the claim uses ``FOR UPDATE SKIP LOCKED`` and leases
rows by stamping ``last_sync_at``, which the reconciler's backoff filter then
skips.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.exceptions import VectorStoreError
from src.reconciler.sync_vectors import (
    _backoff_eligible,  # pyright: ignore[reportPrivateUsage]
    build_message_vector_record,
    compute_chunk_positions,
)
from src.telemetry.events import EmbeddingCallPurpose
from src.utils.types import embedding_call_purpose
from src.vector_store import VectorRecord, VectorStore, get_external_vector_store

logger = logging.getLogger(__name__)

_embed_semaphore: asyncio.Semaphore | None = None


def _get_embed_semaphore() -> asyncio.Semaphore:
    """Lazily create the embed-concurrency semaphore.

    Built on first use (not at import time) so it binds to the running event
    loop rather than whatever loop happened to exist at import.
    """
    global _embed_semaphore
    if _embed_semaphore is None:
        _embed_semaphore = asyncio.Semaphore(
            settings.EMBEDDING.MAX_CONCURRENT_EMBEDDINGS
        )
    return _embed_semaphore


def reset_embed_semaphore() -> None:
    """Test hook: drop the cached semaphore so the next call rebuilds it on the
    current event loop and current config."""
    global _embed_semaphore
    _embed_semaphore = None


@dataclass(frozen=True)
class _ClaimedChunk:
    """Plain snapshot of a claimed ``MessageEmbedding`` row.

    Captured before the claim transaction commits — after commit the ORM object
    is detached and attribute access would lazy-load against a closed session.
    """

    id: int
    message_id: str
    content: str
    workspace_name: str
    session_name: str | None
    peer_name: str | None


async def embed_messages_now(message_ids: list[str]) -> None:
    """Embed freshly created messages immediately, leaving the reconciler as the
    fallback for anything left pending.

    Args:
        message_ids: ``Message.public_id`` values (what
            ``MessageEmbedding.message_id`` references). Messages without
            embeddable content simply have no pending rows to claim.
    """
    if not message_ids:
        return

    # Runs as a fire-and-forget background task, so guard the whole flow: an
    # unhandled error here would escape into the server's task runner and be
    # lost. Any failure just leaves rows pending (claimed rows stay leased),
    # and the reconciler heals them on its next cycle.
    try:
        claimed = await _claim_and_lease(message_ids)
        if not claimed:
            return

        vectors = await _embed_chunks(claimed)
        if vectors is None:
            # Embedding failed; rows stay pending + leased, reconciler will retry.
            return

        await _persist(message_ids, claimed, vectors)
    except Exception:
        logger.exception(
            "Immediate embed failed for %s message(s); reconciler will retry",
            len(message_ids),
        )


async def _claim_and_lease(message_ids: list[str]) -> list[_ClaimedChunk]:
    """Phase 1 (short txn): claim eligible pending rows with FOR UPDATE SKIP
    LOCKED, lease them by stamping ``last_sync_at``, and snapshot their data.

    ``sync_attempts`` is intentionally left untouched: the reconciler owns retry
    accounting and the eventual ``sync_state='failed'`` backstop, so a transient
    embedding failure on this best-effort path never burns that budget.
    """
    async with tracked_db("embed_now_claim") as db:
        rows_stmt = (
            select(models.MessageEmbedding)
            .where(
                and_(
                    models.MessageEmbedding.message_id.in_(message_ids),
                    models.MessageEmbedding.sync_state == "pending",
                    _backoff_eligible(models.MessageEmbedding.last_sync_at),
                )
            )
            .order_by(models.MessageEmbedding.message_id, models.MessageEmbedding.id)
            .with_for_update(skip_locked=True)
        )
        rows = list((await db.execute(rows_stmt)).scalars().all())
        if not rows:
            return []

        claimed = [
            _ClaimedChunk(
                id=row.id,
                message_id=row.message_id,
                content=row.content,
                workspace_name=row.workspace_name,
                session_name=row.session_name,
                peer_name=row.peer_name,
            )
            for row in rows
        ]

        await db.execute(
            update(models.MessageEmbedding)
            .where(models.MessageEmbedding.id.in_([c.id for c in claimed]))
            .values(last_sync_at=func.now())
        )
        await db.commit()
        return claimed


async def _embed_chunks(claimed: list[_ClaimedChunk]) -> list[list[float]] | None:
    """Phase 2 (no DB session): embed the claimed chunk contents under the
    concurrency semaphore. Returns vectors in input order, or None on failure."""
    workspaces = {c.workspace_name for c in claimed}
    try:
        async with _get_embed_semaphore():
            with embedding_call_purpose(
                EmbeddingCallPurpose.MESSAGE_CREATE.value,
                workspace_name=workspaces.pop() if len(workspaces) == 1 else None,
                parent_category="api",
            ):
                return await embedding_client.simple_batch_embed(
                    [c.content for c in claimed]
                )
    except Exception:
        logger.exception(
            "Immediate embedding failed for %s chunk(s); reconciler will retry",
            len(claimed),
        )
        return None


async def _persist(
    message_ids: list[str],
    claimed: list[_ClaimedChunk],
    vectors: list[list[float]],
) -> None:
    """Phase 3: persist vectors and mark rows synced. On failure, rows stay
    pending (already leased) and the reconciler heals them.

    pgvector mode is one short transaction. External-store mode never holds a
    DB session across the vector-store network call: positions are read in one
    short transaction, the upserts run with no session open, and the surviving
    rows are marked synced in a second short transaction."""
    if len(vectors) != len(claimed):
        logger.warning(
            "Embedding count %s != claimed chunk count %s; skipping immediate persist, reconciler will heal",
            len(vectors),
            len(claimed),
        )
        return

    vector_by_id = {c.id: vec for c, vec in zip(claimed, vectors, strict=True)}
    # True for pgvector OR during migration (dual-write to both stores).
    store_in_postgres = (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )
    external = get_external_vector_store()

    if external is None:
        async with tracked_db("embed_now_persist") as db:
            await _persist_pgvector(db, claimed, vector_by_id)
            await db.commit()
        return

    synced = await _upsert_external(message_ids, claimed, vector_by_id, external)
    if not synced:
        return

    async with tracked_db("embed_now_persist") as db:
        await _mark_synced(db, synced, vector_by_id, store_in_postgres)
        await db.commit()


async def _persist_pgvector(
    db: AsyncSession,
    claimed: list[_ClaimedChunk],
    vector_by_id: dict[int, list[float]],
) -> None:
    """pgvector-only mode: write the vector and mark synced per row. The
    ``sync_state='pending'`` guard keeps us idempotent if the reconciler synced
    a row in the gap."""
    for c in claimed:
        await db.execute(
            update(models.MessageEmbedding)
            .where(
                and_(
                    models.MessageEmbedding.id == c.id,
                    models.MessageEmbedding.sync_state == "pending",
                )
            )
            .values(
                sync_state="synced",
                last_sync_at=func.now(),
                sync_attempts=0,
                embedding=vector_by_id[c.id],
            )
        )


async def _upsert_external(
    message_ids: list[str],
    claimed: list[_ClaimedChunk],
    vector_by_id: dict[int, list[float]],
    external: VectorStore,
) -> list[_ClaimedChunk]:
    """External-store mode: upsert vectors per namespace with no DB session
    open, returning the chunks whose namespaces upserted successfully.

    Chunk positions come from the shared helper (full sibling ordering) so vector
    ids match whatever the reconciler writes for any chunk we skipped; reading
    them is the only DB work here, done in its own short transaction before any
    network call."""
    async with tracked_db("embed_now_positions") as db:
        chunk_position = await compute_chunk_positions(db, message_ids)

    by_namespace: dict[str, list[_ClaimedChunk]] = {}
    for c in claimed:
        ns = external.get_vector_namespace("message", c.workspace_name)
        by_namespace.setdefault(ns, []).append(c)

    synced: list[_ClaimedChunk] = []
    for namespace, chunks in by_namespace.items():
        records: list[VectorRecord] = []
        synced_chunks: list[_ClaimedChunk] = []
        for c in chunks:
            pos = chunk_position.get(c.id)
            if pos is None:
                continue
            records.append(
                build_message_vector_record(
                    message_id=c.message_id,
                    chunk_position=pos,
                    session_name=c.session_name,
                    peer_name=c.peer_name,
                    embedding=vector_by_id[c.id],
                )
            )
            synced_chunks.append(c)

        if not records:
            continue

        try:
            await external.upsert_many(namespace, records)
        except VectorStoreError:
            logger.warning(
                "Vector store unavailable during immediate embed of namespace %s; reconciler will retry",
                namespace,
            )
            continue
        except Exception:
            logger.exception(
                "Unexpected error during immediate embed of namespace %s; reconciler will retry",
                namespace,
            )
            continue

        synced.extend(synced_chunks)

    return synced


async def _mark_synced(
    db: AsyncSession,
    chunks: list[_ClaimedChunk],
    vector_by_id: dict[int, list[float]],
    store_in_postgres: bool,
) -> None:
    """Mark upserted chunks synced (DB-only). The ``sync_state='pending'``
    guard keeps us idempotent if the reconciler synced a row in the gap."""
    for c in chunks:
        values: dict[str, Any] = {
            "sync_state": "synced",
            "last_sync_at": func.now(),
            "sync_attempts": 0,
        }
        if store_in_postgres:
            values["embedding"] = vector_by_id[c.id]
        await db.execute(
            update(models.MessageEmbedding)
            .where(
                and_(
                    models.MessageEmbedding.id == c.id,
                    models.MessageEmbedding.sync_state == "pending",
                )
            )
            .values(**values)
        )
