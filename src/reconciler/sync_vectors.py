"""
Vector store reconciliation job.

This module provides a periodic reconciliation job that syncs documents and message
embeddings to the vector store on a rolling basis, healing any missed writes.
"""

import datetime
import logging
import time
from dataclasses import dataclass
from typing import cast

from sqlalchemy import and_, bindparam, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.vector_store import VectorRecord, VectorStore, get_external_vector_store

logger = logging.getLogger(__name__)

# Constants
RECONCILIATION_BATCH_SIZE = 50
RECONCILIATION_TIME_BUDGET_SECONDS = 240  # Leave headroom for other maintenance work
MAX_SYNC_ATTEMPTS = 5  # After this many failures, mark as failed


@dataclass
class ReconciliationMetrics:
    """Metrics for a reconciliation cycle."""

    documents_synced: int = 0
    documents_failed: int = 0
    documents_cleaned: int = 0
    message_embeddings_synced: int = 0
    message_embeddings_failed: int = 0

    @property
    def total_synced(self) -> int:
        return self.documents_synced + self.message_embeddings_synced

    @property
    def total_failed(self) -> int:
        return self.documents_failed + self.message_embeddings_failed

    @property
    def total_cleaned(self) -> int:
        return self.documents_cleaned


async def _get_documents_needing_sync(
    db: AsyncSession,
    batch_size: int = RECONCILIATION_BATCH_SIZE,
) -> list[models.Document]:
    """
    Get documents that need to be synced to the vector store.

    Finds documents where:
    - not soft-deleted (deleted_at is NULL)
    - sync_state is "pending" (never synced or retry needed)
    - Note: "synced" = done forever, "failed" = permanent failure (manual intervention)

    Uses FOR UPDATE SKIP LOCKED to prevent concurrent processing.
    """
    stmt = (
        select(models.Document)
        .where(
            and_(
                models.Document.deleted_at.is_(None),
                models.Document.sync_state == "pending",  # Only pending items
            )
        )
        .order_by(models.Document.last_sync_at.asc().nullsfirst())
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _get_message_embeddings_needing_sync(
    db: AsyncSession,
    batch_size: int = RECONCILIATION_BATCH_SIZE,
) -> list[models.MessageEmbedding]:
    """
    Get pending message embeddings that need to be synced to the vector store.

    Returns only pending embeddings (with full data including embedding vectors).
    The batch_size limits the number of embeddings returned.

    Uses FOR UPDATE SKIP LOCKED to prevent concurrent processing and
    orders by last_sync_at (nulls first) to prioritize never-synced records.

    Note: "synced" = done forever, "failed" = permanent failure (manual intervention)
    """
    stmt = (
        select(models.MessageEmbedding)
        .where(models.MessageEmbedding.sync_state == "pending")
        .order_by(models.MessageEmbedding.last_sync_at.asc().nullsfirst())
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _bump_document_sync_attempts(
    db: AsyncSession,
    documents: list[models.Document],
) -> None:
    if not documents:
        return

    for doc in documents:
        new_attempts = doc.sync_attempts + 1
        new_state = "failed" if new_attempts >= MAX_SYNC_ATTEMPTS else "pending"
        await db.execute(
            update(models.Document)
            .where(models.Document.id == doc.id)
            .values(
                sync_state=new_state,
                sync_attempts=new_attempts,
                last_sync_at=func.now(),
            )
        )


async def _bump_message_embedding_sync_attempts(
    db: AsyncSession,
    embeddings: list[models.MessageEmbedding],
) -> None:
    if not embeddings:
        return

    for emb in embeddings:
        new_attempts = emb.sync_attempts + 1
        new_state = "failed" if new_attempts >= MAX_SYNC_ATTEMPTS else "pending"

        await db.execute(
            update(models.MessageEmbedding)
            .where(models.MessageEmbedding.id == emb.id)
            .values(
                sync_state=new_state,
                sync_attempts=new_attempts,
                last_sync_at=func.now(),
            )
        )


async def _sync_documents(
    db: AsyncSession,
    documents: list[models.Document],
    external_vector_store: VectorStore,
) -> tuple[int, int]:
    """
    Sync a batch of pending documents to the external vector store.

    Handles three cases for each document:
    1. Embedding exists in postgres → use it for external upsert
    2. Embedding missing + need postgres storage → re-embed, write to both stores
    3. Embedding missing + external-only mode → re-embed, write to external only

    Returns (synced_count, failed_count).
    """
    if not documents:
        return 0, 0

    synced_count = 0
    failed_count = 0

    # True when using pgvector OR during migration (dual-write to both stores)
    store_in_postgres = (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )

    # Step 1: Re-embed documents missing embeddings in postgres (cases 2 & 3)
    docs_needing_embed = [
        doc for doc in documents if cast(list[float] | None, doc.embedding) is None
    ]
    freshly_embedded: dict[str, list[float]] = {}

    if docs_needing_embed:
        try:
            contents = [doc.content for doc in docs_needing_embed]
            new_embeddings = await embedding_client.simple_batch_embed(contents)

            if len(new_embeddings) != len(docs_needing_embed):
                logger.warning(
                    "Re-embedded %s/%s documents; remaining will be retried",
                    len(new_embeddings),
                    len(docs_needing_embed),
                )

            for doc, emb in zip(docs_needing_embed, new_embeddings, strict=False):
                freshly_embedded[doc.id] = emb

            # Case 2: Write to postgres if needed
            if store_in_postgres and freshly_embedded:
                stmt = (
                    update(models.Document)
                    .where(models.Document.id == bindparam("doc_id"))
                    .values(embedding=bindparam("emb"))
                )
                await db.execute(
                    stmt,
                    [
                        {"doc_id": doc_id, "emb": emb}
                        for doc_id, emb in freshly_embedded.items()
                    ],
                )
        except Exception:
            logger.exception("Failed to re-embed %s documents", len(docs_needing_embed))

    # Mark documents that failed to get an embedding
    failed_to_embed = [
        doc for doc in docs_needing_embed if doc.id not in freshly_embedded
    ]
    if failed_to_embed:
        await _bump_document_sync_attempts(db, failed_to_embed)
        failed_count += len(failed_to_embed)

    # Step 2: Build vector records and upsert to external store (all cases)
    by_namespace: dict[str, list[models.Document]] = {}
    for doc in documents:
        ns = external_vector_store.get_vector_namespace(
            "document", doc.workspace_name, doc.observer, doc.observed
        )
        by_namespace.setdefault(ns, []).append(doc)

    for namespace, docs in by_namespace.items():
        docs_to_sync: list[models.Document] = []
        vector_records: list[VectorRecord] = []

        for doc in docs:
            # Case 1: use existing embedding, Cases 2&3: use freshly embedded
            existing = cast(list[float] | None, doc.embedding)
            embedding = (
                existing if existing is not None else freshly_embedded.get(doc.id)
            )
            if embedding is None:
                continue

            vector_records.append(
                VectorRecord(
                    id=doc.id,
                    embedding=[float(x) for x in embedding],
                    metadata={
                        "workspace_name": doc.workspace_name,
                        "observer": doc.observer,
                        "observed": doc.observed,
                        "session_name": doc.session_name,
                        "level": doc.level,
                    },
                )
            )
            docs_to_sync.append(doc)

        if not vector_records:
            continue

        try:
            await external_vector_store.upsert_many(namespace, vector_records)
            await db.execute(
                update(models.Document)
                .where(models.Document.id.in_([d.id for d in docs_to_sync]))
                .values(sync_state="synced", last_sync_at=func.now(), sync_attempts=0)
            )
            synced_count += len(docs_to_sync)
        except Exception:
            logger.exception("Failed to sync documents to namespace %s", namespace)
            await _bump_document_sync_attempts(db, docs_to_sync)
            failed_count += len(docs_to_sync)

    return synced_count, failed_count


async def _sync_message_embeddings(
    db: AsyncSession,
    embeddings: list[models.MessageEmbedding],
    external_vector_store: VectorStore,
) -> tuple[int, int]:
    """
    Sync a batch of pending message embeddings to the external vector store.

    Handles three cases for each embedding:
    1. Embedding exists in postgres → use it for external upsert
    2. Embedding missing + need postgres storage → re-embed, write to both stores
    3. Embedding missing + external-only mode → re-embed, write to external only

    Returns (synced_count, failed_count).
    """
    if not embeddings:
        return 0, 0

    synced_count = 0
    failed_count = 0

    # True when using pgvector OR during migration (dual-write to both stores)
    store_in_postgres = (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )

    # Step 1: Re-embed message embeddings missing vectors in postgres (cases 2 & 3)
    embs_needing_embed: list[models.MessageEmbedding] = [
        emb for emb in embeddings if emb.embedding is None
    ]
    freshly_embedded: dict[int, list[float]] = {}

    if embs_needing_embed:
        try:
            contents = [emb.content for emb in embs_needing_embed]
            new_embeddings = await embedding_client.simple_batch_embed(contents)

            if len(new_embeddings) != len(embs_needing_embed):
                logger.warning(
                    "Re-embedded %s/%s message embeddings; remaining will be retried",
                    len(new_embeddings),
                    len(embs_needing_embed),
                )

            for emb, new_emb in zip(embs_needing_embed, new_embeddings, strict=False):
                freshly_embedded[emb.id] = new_emb

            # Case 2: Write to postgres if needed
            if store_in_postgres and freshly_embedded:
                stmt = (
                    update(models.MessageEmbedding)
                    .where(models.MessageEmbedding.id == bindparam("emb_id"))
                    .values(embedding=bindparam("emb"))
                )
                await db.execute(
                    stmt,
                    [
                        {"emb_id": emb_id, "emb": emb}
                        for emb_id, emb in freshly_embedded.items()
                    ],
                )
        except Exception:
            logger.exception(
                "Failed to re-embed %s message embeddings", len(embs_needing_embed)
            )

    # Mark embeddings that failed to get a vector
    failed_to_embed: list[models.MessageEmbedding] = [
        emb for emb in embs_needing_embed if emb.id not in freshly_embedded
    ]
    if failed_to_embed:
        await _bump_message_embedding_sync_attempts(db, failed_to_embed)
        failed_count += len(failed_to_embed)

    # Step 2: Compute chunk positions for vector IDs
    # Messages can be split into multiple chunks; we need {message_id}_{chunk_position}
    message_ids = list({emb.message_id for emb in embeddings})
    sibling_stmt = (
        select(models.MessageEmbedding.id, models.MessageEmbedding.message_id)
        .where(models.MessageEmbedding.message_id.in_(message_ids))
        .order_by(models.MessageEmbedding.message_id, models.MessageEmbedding.id)
    )
    sibling_rows = (await db.execute(sibling_stmt)).all()

    embs_by_message: dict[str, list[int]] = {}
    for emb_id, msg_id in sibling_rows:
        embs_by_message.setdefault(msg_id, []).append(emb_id)

    chunk_position: dict[int, int] = {}
    for emb_ids in embs_by_message.values():
        for pos, emb_id in enumerate(emb_ids):
            chunk_position[emb_id] = pos

    # Step 3: Build vector records and upsert to external store (all cases)
    by_namespace: dict[str, list[models.MessageEmbedding]] = {}
    for emb in embeddings:
        ns = external_vector_store.get_vector_namespace("message", emb.workspace_name)
        by_namespace.setdefault(ns, []).append(emb)

    for namespace, embs in by_namespace.items():
        embs_to_sync: list[models.MessageEmbedding] = []
        vector_records: list[VectorRecord] = []

        for emb in embs:
            # Case 1: use existing embedding, Cases 2&3: use freshly embedded
            existing = emb.embedding
            embedding = (
                existing if existing is not None else freshly_embedded.get(emb.id)
            )
            if embedding is None:
                continue

            vector_records.append(
                VectorRecord(
                    id=f"{emb.message_id}_{chunk_position[emb.id]}",
                    embedding=[float(x) for x in embedding],
                    metadata={
                        "message_id": emb.message_id,
                        "session_name": emb.session_name,
                        "peer_name": emb.peer_name,
                    },
                )
            )
            embs_to_sync.append(emb)

        if not vector_records:
            continue

        try:
            await external_vector_store.upsert_many(namespace, vector_records)
            await db.execute(
                update(models.MessageEmbedding)
                .where(models.MessageEmbedding.id.in_([e.id for e in embs_to_sync]))
                .values(sync_state="synced", last_sync_at=func.now(), sync_attempts=0)
            )
            synced_count += len(embs_to_sync)
        except Exception:
            logger.exception(
                "Failed to sync message embeddings to namespace %s", namespace
            )
            await _bump_message_embedding_sync_attempts(db, embs_to_sync)
            failed_count += len(embs_to_sync)

    return synced_count, failed_count


async def _cleanup_soft_deleted_documents_pgvector(
    db: AsyncSession,
    batch_size: int = RECONCILIATION_BATCH_SIZE,
    older_than_minutes: int = 5,
) -> int:
    """
    Cleanup soft-deleted documents
    """

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        minutes=older_than_minutes
    )

    # Find soft-deleted documents ready for cleanup
    stmt = (
        select(models.Document.id)
        .where(models.Document.deleted_at.is_not(None))
        .where(models.Document.deleted_at < cutoff)
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )
    result = await db.execute(stmt)
    doc_ids = [row[0] for row in result.all()]

    if not doc_ids:
        return 0

    # Hard delete directly (no vector store cleanup needed in pgvector mode)
    await db.execute(delete(models.Document).where(models.Document.id.in_(doc_ids)))
    logger.debug(f"Cleaned up {len(doc_ids)} soft-deleted documents (pgvector mode)")
    return len(doc_ids)


async def _reconcile_documents_batch(
    external_vector_store: VectorStore,
    metrics: ReconciliationMetrics,
) -> bool:
    """
    Reconcile a single batch of documents.

    Returns True if work was done, False otherwise.
    """
    async with tracked_db("reconciliation_docs") as db:
        docs = await _get_documents_needing_sync(db)
        if not docs:
            return False

        synced, failed = await _sync_documents(db, docs, external_vector_store)
        metrics.documents_synced += synced
        metrics.documents_failed += failed
        await db.commit()
        return True


async def _reconcile_message_embeddings_batch(
    external_vector_store: VectorStore,
    metrics: ReconciliationMetrics,
) -> bool:
    """
    Reconcile a single batch of message embeddings.

    Returns True if work was done, False otherwise.
    """
    async with tracked_db("reconciliation_embs") as db:
        embs = await _get_message_embeddings_needing_sync(db)
        if not embs:
            return False

        try:
            synced, failed = await _sync_message_embeddings(
                db, embs, external_vector_store
            )
        except Exception:
            logger.exception(
                "Message embedding reconciliation failed for %s embeddings",
                len(embs),
            )
            await _bump_message_embedding_sync_attempts(db, embs)
            synced = 0
            failed = len(embs)

        metrics.message_embeddings_synced += synced
        metrics.message_embeddings_failed += failed
        await db.commit()
        return True


async def _cleanup_documents_batch(
    external_vector_store: VectorStore,
    metrics: ReconciliationMetrics,
) -> bool:
    """
    Clean up a single batch of soft-deleted documents.

    Returns True if work was done, False otherwise.
    """
    from src.crud.document import cleanup_soft_deleted_documents

    async with tracked_db("reconciliation_cleanup") as db:
        cleaned = await cleanup_soft_deleted_documents(
            db,
            external_vector_store,
            batch_size=RECONCILIATION_BATCH_SIZE,
        )
        if not cleaned:
            return False

        metrics.documents_cleaned += cleaned
        await db.commit()
        return True


async def _cleanup_pgvector_batch(
    metrics: ReconciliationMetrics,
) -> bool:
    """
    Clean up a single batch of soft-deleted documents in pgvector-only mode.

    Returns True if work was done, False otherwise.
    """
    async with tracked_db("reconciliation_pgvector_cleanup") as db:
        cleaned = await _cleanup_soft_deleted_documents_pgvector(
            db, batch_size=RECONCILIATION_BATCH_SIZE
        )
        if not cleaned:
            return False

        metrics.documents_cleaned += cleaned
        await db.commit()
        return True


async def run_vector_reconciliation_cycle() -> ReconciliationMetrics:
    """
    Run a complete reconciliation cycle.

    Runs a rolling sweep to reconcile missing vectors and clean up soft deletes.
    Uses batching and FOR UPDATE SKIP LOCKED for safe concurrent operation.
    Each batch operation uses its own database session to avoid holding
    connections open for the entire cycle duration.

    Returns metrics about what was synced.
    """
    metrics = ReconciliationMetrics()
    external_vector_store = get_external_vector_store()
    deadline = time.monotonic() + RECONCILIATION_TIME_BUDGET_SECONDS

    # If no external vector store (pgvector mode), only clean up soft-deleted documents
    if external_vector_store is None:
        while time.monotonic() < deadline:
            did_work = await _cleanup_pgvector_batch(metrics)
            if not did_work:
                break
        logger.info("Vector reconciliation cycle completed (pgvector mode)")
        return metrics

    # External vector store mode - reconcile documents, embeddings, and cleanup
    while time.monotonic() < deadline:
        # Reconcile documents
        docs_work = await _reconcile_documents_batch(external_vector_store, metrics)

        if time.monotonic() >= deadline:
            break

        # Reconcile message embeddings
        embs_work = await _reconcile_message_embeddings_batch(
            external_vector_store, metrics
        )

        if time.monotonic() >= deadline:
            break

        # Clean up soft-deleted documents
        cleanup_work = await _cleanup_documents_batch(external_vector_store, metrics)

        # Continue only if any operation did work
        if not (docs_work or embs_work or cleanup_work):
            logger.debug("No work done, breaking reconciliation loop")
            break

    logger.info("Vector reconciliation cycle completed")
    return metrics
