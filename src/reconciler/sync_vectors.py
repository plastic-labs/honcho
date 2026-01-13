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

from sqlalchemy import and_, delete, select, update
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
    - embedding may be NULL (will be re-embedded during reconciliation)
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
    Sync a batch of documents to the external vector store.

    Returns (synced_count, failed_count).
    """
    if not documents:
        return 0, 0

    synced_count = 0
    failed_count = 0

    # Determine if we need to persist embeddings to postgres
    # True when: TYPE=pgvector OR still migrating (dual-write to both stores)
    store_embeddings_in_postgres = (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )

    missing_docs: list[models.Document] = []
    for doc in documents:
        if cast(list[float] | None, doc.embedding) is None:
            missing_docs.append(doc)
    reembedded_by_id: dict[str, list[float]] = {}

    if missing_docs:
        try:
            # Re-embed all missing documents in one batch
            contents = [doc.content for doc in missing_docs]
            embeddings = await embedding_client.simple_batch_embed(contents)

            if len(embeddings) != len(missing_docs):
                logger.warning(
                    "Re-embedded %s/%s documents; remaining will be retried",
                    len(embeddings),
                    len(missing_docs),
                )

            for doc, embedding in zip(missing_docs, embeddings, strict=False):
                reembedded_by_id[doc.id] = embedding

            # Write re-embedded vectors to postgres if needed
            if store_embeddings_in_postgres and reembedded_by_id:
                for doc_id, embedding in reembedded_by_id.items():
                    await db.execute(
                        update(models.Document)
                        .where(models.Document.id == doc_id)
                        .values(embedding=embedding)
                    )
        except Exception:
            logger.exception(
                "Failed to re-embed %s documents for reconciliation", len(missing_docs)
            )

    missing_after_embed: list[models.Document] = []
    for doc in documents:
        if (
            cast(list[float] | None, doc.embedding) is None
            and doc.id not in reembedded_by_id
        ):
            missing_after_embed.append(doc)
    if missing_after_embed:
        await _bump_document_sync_attempts(db, missing_after_embed)
        failed_count += len(missing_after_embed)

    # Group documents by namespace (workspace/observer/observed)
    by_namespace: dict[str, list[models.Document]] = {}
    for doc in documents:
        namespace = external_vector_store.get_vector_namespace(
            "document", doc.workspace_name, doc.observer, doc.observed
        )
        by_namespace.setdefault(namespace, []).append(doc)

    # Sync each namespace batch
    for namespace, docs in by_namespace.items():
        docs_with_vectors: list[models.Document] = []
        try:
            # Build vector records
            vector_records: list[VectorRecord] = []
            for doc in docs:
                doc_embedding = cast(list[float] | None, doc.embedding)
                embedding = (
                    doc_embedding
                    if doc_embedding is not None
                    else reembedded_by_id.get(doc.id)
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
                docs_with_vectors.append(doc)

            if not vector_records:
                continue

            doc_ids = [doc.id for doc in docs_with_vectors]

            if vector_records:
                await external_vector_store.upsert_many(namespace, vector_records)

            # Mark as synced
            await db.execute(
                update(models.Document)
                .where(models.Document.id.in_(doc_ids))
                .values(
                    sync_state="synced",
                    last_sync_at=func.now(),
                    sync_attempts=0,
                )
            )
            synced_count += len(docs_with_vectors)

        except Exception:
            logger.exception(
                "Failed to sync documents to external vector store %s",
                namespace,
            )
            # Increment attempts and mark as failed if we've hit max attempts
            await _bump_document_sync_attempts(db, docs_with_vectors)
            failed_count += len(docs_with_vectors)

    return synced_count, failed_count


async def _sync_message_embeddings(
    db: AsyncSession,
    embeddings: list[models.MessageEmbedding],
    external_vector_store: VectorStore,
) -> tuple[int, int]:
    """
    Sync a batch of pending message embeddings to the external vector store.

    Args:
        db: Database session
        embeddings: List of pending MessageEmbedding records to sync
        external_vector_store: External vector store to sync to

    Returns (synced_count, failed_count).
    """
    if not embeddings:
        return 0, 0

    synced_count = 0
    failed_count = 0

    # Determine if we need to persist embeddings to postgres
    # True when: TYPE=pgvector OR still migrating (dual-write to both stores)
    store_embeddings_in_postgres = (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )

    # Re-embed embeddings that are missing their vector payload
    missing_embs = [emb for emb in embeddings if emb.embedding is None]
    reembedded_by_id: dict[int, list[float]] = {}

    if missing_embs:
        try:
            for emb in missing_embs:
                new_embedding = await embedding_client.embed(emb.content)
                reembedded_by_id[emb.id] = new_embedding

                # Only persist embeddings to postgres when needed
                if store_embeddings_in_postgres:
                    await db.execute(
                        update(models.MessageEmbedding)
                        .where(models.MessageEmbedding.id == emb.id)
                        .values(embedding=new_embedding)
                    )
        except Exception:
            logger.exception(
                "Failed to re-embed %s message embeddings for reconciliation",
                len(missing_embs),
            )

    # Track embeddings that still don't have an embedding after re-embed attempt
    missing_after_embed: list[models.MessageEmbedding] = []
    for emb in embeddings:
        if emb.embedding is None and emb.id not in reembedded_by_id:
            missing_after_embed.append(emb)
    if missing_after_embed:
        await _bump_message_embedding_sync_attempts(db, missing_after_embed)
        failed_count += len(missing_after_embed)

    # Compute chunk position for each embedding within its parent message.
    # Messages can be split into multiple embedding chunks; we need to track
    # which chunk position (0, 1, 2, ...) each MessageEmbedding represents.
    # Fetch sibling embedding IDs (lightweight query) to compute correct positions.
    message_ids = list({emb.message_id for emb in embeddings})
    sibling_stmt = (
        select(models.MessageEmbedding.id, models.MessageEmbedding.message_id)
        .where(models.MessageEmbedding.message_id.in_(message_ids))
        .order_by(models.MessageEmbedding.message_id, models.MessageEmbedding.id)
    )
    sibling_result = await db.execute(sibling_stmt)
    sibling_rows = sibling_result.all()

    # Build position mapping from sibling IDs
    embeddings_by_message_id: dict[str, list[int]] = {}
    for emb_id, msg_id in sibling_rows:
        embeddings_by_message_id.setdefault(msg_id, []).append(emb_id)

    chunk_position_by_emb_id: dict[int, int] = {}
    for emb_ids in embeddings_by_message_id.values():
        # IDs are already sorted by the query
        for position, emb_id in enumerate(emb_ids):
            chunk_position_by_emb_id[emb_id] = position

    # Group embeddings by namespace (workspace)
    by_namespace: dict[str, list[models.MessageEmbedding]] = {}
    for emb in embeddings:
        namespace = external_vector_store.get_vector_namespace(
            "message", emb.workspace_name
        )
        by_namespace.setdefault(namespace, []).append(emb)

    # Sync each namespace batch
    for namespace, embs in by_namespace.items():
        embs_with_vectors: list[models.MessageEmbedding] = []
        try:
            # Build vector records with {message_id}_{chunk_index} format
            vector_records: list[VectorRecord] = []
            for emb in embs:
                embedding = (
                    emb.embedding
                    if emb.embedding is not None
                    else reembedded_by_id.get(emb.id)
                )
                if embedding is None:
                    continue

                # Use {message_id}_{chunk_position} as vector ID (consistent with creation)
                vector_id = f"{emb.message_id}_{chunk_position_by_emb_id[emb.id]}"

                vector_records.append(
                    VectorRecord(
                        id=vector_id,
                        embedding=[float(x) for x in embedding],
                        metadata={
                            "message_id": emb.message_id,
                            "session_name": emb.session_name,
                            "peer_name": emb.peer_name,
                        },
                    )
                )
                embs_with_vectors.append(emb)

            if not vector_records:
                continue

            emb_ids = [emb.id for emb in embs_with_vectors]

            if vector_records:
                await external_vector_store.upsert_many(namespace, vector_records)

            # Mark as synced
            await db.execute(
                update(models.MessageEmbedding)
                .where(models.MessageEmbedding.id.in_(emb_ids))
                .values(
                    sync_state="synced",
                    last_sync_at=func.now(),
                    sync_attempts=0,
                )
            )
            synced_count += len(embs_with_vectors)

        except Exception:
            logger.exception(
                "Failed to sync message embeddings to external vector store %s",
                namespace,
            )
            # Increment attempts and mark as failed if we've hit max attempts
            await _bump_message_embedding_sync_attempts(db, embs_with_vectors)
            failed_count += len(embs_with_vectors)

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


async def run_vector_reconciliation_cycle() -> ReconciliationMetrics:
    """
    Run a complete reconciliation cycle.

    Runs a rolling sweep to reconcile missing vectors and clean up soft deletes.
    Uses batching and FOR UPDATE SKIP LOCKED for safe concurrent operation.

    Returns metrics about what was synced.
    """
    metrics = ReconciliationMetrics()
    external_vector_store = get_external_vector_store()
    deadline = time.monotonic() + RECONCILIATION_TIME_BUDGET_SECONDS

    from src.crud.document import cleanup_soft_deleted_documents

    async with tracked_db("reconciliation") as db:
        # If no external vector store (pgvector mode), only clean up soft-deleted documents
        if external_vector_store is None:
            while time.monotonic() < deadline:
                cleaned = await _cleanup_soft_deleted_documents_pgvector(
                    db, batch_size=RECONCILIATION_BATCH_SIZE
                )
                if cleaned:
                    metrics.documents_cleaned += cleaned
                    await db.commit()
                else:
                    break
            return metrics

        while time.monotonic() < deadline:
            did_work = False

            # Reconcile documents
            docs = await _get_documents_needing_sync(db)
            if docs:
                synced, failed = await _sync_documents(db, docs, external_vector_store)
                metrics.documents_synced += synced
                metrics.documents_failed += failed
                await db.commit()
                did_work = True

            if time.monotonic() >= deadline:
                break

            # Reconcile message embeddings
            embs = await _get_message_embeddings_needing_sync(db)
            if embs:
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
                did_work = True

            if time.monotonic() >= deadline:
                break

            # Clean up soft-deleted documents
            cleaned = await cleanup_soft_deleted_documents(
                db,
                external_vector_store,
                batch_size=RECONCILIATION_BATCH_SIZE,
            )
            if cleaned:
                metrics.documents_cleaned += cleaned
                await db.commit()
                did_work = True

            if not did_work:
                logger.debug("No work done, breaking reconciliation loop")
                break
    logger.info("Vector reconciliation cycle completed")

    return metrics
