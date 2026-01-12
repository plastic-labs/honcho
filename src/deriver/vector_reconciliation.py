"""
Vector store reconciliation job.

This module provides a periodic reconciliation job that syncs documents and message
embeddings to the vector store on a rolling basis, healing any missed writes.
"""

import logging
import time
from dataclasses import dataclass
from typing import cast

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.vector_store import VectorRecord, VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Constants
RECONCILIATION_BATCH_SIZE = 50
RECONCILIATION_TIME_BUDGET_SECONDS = 240  # Leave headroom for other maintenance work
MAX_SYNC_ATTEMPTS = 5  # After this many failures, mark as permanently_failed


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
    Get message embeddings that need to be synced to the vector store.

    Finds embeddings where:
    - has an embedding stored in the database
    - sync_state is "pending" (never synced or retry needed)
    - Note: "synced" = done forever, "failed" = permanent failure (manual intervention)

    Uses FOR UPDATE SKIP LOCKED to prevent concurrent processing.
    """
    stmt = (
        select(models.MessageEmbedding)
        .where(
            models.MessageEmbedding.sync_state == "pending"  # Only pending items
        )
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
    vector_store: VectorStore,
) -> tuple[int, int]:
    """
    Sync a batch of documents to the vector store.

    Returns (synced_count, failed_count).
    """
    if not documents:
        return 0, 0

    synced_count = 0
    failed_count = 0

    pgvector_in_use = (
        settings.VECTOR_STORE.PRIMARY_TYPE == "pgvector"
        or settings.VECTOR_STORE.SECONDARY_TYPE == "pgvector"
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

            # Write re-embedded vectors to postgres if pgvector is in use
            if pgvector_in_use and reembedded_by_id:
                for doc_id, embedding in reembedded_by_id.items():
                    await db.execute(
                        update(models.Document)
                        .where(models.Document.id == doc_id)
                        .values(embedding=embedding)
                    )
        except Exception as e:
            logger.warning(
                "Failed to re-embed %s documents for reconciliation: %s",
                len(missing_docs),
                e,
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
        namespace = vector_store.get_vector_namespace(
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

            result = None
            if vector_records:
                result = await vector_store.upsert_many(namespace, vector_records)

            if result is not None and result.secondary_ok is False:
                logger.warning(
                    "Partial sync for namespace %s: %s",
                    namespace,
                    result.secondary_error,
                )
                # Increment attempts and mark as failed if we've hit max attempts
                await _bump_document_sync_attempts(db, docs_with_vectors)
                failed_count += len(docs_with_vectors)
                continue

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

        except Exception as e:
            logger.warning(f"Failed to sync documents to {namespace}: {e}")
            # Increment attempts and mark as failed if we've hit max attempts
            await _bump_document_sync_attempts(db, docs_with_vectors)
            failed_count += len(docs_with_vectors)

    return synced_count, failed_count


async def _sync_message_embeddings(
    db: AsyncSession,
    embeddings: list[models.MessageEmbedding],
    vector_store: VectorStore,
) -> tuple[int, int]:
    """
    Sync a batch of message embeddings to the vector store.

    Returns (synced_count, failed_count).
    """
    if not embeddings:
        return 0, 0

    synced_count = 0
    failed_count = 0

    pgvector_in_use = (
        settings.VECTOR_STORE.PRIMARY_TYPE == "pgvector"
        or settings.VECTOR_STORE.SECONDARY_TYPE == "pgvector"
    )

    # Re-embed missing payloads so reconciliation can heal non-pgvector writes
    missing_embs = [emb for emb in embeddings if emb.embedding is None]
    reembedded_by_id: dict[int, list[float]] = {}

    if missing_embs:
        try:
            for emb in missing_embs:
                new_embedding = await embedding_client.embed(emb.content)
                reembedded_by_id[emb.id] = new_embedding

                # Only persist embeddings to postgres when pgvector is in play
                if pgvector_in_use:
                    await db.execute(
                        update(models.MessageEmbedding)
                        .where(models.MessageEmbedding.id == emb.id)
                        .values(embedding=new_embedding)
                    )
        except Exception as e:
            logger.warning(
                "Failed to re-embed %s message embeddings for reconciliation: %s",
                len(missing_embs),
                e,
            )

    missing_after_embed: list[models.MessageEmbedding] = []
    for emb in embeddings:
        if emb.embedding is None and emb.id not in reembedded_by_id:
            missing_after_embed.append(emb)
    if missing_after_embed:
        await _bump_message_embedding_sync_attempts(db, missing_after_embed)
        failed_count += len(missing_after_embed)

    # Group by namespace (workspace)
    by_namespace: dict[str, list[models.MessageEmbedding]] = {}
    for emb in embeddings:
        namespace = vector_store.get_vector_namespace("message", emb.workspace_name)
        by_namespace.setdefault(namespace, []).append(emb)

    # Compute chunk_index for each embedding based on message_id ordering
    # Group embeddings by message_id and assign chunk_index
    message_chunks: dict[str, list[models.MessageEmbedding]] = {}
    for emb in embeddings:
        message_chunks.setdefault(emb.message_id, []).append(emb)

    # Sort each message's chunks by id and assign chunk_index
    for chunks in message_chunks.values():
        chunks.sort(key=lambda e: e.id)
        for chunk_idx, chunk in enumerate(chunks):
            chunk._chunk_index = chunk_idx

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

                # Use {message_id}_{chunk_index} as vector ID (consistent with creation)
                vector_id = f"{emb.message_id}_{emb._chunk_index}"

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

            result = None
            if vector_records:
                result = await vector_store.upsert_many(namespace, vector_records)

            if result is not None and result.secondary_ok is False:
                logger.warning(
                    "Partial sync for namespace %s: %s",
                    namespace,
                    result.secondary_error,
                )
                # Increment attempts and mark as failed if we've hit max attempts
                await _bump_message_embedding_sync_attempts(db, embs_with_vectors)
                failed_count += len(embs_with_vectors)
                continue

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

        except Exception as e:
            logger.warning(f"Failed to sync message embeddings to {namespace}: {e}")
            # Increment attempts and mark as failed if we've hit max attempts
            await _bump_message_embedding_sync_attempts(db, embs_with_vectors)
            failed_count += len(embs_with_vectors)

    return synced_count, failed_count


async def run_vector_reconciliation_cycle() -> ReconciliationMetrics:
    """
    Run a complete reconciliation cycle.

    Runs a rolling sweep to reconcile missing vectors and clean up soft deletes.
    Uses batching and FOR UPDATE SKIP LOCKED for safe concurrent operation.

    Returns metrics about what was synced.
    """
    metrics = ReconciliationMetrics()
    vector_store = get_vector_store()
    deadline = time.monotonic() + RECONCILIATION_TIME_BUDGET_SECONDS

    from src.crud.document import cleanup_soft_deleted_documents

    print("Running vector reconciliation cycle")
    async with tracked_db("reconciliation") as db:
        while time.monotonic() < deadline:
            did_work = False

            # Reconcile documents
            docs = await _get_documents_needing_sync(db)
            if docs:
                synced, failed = await _sync_documents(db, docs, vector_store)
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
                        db, embs, vector_store
                    )
                except Exception as e:
                    logger.warning(
                        "Message embedding reconciliation failed for %s embeddings: %s",
                        len(embs),
                        e,
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
                vector_store,
                batch_size=RECONCILIATION_BATCH_SIZE,
            )
            if cleaned:
                metrics.documents_cleaned += cleaned
                did_work = True

            if not did_work:
                print("No work done, breaking")
                break
    print("Vector reconciliation cycle completed")

    return metrics
