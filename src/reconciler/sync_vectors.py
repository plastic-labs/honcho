"""
Vector store reconciliation job.

This module provides a periodic reconciliation job that syncs documents and message
embeddings to the vector store on a rolling basis, healing any missed writes.
"""

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass
from typing import Any, cast

import sentry_sdk
from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ColumnElement
from sqlalchemy.sql.functions import func

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import EmbeddingTokenLimitError, embedding_client
from src.exceptions import VectorStoreError
from src.telemetry import prometheus_metrics
from src.telemetry.events import EmbeddingCallPurpose
from src.utils.types import embedding_call_purpose
from src.vector_store import VectorRecord, VectorStore, get_external_vector_store

logger = logging.getLogger(__name__)

# Constants
RECONCILIATION_BATCH_SIZE = 50
RECONCILIATION_TIME_BUDGET_SECONDS = 240  # Leave headroom for other maintenance work
MAX_SYNC_ATTEMPTS = 20  # After this many failures, mark as failed
# Flat wait between sync attempts. With MAX_SYNC_ATTEMPTS=20 this gives ~3 hours
# of outage headroom before a row is marked failed.
SYNC_BACKOFF = datetime.timedelta(minutes=10)


def backoff_eligible(
    last_sync_at: InstrumentedAttribute[datetime.datetime | None],
) -> ColumnElement[bool]:
    """Rows are eligible for sync if never attempted or past the backoff window."""
    return or_(
        last_sync_at.is_(None),
        last_sync_at < func.now() - SYNC_BACKOFF,
    )


async def has_pending_work(db: AsyncSession) -> bool:
    """True when a reconciliation cycle would find something to sync or clean up."""
    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)
    checks = [
        select(models.MessageEmbedding.id).where(
            models.MessageEmbedding.sync_state == "pending",
            backoff_eligible(models.MessageEmbedding.last_sync_at),
        ),
        select(models.Document.id).where(
            models.Document.deleted_at.is_not(None), models.Document.deleted_at < cutoff
        ),
    ]
    if get_external_vector_store() is not None:
        checks.append(
            select(models.Document.id).where(
                models.Document.deleted_at.is_(None),
                models.Document.sync_state == "pending",
                backoff_eligible(models.Document.last_sync_at),
            )
        )
    return any([await db.scalar(c.limit(1)) is not None for c in checks])


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
                backoff_eligible(models.Document.last_sync_at),
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

    Claims up to `batch_size` distinct message_ids that have at least one
    eligible pending row, then loads ALL pending rows for those message_ids.
    This guarantees a single message's chunks are always processed together in
    one batch, which keeps vector-ID assignment (`{message_id}_{chunk_index}`,
    derived from row-id ordering) stable across reconciler cycles.

    Uses FOR UPDATE SKIP LOCKED on the per-row claim so concurrent reconcilers
    don't double-process the same chunks.

    Note: "synced" = done forever, "failed" = permanent failure (manual intervention)
    """
    # Step 1: pick distinct message_ids with at least one eligible pending row,
    # prioritizing those with the oldest last_sync_at.
    msg_id_stmt = (
        select(
            models.MessageEmbedding.message_id,
            func.min(models.MessageEmbedding.last_sync_at).label("oldest_attempt"),
        )
        .where(
            and_(
                models.MessageEmbedding.sync_state == "pending",
                backoff_eligible(models.MessageEmbedding.last_sync_at),
            )
        )
        .group_by(models.MessageEmbedding.message_id)
        .order_by(func.min(models.MessageEmbedding.last_sync_at).asc().nullsfirst())
        .limit(batch_size)
    )
    msg_id_rows = (await db.execute(msg_id_stmt)).all()
    message_ids = [row[0] for row in msg_id_rows]
    if not message_ids:
        return []

    # Step 2: claim all pending rows for those messages. Skip rows another
    # reconciler holds; if we can't claim every chunk of a message right now,
    # the message will be retried next cycle.
    rows_stmt = (
        select(models.MessageEmbedding)
        .where(
            and_(
                models.MessageEmbedding.message_id.in_(message_ids),
                models.MessageEmbedding.sync_state == "pending",
                backoff_eligible(models.MessageEmbedding.last_sync_at),
            )
        )
        .order_by(models.MessageEmbedding.message_id, models.MessageEmbedding.id)
        .with_for_update(skip_locked=True)
    )
    result = await db.execute(rows_stmt)
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
    embedding_ids: list[int],
) -> None:
    """Increment ``sync_attempts`` for the given rows, marking them ``failed``
    once the retry cap is reached.

    Id-based (not ORM-object-based) so the persist phase can run in a fresh
    transaction after the claim session has been closed. Re-reads each row's
    current ``sync_attempts`` under ``FOR UPDATE`` rather than trusting a
    snapshot, since a concurrent writer may have touched the row since the
    claim.
    """
    if not embedding_ids:
        return

    rows = (
        (
            await db.execute(
                select(models.MessageEmbedding)
                .where(
                    models.MessageEmbedding.id.in_(embedding_ids),
                    models.MessageEmbedding.sync_state == "pending",
                )
                .with_for_update()
            )
        )
        .scalars()
        .all()
    )
    for emb in rows:
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


async def _reclaim_still_owned_message_embeddings(
    db: AsyncSession,
    embedding_ids: list[int],
) -> set[int]:
    """Re-verify ownership and re-lease rows immediately before an external
    upsert, returning the subset still owned.

    The claim lease is soft: a row becomes re-claimable once ``last_sync_at``
    passes ``SYNC_BACKOFF``. The embed phase can outrun that window, so by the
    time this worker reaches the external upsert another worker may have
    claimed and completed the same row. An unfenced upsert would then overwrite
    the newer worker's external vector with this worker's (possibly stale)
    result, leaving the external index inconsistent with the completed DB row.

    Re-claiming under ``FOR UPDATE SKIP LOCKED`` fences that window: only the
    worker that actually holds the row lock — i.e. still owns a ``pending``
    row — may proceed, and re-stamping ``last_sync_at`` extends the lease so no
    other worker can claim the row for another ``SYNC_BACKOFF`` window (far
    longer than the single upsert call that follows). Rows another worker
    already completed are no longer ``pending`` and are excluded, so this
    worker neither overwrites them nor counts them as synced.
    """
    if not embedding_ids:
        return set()

    rows = (
        (
            await db.execute(
                select(models.MessageEmbedding.id)
                .where(
                    models.MessageEmbedding.id.in_(embedding_ids),
                    models.MessageEmbedding.sync_state == "pending",
                )
                .with_for_update(skip_locked=True)
            )
        )
        .scalars()
        .all()
    )
    owned = set(rows)
    if owned:
        await db.execute(
            update(models.MessageEmbedding)
            .where(models.MessageEmbedding.id.in_(owned))
            .values(last_sync_at=func.now())
        )
    return owned


async def compute_chunk_positions(
    db: AsyncSession, message_ids: list[str]
) -> dict[int, int]:
    """Map each MessageEmbedding row id to its 0-indexed chunk position within
    its message.

    Positions are derived from the full set of sibling rows for each message,
    ordered by ``(message_id, id)`` — never from a partial subset — so the
    ``{message_id}_{chunk_position}`` vector id stays stable no matter which
    rows a given caller claimed. Shared by the reconciler and the immediate
    embed path so the two writers always agree on vector ids.
    """
    if not message_ids:
        return {}

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
    return chunk_position


def build_message_vector_record(
    *,
    message_id: str,
    chunk_position: int,
    session_name: str | None,
    peer_name: str | None,
    embedding: list[float],
) -> VectorRecord:
    """Build the external-store record for one message-embedding chunk.

    Single source of the ``{message_id}_{chunk_position}`` vector id and the
    metadata shape, shared by the reconciler and the immediate embed path.
    """
    return VectorRecord(
        id=f"{message_id}_{chunk_position}",
        embedding=[float(x) for x in embedding],
        metadata={
            "message_id": message_id,
            "session_name": session_name,
            "peer_name": peer_name,
        },
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
            with embedding_call_purpose(
                EmbeddingCallPurpose.VECTOR_SYNC.value,
                parent_category="reconciliation",
            ):
                new_embeddings = await embedding_client.simple_batch_embed(
                    contents, on_oversize="truncate"
                )

            if len(new_embeddings) != len(docs_needing_embed):
                logger.warning(
                    "Re-embedded %s/%s documents; remaining will be retried",
                    len(new_embeddings),
                    len(docs_needing_embed),
                )

            for doc, emb in zip(docs_needing_embed, new_embeddings, strict=False):
                freshly_embedded[doc.id] = emb
                if store_in_postgres:
                    doc.embedding = emb
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
        except VectorStoreError:
            logger.warning(
                "Vector store unavailable while syncing namespace %s", namespace
            )
            await _bump_document_sync_attempts(db, docs_to_sync)
            failed_count += len(docs_to_sync)
        except Exception:
            logger.exception(
                "Unexpected error syncing documents to namespace %s", namespace
            )
            await _bump_document_sync_attempts(db, docs_to_sync)
            failed_count += len(docs_to_sync)

    return synced_count, failed_count


@dataclass
class _ClaimedEmbedding:
    """Plain snapshot of a claimed ``MessageEmbedding`` row.

    Captured before the claim transaction commits — after commit the ORM object
    is detached, and the embed + persist phases must not lazy-load against a
    closed session, so they work off this snapshot instead.
    """

    id: int
    message_id: str
    content: str
    workspace_name: str
    session_name: str
    peer_name: str
    embedding: Any | None  # pre-existing stored vector, if the row already had one


async def _claim_and_lease_message_embeddings(
    db: AsyncSession,
) -> list[_ClaimedEmbedding]:
    """Phase 1 (short txn): claim pending rows (``FOR UPDATE SKIP LOCKED`` via
    ``_get_message_embeddings_needing_sync``), lease them by stamping
    ``last_sync_at`` so a concurrent reconciler skips them, and snapshot their
    data into plain dataclasses.

    The caller commits immediately after this returns, releasing both the row
    locks and the pooled connection before any embedding network call.
    """
    rows = await _get_message_embeddings_needing_sync(db)
    if not rows:
        return []

    claimed = [
        _ClaimedEmbedding(
            id=row.id,
            message_id=row.message_id,
            content=row.content,
            workspace_name=row.workspace_name,
            session_name=row.session_name,
            peer_name=row.peer_name,
            embedding=row.embedding,
        )
        for row in rows
    ]
    await db.execute(
        update(models.MessageEmbedding)
        .where(models.MessageEmbedding.id.in_([c.id for c in claimed]))
        .values(last_sync_at=func.now())
    )
    return claimed


async def _embed_claimed(
    claimed: list[_ClaimedEmbedding],
) -> tuple[dict[int, list[float]], set[int]]:
    """Phase 2 (no DB session): embed each claimed chunk that is missing a
    vector. Returns ``(freshly_embedded, permanently_failed)``.

    Embedding is per-text so a single oversized text (rejected by the
    provider) can't poison the whole batch. The chunking tokenizer can
    undercount a provider's real token count (e.g. tiktoken o200k vs
    nomic-embed-text), so oversized texts can still slip through the chunk cap
    and get rejected by the provider. Isolating per-text keeps the rest of the
    batch embeddable and lets genuinely-oversized rows fail on their own.

    Only ``EmbeddingTokenLimitError`` (oversized input) is treated as a
    permanent failure. Every other ``ValueError`` — count/dimension mismatch,
    "no embedding returned" — is a transient provider response error and stays
    retryable, so a flaky response can't permanently fail a message.
    """
    freshly_embedded: dict[int, list[float]] = {}
    permanently_failed: set[int] = set()

    embs_needing_embed = [c for c in claimed if c.embedding is None]
    if not embs_needing_embed:
        return freshly_embedded, permanently_failed

    workspaces = {c.workspace_name for c in embs_needing_embed}
    with embedding_call_purpose(
        EmbeddingCallPurpose.MESSAGE_CREATE.value,
        workspace_name=workspaces.pop() if len(workspaces) == 1 else None,
        parent_category="reconciliation",
    ):
        # Embed each text individually so a single oversized text can't poison
        # the batch, but run them concurrently (bounded) so a large batch of
        # sequential provider round trips doesn't burn the reconciliation time
        # budget and let early-claimed row leases expire before persist.
        sem = asyncio.Semaphore(8)

        async def _embed_one(c: _ClaimedEmbedding) -> None:
            async with sem:
                try:
                    new_emb = await embedding_client.simple_batch_embed([c.content])
                    freshly_embedded[c.id] = new_emb[0]
                except EmbeddingTokenLimitError as e:
                    # Oversized input is a permanent validation failure — the
                    # same text will never embed. Mark failed immediately
                    # instead of burning MAX_SYNC_ATTEMPTS retries on it.
                    logger.warning(
                        "Message %s chunk %s oversized for embedding provider: %s",
                        c.message_id,
                        c.id,
                        e,
                    )
                    permanently_failed.add(c.id)
                except ValueError as e:
                    # Transient provider response errors (count/dimension
                    # mismatch, no embedding returned) — retryable, keep the
                    # row pending.
                    logger.warning(
                        "Transient embedding error for message %s chunk %s; will retry: %s",
                        c.message_id,
                        c.id,
                        e,
                    )
                except Exception:
                    logger.exception(
                        "Unexpected error embedding message %s chunk %s; will retry",
                        c.message_id,
                        c.id,
                    )

        await asyncio.gather(*(_embed_one(c) for c in embs_needing_embed))
    return freshly_embedded, permanently_failed


async def _persist_message_embeddings(
    claimed: list[_ClaimedEmbedding],
    freshly_embedded: dict[int, list[float]],
    permanently_failed: set[int],
    external_vector_store: VectorStore | None,
) -> tuple[int, int]:
    """Phase 3 (short txns): persist embedding results.

    Never holds a DB session across a network call. Failure accounting (bump
    attempts / mark permanently failed) and success marking each run in their
    own short transactions; external-store upserts run with no session open.

    Every write is fenced with a ``sync_state == "pending"`` predicate: the
    claim lease can lapse (or another writer can claim the row) while embedding
    or upserting, and a late write from this worker must not regress a row
    that has already left ``pending``.
    """
    synced_count = 0
    failed_count = 0

    # True when using pgvector OR during migration (dual-write to both stores)
    store_in_postgres = (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )

    # Rows that never got a vector: permanent validation failures vs retryable.
    failed_ids = [
        c.id for c in claimed if c.embedding is None and c.id not in freshly_embedded
    ]
    permanent = [i for i in failed_ids if i in permanently_failed]
    retryable = [i for i in failed_ids if i not in permanently_failed]

    if permanent or retryable:
        async with tracked_db("reconciliation_embs_fail") as db:
            if permanent:
                await db.execute(
                    update(models.MessageEmbedding)
                    .where(
                        models.MessageEmbedding.id.in_(permanent),
                        models.MessageEmbedding.sync_state == "pending",
                    )
                    .values(sync_state="failed", last_sync_at=func.now())
                )
            if retryable:
                await _bump_message_embedding_sync_attempts(db, retryable)
            await db.commit()
        failed_count += len(permanent) + len(retryable)

    # Vector per claimed row that now has one (pre-existing or freshly embedded).
    vector_by_id: dict[int, list[float]] = {}
    for c in claimed:
        if c.embedding is not None:
            vector_by_id[c.id] = cast(list[float], c.embedding)
        elif c.id in freshly_embedded:
            vector_by_id[c.id] = freshly_embedded[c.id]

    if not vector_by_id:
        return synced_count, failed_count

    if external_vector_store is None:
        # pgvector-only mode: write vectors + mark synced in one short txn.
        async with tracked_db("reconciliation_embs_persist") as db:
            for c in claimed:
                if c.id not in vector_by_id:
                    continue
                sync_values: dict[str, Any] = {
                    "sync_state": "synced",
                    "last_sync_at": func.now(),
                    "sync_attempts": 0,
                }
                if c.id in freshly_embedded:
                    sync_values["embedding"] = freshly_embedded[c.id]
                await db.execute(
                    update(models.MessageEmbedding)
                    .where(
                        models.MessageEmbedding.id == c.id,
                        models.MessageEmbedding.sync_state == "pending",
                    )
                    .values(**sync_values)
                )
            await db.commit()
            synced_count += len(vector_by_id)
        return synced_count, failed_count

    # External-store mode: positions in one short txn, upserts with no session,
    # then mark synced / bump attempts in short txns.
    message_ids = list({c.message_id for c in claimed})
    async with tracked_db("reconciliation_embs_positions") as db:
        chunk_position = await compute_chunk_positions(db, message_ids)

    # Re-verify ownership and re-lease right before writing to the external
    # store. The embed phase runs with no session and can outrun the soft lease
    # (SYNC_BACKOFF), so another worker may have claimed and completed some of
    # these rows in the meantime. Only rows we still hold may be upserted, so a
    # late worker can't overwrite a newer worker's external vector.
    async with tracked_db("reconciliation_embs_reclaim") as db:
        owned_ids = await _reclaim_still_owned_message_embeddings(
            db, list(vector_by_id)
        )
        await db.commit()
    if not owned_ids:
        return synced_count, failed_count

    by_namespace: dict[str, list[_ClaimedEmbedding]] = {}
    for c in claimed:
        if c.id not in owned_ids:
            continue
        ns = external_vector_store.get_vector_namespace("message", c.workspace_name)
        by_namespace.setdefault(ns, []).append(c)

    synced_ids: list[int] = []
    for namespace, chunks in by_namespace.items():
        records: list[VectorRecord] = []
        ns_synced: list[int] = []
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
            ns_synced.append(c.id)

        if not records:
            continue

        try:
            await external_vector_store.upsert_many(namespace, records)
        except VectorStoreError:
            logger.warning(
                "Vector store unavailable while syncing message embeddings to namespace %s",
                namespace,
            )
            async with tracked_db("reconciliation_embs_retry") as db:
                await _bump_message_embedding_sync_attempts(db, ns_synced)
                await db.commit()
            failed_count += len(ns_synced)
            continue
        except Exception:
            logger.exception(
                "Unexpected error syncing message embeddings to namespace %s",
                namespace,
            )
            async with tracked_db("reconciliation_embs_retry") as db:
                await _bump_message_embedding_sync_attempts(db, ns_synced)
                await db.commit()
            failed_count += len(ns_synced)
            continue

        synced_ids.extend(ns_synced)

    if synced_ids:
        async with tracked_db("reconciliation_embs_synced") as db:
            for c in claimed:
                if c.id not in synced_ids:
                    continue
                values: dict[str, Any] = {
                    "sync_state": "synced",
                    "last_sync_at": func.now(),
                    "sync_attempts": 0,
                }
                if store_in_postgres and c.id in freshly_embedded:
                    values["embedding"] = freshly_embedded[c.id]
                await db.execute(
                    update(models.MessageEmbedding)
                    .where(
                        models.MessageEmbedding.id == c.id,
                        models.MessageEmbedding.sync_state == "pending",
                    )
                    .values(**values)
                )
            await db.commit()
        synced_count += len(synced_ids)

    return synced_count, failed_count


async def _cleanup_soft_deleted_documents_pgvector(
    db: AsyncSession,
    batch_size: int = RECONCILIATION_BATCH_SIZE,
    older_than_minutes: int = 5,
) -> int:
    """
    Cleanup soft-deleted documents
    """

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
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

        with sentry_sdk.start_transaction(
            name="reconcile_documents_batch", op="reconciler"
        ):
            synced, failed = await _sync_documents(db, docs, external_vector_store)
            metrics.documents_synced += synced
            metrics.documents_failed += failed
            await db.commit()
        return True


async def _reconcile_message_embeddings_batch(
    external_vector_store: VectorStore | None,
    metrics: ReconciliationMetrics,
) -> bool:
    """
    Reconcile a single batch of message embeddings.

    Three phases mirror the immediate-embed path and never hold a DB session
    across a network call: claim+lease+snapshot in one short transaction
    (releasing the row locks), embed with no session open, then persist in
    short transactions. Failure accounting is the reconciler's own (retry bump
    vs permanent-fail), unlike the best-effort immediate path.

    Returns True if work was done, False otherwise.
    """
    async with tracked_db("reconciliation_embs") as db:
        claimed = await _claim_and_lease_message_embeddings(db)
        if not claimed:
            return False
        await db.commit()

    with sentry_sdk.start_transaction(
        name="reconcile_message_embeddings_batch", op="reconciler"
    ):
        freshly_embedded, permanently_failed = await _embed_claimed(claimed)
        synced, failed = await _persist_message_embeddings(
            claimed, freshly_embedded, permanently_failed, external_vector_store
        )
        metrics.message_embeddings_synced += synced
        metrics.message_embeddings_failed += failed
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


async def record_pending_embeddings_backlog() -> None:
    """Set the pending-embeddings backlog gauge to the current count of
    MessageEmbedding rows awaiting a vector (sync_state='pending')."""
    # region ai
    # Called from ``ReconcilerScheduler._scheduler_loop``, deliberately NOT from
    # ``run_vector_reconciliation_cycle``: the cycle runs off the queue behind
    # work-unit dedup, so exactly one deriver replica executes it. Driving the gauge
    # from there would leave every other replica exporting a stale value — or, since
    # this metric is zero-initialized, a confident permanent 0 it never measured. The
    # count is a property of the database, not the process, so every replica must
    # refresh it on its own timer for ``max()``/``avg()`` to mean anything.
    #
    # Cost: one COUNT per replica per scheduler interval (~5 min by default).
    # ``ix_message_embeddings_sync_state_last_sync_at`` keeps the scan proportional to
    # the pending backlog, not the whole table — which is not the same as cheap: after
    # an embedding outage the backlog is exactly what is large. Still a small duty
    # cycle, and the cost shrinks as the reconciler drains.
    #
    # Best-effort: a metrics/DB hiccup here must never break the scheduler loop.
    # endregion
    if not settings.METRICS.ENABLED:
        return
    try:
        async with tracked_db("reconciler_pending_count", read_only=True) as db:
            count = await db.scalar(
                select(func.count())
                .select_from(models.MessageEmbedding)
                .where(models.MessageEmbedding.sync_state == "pending")
            )
        prometheus_metrics.set_message_embeddings_pending(count=count or 0)
    except Exception:
        logger.warning(
            "Failed to record pending-embeddings backlog gauge", exc_info=True
        )


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

    # pgvector-only mode: still need to embed pending MessageEmbedding rows
    # (create_messages defers embedding to the reconciler), then clean up.
    if external_vector_store is None:
        while time.monotonic() < deadline:
            embs_work = await _reconcile_message_embeddings_batch(None, metrics)

            if time.monotonic() >= deadline:
                break

            cleanup_work = await _cleanup_pgvector_batch(metrics)

            if not (embs_work or cleanup_work):
                break
        logger.debug("Vector reconciliation cycle completed (pgvector mode)")
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

    logger.debug("Vector reconciliation cycle completed")
    return metrics
