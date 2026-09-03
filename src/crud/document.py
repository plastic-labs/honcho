import asyncio
import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import Any, Literal, cast

from sqlalchemy import delete, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import DBAPIError, IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy.sql.functions import func

from src import models, schemas
from src.config import settings
from src.crud.collection import get_or_create_collection
from src.crud.peer import get_peer, reject_scope_observed
from src.crud.session import get_session
from src.dependencies import tracked_db
from src.embedding_client import EmbeddingTokenLimitError, embedding_client
from src.exceptions import (
    ResourceNotFoundException,
    ValidationException,
    VectorStoreError,
)
from src.utils.filter import apply_filter
from src.vector_store import (
    VectorRecord,
    VectorStore,
    get_external_vector_store,
)

logger = getLogger(__name__)


def get_all_documents(
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
    limit: int | None = None,
) -> Select[tuple[models.Document]]:
    """
    Get all documents in a collection.

    Returns a Select query for pagination support via apaginate().
    Results are ordered by created_at timestamp.

    Args:
        workspace_name: Name of the workspace
        observer: Name of the observing peer
        observed: Name of the observed peer
        filters: Optional filters to apply
        reverse: Whether to reverse the order (oldest first)

    Returns:
        Select query for documents
    """
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
        .where(models.Document.deleted_at.is_(None))  # Exclude soft-deleted
    )

    # Apply additional filters if provided
    stmt = apply_filter(stmt, models.Document, filters)

    # Order by created_at (newest first by default)
    if reverse:
        stmt = stmt.order_by(models.Document.created_at.asc())
    else:
        stmt = stmt.order_by(models.Document.created_at.desc())

    if limit is not None:
        stmt = stmt.limit(limit)

    return stmt


def get_documents_with_filters(
    workspace_name: str,
    *,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
) -> Select[tuple[models.Document]]:
    """
    Get all documents using custom filters.

    Returns a Select query for pagination support via apaginate().
    Results are ordered by created_at timestamp.

    Args:
        workspace_name: Name of the workspace
        filters: Optional filters to apply
        reverse: Whether to reverse the order (oldest first)

    Returns:
        Select query for documents
    """
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.deleted_at.is_(None))  # Exclude soft-deleted
    )

    # Apply additional filters if provided
    stmt = apply_filter(stmt, models.Document, filters)

    # Order by created_at (newest first by default)
    if reverse:
        stmt = stmt.order_by(models.Document.created_at.asc())
    else:
        stmt = stmt.order_by(models.Document.created_at.desc())

    return stmt


async def query_documents_recent(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    limit: int = 10,
    session_name: str | None = None,
) -> Sequence[models.Document]:
    """
    Query most recent documents.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer: Name of the observing peer
        observed: Name of the observed peer
        limit: Maximum number of documents to return
        session_name: Optional session name to filter by

    Returns:
        Sequence of documents ordered by created_at descending
    """
    stmt = select(models.Document).where(
        models.Document.workspace_name == workspace_name,
        models.Document.observer == observer,
        models.Document.observed == observed,
        models.Document.deleted_at.is_(None),
    )

    if session_name is not None:
        stmt = stmt.where(models.Document.session_name == session_name)

    stmt = stmt.order_by(models.Document.created_at.desc()).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()


async def query_documents_most_derived(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    limit: int = 10,
) -> Sequence[models.Document]:
    """
    Query documents sorted by times_derived (most reinforced first).

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer: Name of the observing peer
        observed: Name of the observed peer
        limit: Maximum number of documents to return

    Returns:
        Sequence of documents ordered by times_derived descending,
        ties broken by created_at descending (most recent first)
    """
    stmt = (
        select(models.Document)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
            models.Document.deleted_at.is_(None),
        )
        .order_by(
            models.Document.times_derived.desc(),
            models.Document.created_at.desc(),
            # created_at is the transaction timestamp, so documents created in
            # the same batch share it -- id keeps the order deterministic.
            models.Document.id,
        )
        .limit(limit)
    )

    result = await db.execute(stmt)
    return result.scalars().all()


def _uses_pgvector() -> bool:
    """Check whether queries should go through pgvector (DB-only) path."""
    return (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )


# Shared by is_rejected_duplicate and create_documents candidate resolution.
_SEMANTIC_DUP_MAX_DISTANCE = 0.05
_SEMANTIC_DUP_TOP_K = 1
_SEMANTIC_CANDIDATE_CONCURRENCY = 8


def _semantic_dup_filters(doc: schemas.DocumentCreate) -> dict[str, Any] | None:
    """Merge scope for semantic dedup: never across levels, never across
    sessions for explicit documents. None when the document has no valid
    merge partner (session-less explicit)."""
    filters: dict[str, Any] = {"level": doc.level}
    if doc.level == "explicit":
        if doc.session_name is None:
            return None
        filters["session_name"] = doc.session_name
    return filters


async def query_external_vector_document_ids(
    workspace_name: str,
    observer: str,
    observed: str,
    embedding: list[float],
    top_k: int = 5,
    max_distance: float | None = None,
    filters: dict[str, Any] | None = None,
) -> list[str] | None:
    """Query external vector store for document IDs sorted by similarity.

    No DB session needed — safe to call outside a tracked_db scope.

    Returns:
        Ordered list of document IDs on the external-store path,
        empty list when the external store has no results,
        or None when the pgvector (DB-only) path should be used instead.
    """
    if _uses_pgvector():
        return None

    if top_k <= 0:
        return []

    external_vector_store = get_external_vector_store()
    if external_vector_store is None:
        return []

    namespace = external_vector_store.get_vector_namespace(
        "document", workspace_name, observer, observed
    )

    vector_filters: dict[str, Any] = {}
    if filters:
        for key in ["level", "session_name"]:
            if key in filters:
                vector_filters[key] = filters[key]

    vector_results = await external_vector_store.query(
        namespace,
        embedding,
        top_k=top_k,
        max_distance=max_distance,
        filters=vector_filters if vector_filters else None,
        include_attributes=False,
    )

    if not vector_results:
        return []

    return [result.id for result in vector_results]


async def fetch_documents_by_ids(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    document_ids: list[str],
    filters: dict[str, Any] | None = None,
) -> list[models.Document]:
    """Fetch documents by IDs, preserving input order. DB-only operation."""
    if not document_ids:
        return []

    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
        .where(models.Document.deleted_at.is_(None))
        .where(models.Document.id.in_(document_ids))
    )
    stmt = apply_filter(stmt, models.Document, filters)

    result = await db.execute(stmt)
    documents = {doc.id: doc for doc in result.scalars().all()}

    return [documents[doc_id] for doc_id in document_ids if doc_id in documents]


async def _query_documents_pgvector(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    embedding: list[float],
    filters: dict[str, Any] | None,
    max_distance: float | None,
    top_k: int,
) -> list[models.Document]:
    """pgvector similarity search — pure DB operation."""
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
        .where(models.Document.embedding.isnot(None))
        .where(models.Document.deleted_at.is_(None))
    )

    if max_distance is not None:
        stmt = stmt.where(
            models.Document.embedding.cosine_distance(embedding) <= max_distance
        )

    stmt = apply_filter(stmt, models.Document, filters)
    stmt = stmt.order_by(models.Document.embedding.cosine_distance(embedding)).limit(
        top_k
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def query_documents(
    db: AsyncSession | None,
    workspace_name: str,
    query: str,
    *,
    observer: str,
    observed: str,
    filters: dict[str, Any] | None = None,
    max_distance: float | None = None,
    top_k: int = 5,
    embedding: list[float] | None = None,
) -> Sequence[models.Document]:
    """
    Query documents using semantic similarity.

    When *db* is provided the caller owns the session lifetime.  When *db* is
    ``None`` the function opens (and closes) its own short-lived session so that
    no DB connection is held during external vector-store calls.

    Args:
        db: Database session, or None to let the function manage its own
        workspace_name: Name of the workspace
        query: Search query text
        observer: Name of the observing peer
        observed: Name of the observed peer
        filters: Optional filters to apply at vector store level (supports: level, session_name)
        max_distance: Maximum cosine distance for results
        top_k: Number of results to return
        embedding: Optional pre-computed embedding for the query (avoids extra API call if possible)

    Returns:
        Sequence of matching documents
    """
    if top_k <= 0:
        return []

    # Use provided embedding or generate one
    if embedding is None:
        try:
            embedding = await embedding_client.embed(query)
        except EmbeddingTokenLimitError as e:
            raise ValidationException(
                "Query exceeds maximum token limit of "
                + f"{settings.EMBEDDING.MAX_INPUT_TOKENS}."
            ) from e

    if _uses_pgvector():
        # pgvector path — pure DB, open a short session if none provided
        if db is not None:
            return await _query_documents_pgvector(
                db,
                workspace_name,
                observer,
                observed,
                embedding,
                filters,
                max_distance,
                top_k,
            )
        async with tracked_db("query_documents.pgvector", read_only=True) as managed_db:
            docs = await _query_documents_pgvector(
                managed_db,
                workspace_name,
                observer,
                observed,
                embedding,
                filters,
                max_distance,
                top_k,
            )
            for doc in docs:
                managed_db.expunge(doc)
            return docs

    # External vector store — network call first, DB only for the ID fetch
    document_ids = await query_external_vector_document_ids(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        embedding=embedding,
        top_k=top_k,
        max_distance=max_distance,
        filters=filters,
    )

    if not document_ids:
        return []

    if db is not None:
        return await fetch_documents_by_ids(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            document_ids=document_ids,
            filters=filters,
        )
    async with tracked_db("query_documents.fetch", read_only=True) as managed_db:
        docs = await fetch_documents_by_ids(
            db=managed_db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            document_ids=document_ids,
            filters=filters,
        )
        for doc in docs:
            managed_db.expunge(doc)
        return docs


def _normalize_content(content: str) -> str:
    """Normalize document content for exact-match deduplication.

    Content is compared after trimming surrounding whitespace and lowercasing

    The SQL filter in ``create_documents`` must stay in sync with this:
    ``lower(regexp_replace(content, '^\\s+|\\s+$', '', 'g'))``. Postgres'
    ``trim()`` only strips spaces, so a regex is used to match Python's
    ``str.strip()`` across all whitespace.
    """
    return content.strip().lower()


def _dedup_key(
    content: str, level: str, session_name: str | None
) -> tuple[str, str, str | None]:
    """Build the exact-match dedup key for a document.

    Dedup never crosses levels: a same-content document at a different level is
    a different kind of record (an explicit fact is not interchangeable with a
    deductive conclusion that happens to share its text).

    For **explicit** documents dedup additionally never crosses sessions.
    Explicit documents are session-pure records of what was derived from that
    session's messages — the Scopes copy-by-session model depends on this — so
    a repeat of the same fact in a different session must produce a new
    document in that session rather than reinforce another session's row.
    Derived levels (deductive/inductive/contradiction) are consolidations and
    may still dedup across sessions.
    """
    return (
        _normalize_content(content),
        level,
        session_name if level == "explicit" else None,
    )


@dataclass(frozen=True, slots=True)
class _DocumentRowOp:
    kind: Literal["reinforce", "replace"]
    document_id: str
    incoming_times_derived: int = 1
    # When a reinforce skipped insert and the locked target is gone/deleted,
    # insert this document instead of dropping it.
    fallback_document: schemas.DocumentCreate | None = None


@dataclass
class CreateDocumentsResult:
    created_documents: list[schemas.DocumentCreate] = field(default_factory=list)
    exact_dup_in_batch_count: int = 0
    exact_dup_existing_count: int = 0
    semantic_dup_rejected_count: int = 0
    semantic_dup_replaced_count: int = 0


async def create_documents(
    db: AsyncSession,
    documents: list[schemas.DocumentCreate],
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    deduplicate: bool = False,
) -> CreateDocumentsResult:
    """
    Create multiple documents with optional duplicate detection.

    The ``deduplicate`` flag additionally enables semantic (cosine-similarity)
    dedup via ``is_rejected_duplicate`` for documents that survive the exact
    deduplication check.

    Args:
        db: Database session
        documents: List of document creation schemas
        workspace_name: Name of the workspace
        observer: Name of the observing peer
        observed: Name of the observed peer
        deduplicate: Enable semantic duplicate detection

    Returns:
        List of DocumentCreate schemas that were actually inserted (excludes
        duplicates and failures).
    """
    honcho_documents: list[models.Document] = []
    accepted_documents: list[schemas.DocumentCreate] = []
    # Store (document_model, embedding) pairs - IDs aren't available until after commit
    docs_with_embeddings: list[tuple[models.Document, list[float]]] = []

    # Resolve external-store dup candidates before the first DB statement.
    # None = pgvector in-place fallback; [] = skip semantic (no external I/O under db).
    semantic_candidates: list[list[str] | None] = [None] * len(documents)
    if deduplicate and not _uses_pgvector():
        resolve_sem = asyncio.Semaphore(_SEMANTIC_CANDIDATE_CONCURRENCY)

        async def _resolve_candidates(index: int, doc: schemas.DocumentCreate) -> None:
            filters = _semantic_dup_filters(doc)
            if filters is None or not doc.embedding:
                semantic_candidates[index] = []
                return
            async with resolve_sem:
                try:
                    ids = await query_external_vector_document_ids(
                        workspace_name=workspace_name,
                        observer=observer,
                        observed=observed,
                        embedding=doc.embedding,
                        top_k=_SEMANTIC_DUP_TOP_K,
                        max_distance=_SEMANTIC_DUP_MAX_DISTANCE,
                        filters=filters,
                    )
                except Exception:
                    logger.exception(
                        "External semantic-candidate resolve failed for %s/%s/%s",
                        workspace_name,
                        observer,
                        observed,
                    )
                    semantic_candidates[index] = []
                    return
                semantic_candidates[index] = ids or []

        await asyncio.gather(
            *(_resolve_candidates(i, doc) for i, doc in enumerate(documents))
        )

    # exact-content dedup (independent of `deduplicate`): pre-fetch
    # existing live documents whose normalized content matches anything in this
    # batch, scoped to (workspace, observer, observed). The SQL normalization must
    # mirror _normalize_content. Matching is further scoped per-document by
    # level (always) and session (for explicit documents) via _dedup_key.
    batch_normalized: set[str] = {_normalize_content(d.content) for d in documents}
    existing_by_key: dict[tuple[str, str, str | None], models.Document] = {}
    if batch_normalized:
        # The `normalized_content_sql.in_(...)` filter below narrows to the
        # (workspace, observer, observed) partition via the single-column indexes,
        # then evaluates lower(regexp_replace(...)) per row.
        # TODO: add a partial expression index matching
        # this filter exactly
        #     CREATE INDEX ix_documents_normalized_content
        #     ON documents (
        #         workspace_name,
        #         observer,
        #         observed,
        #         (lower(regexp_replace(content, '^\s+|\s+$', '', 'g')))
        #     )
        #     WHERE deleted_at IS NULL;
        normalized_content_sql = func.lower(
            func.regexp_replace(models.Document.content, r"^\s+|\s+$", "", "g")
        )
        existing_result = await db.execute(
            select(models.Document).where(
                models.Document.workspace_name == workspace_name,
                models.Document.observer == observer,
                models.Document.observed == observed,
                models.Document.deleted_at.is_(None),
                normalized_content_sql.in_(batch_normalized),
            )
        )
        for existing_doc in existing_result.scalars():
            # If multiple historical rows share a dedup key, reinforcing
            # one is sufficient; keep the first.
            existing_by_key.setdefault(
                _dedup_key(
                    existing_doc.content,
                    existing_doc.level,
                    existing_doc.session_name,
                ),
                existing_doc,
            )

    # Tracks dedup keys already accepted from this batch so exact
    # duplicates within a single inference call collapse to one document.
    seen_in_batch: set[tuple[str, str, str | None]] = set()
    row_ops: list[_DocumentRowOp] = []
    pending_times_derived: dict[str, int] = {}

    exact_dup_existing_count = 0
    exact_dup_in_batch_count = 0
    semantic_dup_rejected_count = 0
    semantic_dup_replaced_count = 0
    for index, doc in enumerate(documents):
        try:
            # Session-purity invariant: an explicit document must always carry
            # the session it was derived from. Refuse to write session-less
            # explicit documents rather than silently minting global explicit
            # memory (the Scopes copy-by-session model depends on explicit
            # documents staying session-pure).
            if doc.level == "explicit" and doc.session_name is None:
                logger.error(
                    "Refusing to create explicit document without session_name in %s/%s/%s (session-purity invariant): %r",
                    workspace_name,
                    observer,
                    observed,
                    doc.content[:80],
                )
                continue

            dedup_key = _dedup_key(doc.content, doc.level, doc.session_name)

            # Exact-match dedup, always on:
            # 1) collapse exact duplicates within this batch (drop silently).
            if dedup_key in seen_in_batch:
                exact_dup_in_batch_count += 1
                continue
            seen_in_batch.add(dedup_key)

            # 2) drop exact duplicates of an existing live document, recording
            #    the re-derivation as reinforcement on the existing row.
            existing_match = existing_by_key.get(dedup_key)
            if existing_match is not None:
                current_td = pending_times_derived.get(
                    existing_match.id, existing_match.times_derived
                )
                pending_times_derived[existing_match.id] = max(
                    current_td + 1, doc.times_derived
                )
                row_ops.append(
                    _DocumentRowOp(
                        "reinforce",
                        existing_match.id,
                        doc.times_derived,
                        fallback_document=doc,
                    )
                )
                exact_dup_existing_count += 1
                continue

            if deduplicate:
                duplicate_result, existing_dup = await _semantic_dup_decision(
                    db,
                    doc,
                    workspace_name,
                    observer=observer,
                    observed=observed,
                    candidate_document_ids=semantic_candidates[index],
                )
                if (
                    duplicate_result is SemanticRejectionResult.REPLACED_EXISTING
                    and existing_dup is not None
                ):
                    current_td = pending_times_derived.get(
                        existing_dup.id, existing_dup.times_derived
                    )
                    doc.times_derived = max(doc.times_derived, current_td + 1)
                    pending_times_derived[existing_dup.id] = doc.times_derived
                    row_ops.append(_DocumentRowOp("replace", existing_dup.id))
                    semantic_dup_replaced_count += 1
                elif (
                    duplicate_result is SemanticRejectionResult.REJECTED
                    and existing_dup is not None
                ):
                    current_td = pending_times_derived.get(
                        existing_dup.id, existing_dup.times_derived
                    )
                    pending_times_derived[existing_dup.id] = max(
                        current_td + 1, doc.times_derived
                    )
                    row_ops.append(
                        _DocumentRowOp(
                            "reinforce",
                            existing_dup.id,
                            doc.times_derived,
                            fallback_document=doc,
                        )
                    )
                    semantic_dup_rejected_count += 1
                    continue

            new_doc = _document_model_from_create(
                doc, workspace_name=workspace_name, observer=observer, observed=observed
            )
            honcho_documents.append(new_doc)
            accepted_documents.append(doc)
            if doc.embedding:
                docs_with_embeddings.append((new_doc, doc.embedding))

        except IntegrityError as e:
            await db.rollback()
            raise ValidationException(
                "Failed to create documents due to integrity constraint violation"
            ) from e
        except SQLAlchemyError:
            # Dead transaction: continuing would cascade PendingRollbackErrors.
            await db.rollback()
            raise
        except Exception as e:
            # Per-document failures (bad content, metadata, token overflow).
            logger.error(
                f"Error adding new document to {workspace_name}/{doc.session_name}/{observer}/{observed}: {e}"
            )
            continue

    try:
        fallback_docs = await _apply_document_row_updates(
            db,
            row_ops,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        for fallback_doc in fallback_docs:
            new_doc = _document_model_from_create(
                fallback_doc,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
            )
            honcho_documents.append(new_doc)
            accepted_documents.append(fallback_doc)
            if fallback_doc.embedding:
                docs_with_embeddings.append((new_doc, fallback_doc.embedding))
        db.add_all(honcho_documents)
        # NOTE
        # If the process crashes after this commit but before vector upsert completes,
        # documents will be left in sync_state='pending' with NULL embeddings.
        # The reconciliation job will automatically re-embed and sync these documents,
        await db.commit()

        # Store embeddings in external vector store after documents are committed (IDs now available)
        if docs_with_embeddings:
            doc_ids = [doc.id for doc, _ in docs_with_embeddings]
            external_vector_store = get_external_vector_store()

            # If no external vector store (pgvector mode), mark as synced immediately
            if external_vector_store is None:
                await db.execute(
                    update(models.Document)
                    .where(models.Document.id.in_(doc_ids))
                    .values(
                        sync_state="synced",
                        last_sync_at=func.now(),
                        sync_attempts=0,
                    )
                )
                await db.commit()
            else:
                # External vector store - upsert and track sync state
                namespace = external_vector_store.get_vector_namespace(
                    "document",
                    workspace_name,
                    observer,
                    observed,
                )

                # Build vector records with metadata for filtering
                vector_records: list[VectorRecord] = []
                for doc, embedding in docs_with_embeddings:
                    vector_records.append(
                        VectorRecord(
                            id=doc.id,
                            embedding=embedding,
                            metadata={
                                "workspace_name": workspace_name,
                                "observer": observer,
                                "observed": observed,
                                "session_name": doc.session_name,
                                "level": doc.level,
                            },
                        )
                    )

                # Upsert to external vector store and update sync state
                try:
                    await external_vector_store.upsert_many(namespace, vector_records)
                    # Success: mark as synced
                    await db.execute(
                        update(models.Document)
                        .where(models.Document.id.in_(doc_ids))
                        .values(
                            sync_state="synced",
                            last_sync_at=func.now(),
                            sync_attempts=0,
                        )
                    )
                    await db.commit()

                except VectorStoreError:
                    # Vector store unavailable - increment sync_attempts for reconciliation
                    logger.warning("Vector store unavailable; leaving docs unsynced")
                    await db.execute(
                        update(models.Document)
                        .where(models.Document.id.in_(doc_ids))
                        .values(
                            sync_attempts=models.Document.sync_attempts + 1,
                            last_sync_at=func.now(),
                        )
                    )
                    await db.commit()

                except Exception:
                    logger.exception("Unexpected error upserting vectors")
                    await db.execute(
                        update(models.Document)
                        .where(models.Document.id.in_(doc_ids))
                        .values(
                            sync_attempts=models.Document.sync_attempts + 1,
                            last_sync_at=func.now(),
                        )
                    )
                    await db.commit()

    except IntegrityError as e:
        await db.rollback()
        raise ValidationException(
            "Failed to create documents due to integrity constraint violation"
        ) from e
    except DBAPIError:
        # Leave the session clean for callers that own it (e.g. a deadlock
        # at the final commit); the queue layer classifies and retries.
        await db.rollback()
        raise

    return CreateDocumentsResult(
        created_documents=accepted_documents,
        exact_dup_existing_count=exact_dup_existing_count,
        exact_dup_in_batch_count=exact_dup_in_batch_count,
        semantic_dup_rejected_count=semantic_dup_rejected_count,
        semantic_dup_replaced_count=semantic_dup_replaced_count,
    )


async def delete_document(
    db: AsyncSession,
    workspace_name: str,
    document_id: str,
    *,
    observer: str,
    observed: str,
    session_name: str | None = None,
) -> None:
    """
    Soft-delete a document by ID.

    Sets deleted_at timestamp to mark the document as deleted. The reconciliation
    job handles vector store cleanup and hard deletion from the database.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        document_id: ID of the document to delete
        observer: Name of the observing peer (for authorization)
        observed: Name of the observed peer (for authorization)
        session_name: Optional session name to verify document belongs to session

    Raises:
        ResourceNotFoundException: If document not found or doesn't match criteria
    """
    conditions = [
        models.Document.id == document_id,
        models.Document.workspace_name == workspace_name,
        models.Document.observer == observer,
        models.Document.observed == observed,
        models.Document.deleted_at.is_(None),
    ]
    if session_name is not None:
        conditions.append(models.Document.session_name == session_name)

    update_stmt = (
        update(models.Document).where(*conditions).values(deleted_at=func.now())
    )
    result = cast(CursorResult[Any], await db.execute(update_stmt))

    if result.rowcount == 0:
        raise ResourceNotFoundException(
            f"Document {document_id} not found or does not belong to the specified collection/session"
        )

    await db.commit()


async def delete_documents(
    db: AsyncSession,
    workspace_name: str,
    document_ids: Sequence[str],
    *,
    observer: str,
    observed: str,
    session_name: str | None = None,
) -> list[tuple[str, str]]:
    """
    Soft-delete multiple documents in a single UPDATE ... RETURNING statement.

    Returns (id, level) tuples for rows that actually got deleted — i.e. rows
    that matched the workspace/observer/observed filter and were not already
    soft-deleted. IDs that didn't match are silently skipped; callers can diff
    the returned ids against the input to detect misses.
    """
    if not document_ids:
        return []

    conditions = [
        models.Document.id.in_(document_ids),
        models.Document.workspace_name == workspace_name,
        models.Document.observer == observer,
        models.Document.observed == observed,
        models.Document.deleted_at.is_(None),
    ]
    if session_name is not None:
        conditions.append(models.Document.session_name == session_name)

    stmt = (
        update(models.Document)
        .where(*conditions)
        .values(deleted_at=func.now())
        .returning(models.Document.id, models.Document.level)
    )
    result = await db.execute(stmt)
    rows = result.all()
    await db.commit()
    return [(row.id, row.level) for row in rows]


async def delete_document_by_id(
    db: AsyncSession,
    workspace_name: str,
    document_id: str,
) -> None:
    """
    Soft-delete a document by ID and workspace.

    Sets deleted_at timestamp to mark the document as deleted. The reconciliation
    job handles vector store cleanup and hard deletion from the database.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        document_id: ID of the document to delete

    Raises:
        ResourceNotFoundException: If document not found or doesn't belong to the workspace
    """
    update_stmt = (
        update(models.Document)
        .where(
            models.Document.id == document_id,
            models.Document.workspace_name == workspace_name,
            models.Document.deleted_at.is_(None),
        )
        .values(deleted_at=func.now())
    )
    result = cast(CursorResult[Any], await db.execute(update_stmt))

    if result.rowcount == 0:
        raise ResourceNotFoundException(
            f"Document {document_id} not found or does not belong to workspace {workspace_name}"
        )

    await db.commit()


async def create_observations(
    db: AsyncSession,
    observations: Sequence[schemas.ConclusionCreate],
    workspace_name: str,
) -> list[models.Document]:
    """
    Create multiple observations (documents) from user input.

    This function validates all referenced resources, generates embeddings
    in batch, and creates the documents.

    Args:
        db: Database session
        observations: List of observation creation schemas
        workspace_name: Name of the workspace

    Returns:
        List of created Document objects

    Raises:
        ResourceNotFoundException: If any session or peer is not found
        ValidationException: If embedding generation fails or integrity constraint is violated
    """
    if not observations:
        return []

    # Collect unique sessions and peer pairs to validate
    sessions_to_validate: set[str] = set()
    peers_to_validate: set[str] = set()
    collection_pairs: set[tuple[str, str]] = set()

    for obs in observations:
        if obs.session_id is not None:
            sessions_to_validate.add(obs.session_id)
        peers_to_validate.add(obs.observer_id)
        peers_to_validate.add(obs.observed_id)
        collection_pairs.add((obs.observer_id, obs.observed_id))

    # Validate all sessions exist
    for session_name in sessions_to_validate:
        await get_session(db, session_name, workspace_name)

    # Validate all peers exist
    for peer_name in peers_to_validate:
        await get_peer(db, workspace_name, peer_name)

    # A scope may be an *observer* — that is how scoped conclusions are stored —
    # but it must never be *observed*: scope peers carry observe_me=false and no
    # representation is ever formed of one. Without this, a conclusion about a
    # scope persists and a (observer, scope) collection is created for it.
    #
    # The strict variant because this is an observed position, though defence in
    # depth rather than the active guard: the loop above resolves every peer, so a
    # reserved name that does not exist yet already 404s before reaching here. If
    # that validation ever stops covering observed_id, this still refuses the
    # pre-seeding case instead of persisting a conclusion that a later-created
    # scope would retroactively own.
    await reject_scope_observed(
        db,
        workspace_name,
        {obs.observed_id for obs in observations},
        action="No conclusion is ever formed about a scope.",
    )

    # Get or create all collections
    for observer, observed in collection_pairs:
        await get_or_create_collection(
            db, workspace_name, observer=observer, observed=observed
        )

    # Generate embeddings in batch
    contents = [obs.content for obs in observations]
    try:
        embeddings = await embedding_client.simple_batch_embed(
            contents, on_oversize="truncate"
        )
    except EmbeddingTokenLimitError as e:
        raise ValidationException(str(e)) from e

    # Create document objects and track embeddings for vector store
    honcho_documents: list[models.Document] = []
    # Group observations by collection (observer, observed) for vector store upserts
    collection_embeddings: dict[
        tuple[str, str], list[tuple[models.Document, list[float]]]
    ] = {}

    # Determine if we need to persist embeddings to postgres
    # True when: TYPE=pgvector OR still migrating (dual-write to both stores)
    store_embeddings_in_postgres = (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )

    for obs, embedding in zip(observations, embeddings, strict=True):
        if store_embeddings_in_postgres:
            doc = models.Document(
                workspace_name=workspace_name,
                observer=obs.observer_id,
                observed=obs.observed_id,
                content=obs.content,
                level="explicit",  # Manually created observations are always explicit
                times_derived=1,
                internal_metadata={},  # No message_ids since not derived from messages
                session_name=obs.session_id,
                embedding=embedding,
            )
        else:
            doc = models.Document(
                workspace_name=workspace_name,
                observer=obs.observer_id,
                observed=obs.observed_id,
                content=obs.content,
                level="explicit",  # Manually created observations are always explicit
                times_derived=1,
                internal_metadata={},  # No message_ids since not derived from messages
                session_name=obs.session_id,
            )
        doc.sync_state = "pending"
        honcho_documents.append(doc)

        # Track embedding for vector store (grouped by collection)
        collection_key = (obs.observer_id, obs.observed_id)
        if collection_key not in collection_embeddings:
            collection_embeddings[collection_key] = []
        collection_embeddings[collection_key].append((doc, embedding))

    try:
        db.add_all(honcho_documents)
        await db.commit()
        # Refresh all documents to get generated IDs and timestamps
        for doc in honcho_documents:
            await db.refresh(doc)

        # Store embeddings in external vector store after documents are committed (IDs now available)
        external_vector_store = get_external_vector_store()
        all_doc_ids = [doc.id for doc in honcho_documents]

        # If no external vector store (pgvector mode), mark as synced immediately
        if external_vector_store is None:
            await db.execute(
                update(models.Document)
                .where(models.Document.id.in_(all_doc_ids))
                .values(
                    sync_state="synced",
                    last_sync_at=func.now(),
                    sync_attempts=0,
                )
            )
            await db.commit()
        else:
            # External vector store - upsert each collection's embeddings
            for (
                observer,
                observed,
            ), docs_with_embeddings in collection_embeddings.items():
                namespace = external_vector_store.get_vector_namespace(
                    "document",
                    workspace_name,
                    observer,
                    observed,
                )

                # Build vector records with metadata for filtering
                vector_records: list[VectorRecord] = []
                doc_ids: list[str] = []
                for doc, embedding in docs_with_embeddings:
                    doc_ids.append(doc.id)
                    vector_records.append(
                        VectorRecord(
                            id=doc.id,
                            embedding=embedding,
                            metadata={
                                "workspace_name": workspace_name,
                                "observer": observer,
                                "observed": observed,
                                "session_name": doc.session_name,
                                "level": doc.level,
                            },
                        )
                    )

                # Upsert to external vector store and update sync state
                try:
                    await external_vector_store.upsert_many(namespace, vector_records)
                    # Success: mark as synced
                    await db.execute(
                        update(models.Document)
                        .where(models.Document.id.in_(doc_ids))
                        .values(
                            sync_state="synced",
                            last_sync_at=func.now(),
                            sync_attempts=0,
                        )
                    )
                    await db.commit()

                except VectorStoreError:
                    logger.warning(
                        "Vector store unavailable for namespace %s; leaving observations unsynced",
                        namespace,
                    )
                    await db.execute(
                        update(models.Document)
                        .where(models.Document.id.in_(doc_ids))
                        .values(
                            sync_attempts=models.Document.sync_attempts + 1,
                            last_sync_at=func.now(),
                        )
                    )
                    await db.commit()

                except Exception:
                    logger.exception(
                        "Unexpected error upserting vectors for %s", namespace
                    )
                    await db.execute(
                        update(models.Document)
                        .where(models.Document.id.in_(doc_ids))
                        .values(
                            sync_attempts=models.Document.sync_attempts + 1,
                            last_sync_at=func.now(),
                        )
                    )
                    await db.commit()

    except IntegrityError as e:
        await db.rollback()
        raise ValidationException(
            "Failed to create observations due to integrity constraint violation"
        ) from e

    logger.debug(
        "Created %d observations in workspace %s",
        len(honcho_documents),
        workspace_name,
    )
    return honcho_documents


def _document_model_from_create(
    doc: schemas.DocumentCreate,
    *,
    workspace_name: str,
    observer: str,
    observed: str,
) -> models.Document:
    metadata_dict = doc.metadata.model_dump(exclude_none=True)
    store_embeddings_in_postgres = (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )
    if store_embeddings_in_postgres and doc.embedding:
        new_doc = models.Document(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content=doc.content,
            level=doc.level,
            times_derived=doc.times_derived,
            internal_metadata=metadata_dict,
            session_name=doc.session_name,
            embedding=doc.embedding,
            source_ids=doc.source_ids,
        )
    else:
        new_doc = models.Document(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content=doc.content,
            level=doc.level,
            times_derived=doc.times_derived,
            internal_metadata=metadata_dict,
            session_name=doc.session_name,
            source_ids=doc.source_ids,
        )
    if doc.embedding:
        new_doc.sync_state = "pending"
    return new_doc


async def _apply_document_row_updates(
    db: AsyncSession,
    ops: list[_DocumentRowOp],
    *,
    workspace_name: str,
    observer: str,
    observed: str,
) -> list[schemas.DocumentCreate]:
    """Lock target rows by id, apply ops, return fallbacks for vanished targets."""
    if not ops:
        return []
    # Deadlock fix: lock in id order (IN-clause order is ignored).
    ids = sorted({op.document_id for op in ops})
    result = await db.execute(
        select(models.Document)
        .where(
            models.Document.id.in_(ids),
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
        )
        .order_by(models.Document.id)
        .with_for_update()
        # Reload identity-map rows so the Python max() sees concurrent increments.
        .execution_options(populate_existing=True)
    )
    locked = {doc.id: doc for doc in result.scalars()}
    now = datetime.datetime.now(datetime.UTC)
    fallbacks: list[schemas.DocumentCreate] = []
    stale_at_lock = {
        op.document_id
        for op in ops
        if (locked_row := locked.get(op.document_id)) is None
        or locked_row.deleted_at is not None
    }
    for op in ops:
        row = locked.get(op.document_id)
        if op.kind == "replace":
            if row is not None and row.deleted_at is None:
                row.deleted_at = now
            continue
        # reinforce
        if op.document_id in stale_at_lock:
            if op.fallback_document is not None:
                fallbacks.append(op.fallback_document)
            continue
        if row is None or row.deleted_at is not None:
            # An earlier op in this batch replaced this row.
            continue
        row.times_derived = max(row.times_derived + 1, op.incoming_times_derived)
    await db.flush()
    return fallbacks


class SemanticRejectionResult(Enum):
    NOT_DUPLICATE = 0
    REPLACED_EXISTING = 1
    REJECTED = 2


async def _semantic_dup_decision(
    db: AsyncSession,
    doc: schemas.DocumentCreate,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    candidate_document_ids: list[str] | None = None,
) -> tuple[SemanticRejectionResult, models.Document | None]:
    """Classify a semantic duplicate without writing."""
    filters = _semantic_dup_filters(doc)
    if filters is None:
        return SemanticRejectionResult.NOT_DUPLICATE, None

    if candidate_document_ids is not None:
        similar_docs: Sequence[models.Document] = await fetch_documents_by_ids(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            document_ids=candidate_document_ids,
            filters=filters,
        )
    elif _uses_pgvector():
        if not doc.embedding:
            # Match external-store path: never embed under an open session.
            return SemanticRejectionResult.NOT_DUPLICATE, None
        similar_docs = await query_documents(
            db=db,
            workspace_name=workspace_name,
            query=doc.content,
            observer=observer,
            observed=observed,
            filters=filters,
            max_distance=_SEMANTIC_DUP_MAX_DISTANCE,
            top_k=_SEMANTIC_DUP_TOP_K,
            embedding=doc.embedding,
        )
    else:
        return SemanticRejectionResult.NOT_DUPLICATE, None

    if not similar_docs:
        return SemanticRejectionResult.NOT_DUPLICATE, None

    existing_doc = similar_docs[0]
    tokens_new = set(embedding_client.encoding.encode(doc.content))
    tokens_existing = set(embedding_client.encoding.encode(existing_doc.content))
    unique_new = len(tokens_new - tokens_existing)
    unique_existing = len(tokens_existing - tokens_new)
    score_new = len(tokens_new) + (unique_new * 10)
    score_existing = len(tokens_existing) + (unique_existing * 10)
    if score_new >= score_existing:
        return SemanticRejectionResult.REPLACED_EXISTING, existing_doc
    return SemanticRejectionResult.REJECTED, existing_doc


async def is_rejected_duplicate(
    db: AsyncSession,
    doc: schemas.DocumentCreate,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    candidate_document_ids: list[str] | None = None,
) -> SemanticRejectionResult:
    """Classify a semantic duplicate and apply the corresponding row write."""
    result, existing_doc = await _semantic_dup_decision(
        db,
        doc,
        workspace_name,
        observer=observer,
        observed=observed,
        candidate_document_ids=candidate_document_ids,
    )
    if existing_doc is None:
        return result
    if result is SemanticRejectionResult.REPLACED_EXISTING:
        logger.debug(
            "[DUPLICATE DETECTION] Deleting existing in favor of new. new=%r, existing=%r.",
            doc.content,
            existing_doc.content,
        )
        doc.times_derived = max(doc.times_derived, existing_doc.times_derived + 1)
        existing_doc.deleted_at = datetime.datetime.now(datetime.UTC)
        await db.flush()
        return result
    existing_doc.times_derived = func.greatest(
        models.Document.times_derived + 1,
        doc.times_derived,
    )
    await db.flush()
    logger.debug(
        "[DUPLICATE DETECTION] Rejecting new in favor of existing. new=%r, existing=%r.",
        doc.content,
        existing_doc.content,
    )
    return result


async def cleanup_soft_deleted_documents(
    db: AsyncSession,
    external_vector_store: VectorStore,
    batch_size: int = 100,
    older_than_minutes: int = 5,
) -> int:
    """
    Cleanup soft-deleted documents by removing their vectors and database records.

    This function implements a two-phase cleanup process for documents that have been
    soft-deleted (deleted_at is not NULL)

    Args:
        db: Database session for executing queries
        external_vector_store: External vector store instance for deleting vectors
        batch_size: Maximum number of documents to process per call (default 100)
        older_than_minutes: Only process documents soft-deleted more than this many
            minutes ago (default 5).

    Returns:
        Count of documents cleaned up (only those where vector deletion succeeded).
    """
    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
        minutes=older_than_minutes
    )

    # Find soft-deleted documents ready for cleanup
    # Use FOR UPDATE SKIP LOCKED to prevent multiple deriver instances from
    # processing the same documents simultaneously
    stmt = (
        select(models.Document)
        .where(models.Document.deleted_at.is_not(None))
        .where(models.Document.deleted_at < cutoff)
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )
    result = await db.execute(stmt)
    documents = list(result.scalars().all())

    if not documents:
        return 0

    # Group by namespace for batch vector deletion
    by_namespace: dict[str, list[str]] = {}
    for doc in documents:
        namespace = external_vector_store.get_vector_namespace(
            "document",
            doc.workspace_name,
            doc.observer,
            doc.observed,
        )
        by_namespace.setdefault(namespace, []).append(doc.id)

    # Delete from external vector store (per namespace) and track successful deletions
    successfully_deleted_ids: set[str] = set()
    for namespace, ids in by_namespace.items():
        try:
            await external_vector_store.delete_many(namespace, ids)
            # Only add to successfully_deleted_ids if vector deletion succeeded
            successfully_deleted_ids.update(ids)
        except Exception as e:
            # Log but continue - vectors may already be deleted or namespace may not exist
            logger.warning(f"Failed to delete vectors from {namespace}: {e}")

    # Only hard delete documents where vector deletion succeeded
    if successfully_deleted_ids:
        await db.execute(
            delete(models.Document).where(
                models.Document.id.in_(successfully_deleted_ids)
            )
        )
        await db.commit()
        logger.debug(
            f"Cleaned up {len(successfully_deleted_ids)} soft-deleted documents"
        )
        return len(successfully_deleted_ids)

    # No documents were successfully deleted from vector store
    # Release FOR UPDATE locks by rolling back the transaction
    await db.rollback()
    return 0


# =============================================================================
# Tree Traversal Functions - For reasoning chain navigation
# =============================================================================


async def get_documents_by_ids(
    db: AsyncSession,
    workspace_name: str,
    document_ids: list[str],
) -> Sequence[models.Document]:
    """
    Get multiple documents by their IDs.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        document_ids: List of document IDs to retrieve

    Returns:
        Sequence of documents found (may be fewer than requested if some IDs don't exist)
    """
    if not document_ids:
        return []
    stmt = select(models.Document).where(
        models.Document.workspace_name == workspace_name,
        models.Document.id.in_(document_ids),
        models.Document.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_child_observations(
    db: AsyncSession,
    workspace_name: str,
    parent_id: str,
    *,
    observer: str | None = None,
    observed: str | None = None,
) -> Sequence[models.Document]:
    """
    Get all observations that have this document as a source/premise.

    Useful for traversing the reasoning tree upward (source -> derived observations).
    Uses GIN index on source_ids for efficient lookups.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        parent_id: Document ID to find children of
        observer: Optional filter by observer
        observed: Optional filter by observed

    Returns:
        Sequence of documents that reference this document as a source
    """
    # Find documents where source_ids contains the parent_id
    stmt = select(models.Document).where(
        models.Document.workspace_name == workspace_name,
        models.Document.source_ids.contains([parent_id]),
        models.Document.deleted_at.is_(None),
    )
    if observer:
        stmt = stmt.where(models.Document.observer == observer)
    if observed:
        stmt = stmt.where(models.Document.observed == observed)

    result = await db.execute(stmt)
    return result.scalars().all()
