import datetime
from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy.sql.functions import func
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from src import models, schemas
from src.config import settings
from src.crud.collection import get_or_create_collection
from src.crud.peer import get_peer
from src.crud.session import get_session
from src.embedding_client import embedding_client
from src.exceptions import ResourceNotFoundException, ValidationException
from src.utils.filter import apply_filter
from src.vector_store import VectorRecord, VectorStore, get_vector_store

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


async def query_documents(
    db: AsyncSession,
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

    Args:
        db: Database session
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
    # Use provided embedding or generate one
    if embedding is None:
        try:
            embedding = await embedding_client.embed(query)
        except ValueError as e:
            raise ValidationException(
                f"Query exceeds maximum token limit of {settings.MAX_EMBEDDING_TOKENS}."
            ) from e

    # Get vector store and namespace for this collection
    vector_store = get_vector_store()
    namespace = vector_store.get_document_namespace(workspace_name, observer, observed)

    # Build vector store filters
    # Convert filter dict to vector store format (handles level, session_name, etc.)
    vector_filters: dict[str, Any] = {}
    if filters:
        # Direct pass-through for simple equality filters
        # The filters dict can contain: level, session_name, or other document fields
        # We can push level and session_name to vector store since they're in metadata
        for key in ["level", "session_name"]:
            if key in filters:
                vector_filters[key] = filters[key]

    # Query vector store for similar documents with filters applied
    vector_results = await vector_store.query(
        namespace,
        embedding,
        top_k=top_k,
        max_distance=max_distance,
        filters=vector_filters if vector_filters else None,
    )

    if not vector_results:
        return []

    # Get document IDs from vector results (vector ID = document ID for documents)
    document_ids = [result.id for result in vector_results]

    # Fetch documents from database
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
        .where(models.Document.deleted_at.is_(None))
        .where(models.Document.id.in_(document_ids))
    )
    # Re-apply all filters at the database layer to catch any constraints
    # that aren't supported by the vector store metadata.
    stmt = apply_filter(stmt, models.Document, filters)

    result = await db.execute(stmt)
    documents = {doc.id: doc for doc in result.scalars().all()}

    # Return documents in order of similarity (preserving vector store order)
    ordered_docs: list[models.Document] = []
    for vr in vector_results:
        if vr.id in documents:
            ordered_docs.append(documents[vr.id])

    return ordered_docs


async def create_documents(
    db: AsyncSession,
    documents: list[schemas.DocumentCreate],
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    deduplicate: bool = False,
) -> int:
    """
    Create multiple documents with optional duplicate detection.

    Args:
        db: Database session
        documents: List of document creation schemas
        workspace_name: Name of the workspace
        observer: Name of the observing peer
        observed: Name of the observed peer

    Returns:
        Count of new documents
    """
    honcho_documents: list[models.Document] = []
    # Store (document_model, embedding) pairs - IDs aren't available until after commit
    docs_with_embeddings: list[tuple[models.Document, list[float]]] = []

    for doc in documents:
        try:
            # for each document, if deduplicate is True, perform a process
            # that checks against existing documents and either rejects this document
            # as a duplicate OR deletes an existing document that is a duplicate.
            if deduplicate:
                is_duplicate = await is_rejected_duplicate(
                    db, doc, workspace_name, observer=observer, observed=observed
                )
                if is_duplicate:
                    continue

            metadata_dict = doc.metadata.model_dump(exclude_none=True)
            new_doc = models.Document(
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                content=doc.content,
                level=doc.level,
                times_derived=doc.times_derived,
                internal_metadata=metadata_dict,
                session_name=doc.session_name,
            )
            honcho_documents.append(new_doc)

            # Track embedding for vector store (ID will be available after commit)
            if doc.embedding:
                docs_with_embeddings.append((new_doc, doc.embedding))

        except Exception as e:
            logger.error(
                f"Error adding new document to {workspace_name}/{doc.session_name}/{observer}/{observed}: {e}"
            )
            continue

    try:
        db.add_all(honcho_documents)
        await db.commit()

        # Store embeddings in vector store after documents are committed (IDs now available)
        if docs_with_embeddings:
            vector_store = get_vector_store()
            namespace = vector_store.get_document_namespace(
                workspace_name, observer, observed
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

            # Retry vector upsert with exponential backoff
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
                    reraise=True,
                ):
                    with attempt:
                        await vector_store.upsert_many(namespace, vector_records)
            except Exception as e:
                # Final attempt failed - log but don't raise
                # Documents exist in DB, vectors can be added manually later
                logger.error(f"Failed to upsert vectors after retries: {e}")

    except IntegrityError as e:
        await db.rollback()
        raise ValidationException(
            "Failed to create documents due to integrity constraint violation"
        ) from e

    return len(honcho_documents)


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
    Delete a single document by ID using hybrid sync/soft delete pattern.

    Tries to delete from vector store first, then hard deletes from DB.
    If vector store delete fails, soft deletes (sets deleted_at) and lets
    cleanup job handle vector deletion later.

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
    # Build base query conditions
    conditions = [
        models.Document.id == document_id,
        models.Document.workspace_name == workspace_name,
        models.Document.observer == observer,
        models.Document.observed == observed,
        models.Document.deleted_at.is_(None),  # Only delete non-deleted docs
    ]
    if session_name is not None:
        conditions.append(models.Document.session_name == session_name)

    # Check document exists first
    check_stmt = select(models.Document).where(*conditions)
    result = await db.execute(check_stmt)
    doc = result.scalar_one_or_none()

    if doc is None:
        raise ResourceNotFoundException(
            f"Document {document_id} not found or does not belong to the specified collection/session"
        )

    # Try to delete from vector store first
    vector_store = get_vector_store()
    namespace = vector_store.get_document_namespace(workspace_name, observer, observed)
    vector_deleted = False

    try:
        await vector_store.delete_many(namespace, [document_id])
        vector_deleted = True
    except Exception as e:
        logger.warning(f"Failed to delete vector for document {document_id}: {e}")

    if vector_deleted:
        # Happy path: hard delete from DB
        delete_stmt = delete(models.Document).where(models.Document.id == document_id)
        await db.execute(delete_stmt)
    else:
        # Fallback: soft delete, let cleanup job handle vector
        update_stmt = (
            update(models.Document)
            .where(models.Document.id == document_id)
            .values(deleted_at=func.now())
        )
        await db.execute(update_stmt)

    await db.commit()


async def delete_document_by_id(
    db: AsyncSession,
    workspace_name: str,
    document_id: str,
) -> None:
    """
    Delete a single document by ID and workspace using hybrid sync/soft delete pattern.

    Tries to delete from vector store first, then hard deletes from DB.
    If vector store delete fails, soft deletes (sets deleted_at) and lets
    cleanup job handle vector deletion later.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        document_id: ID of the document to delete

    Raises:
        ResourceNotFoundException: If document not found or doesn't belong to the workspace
    """
    # Fetch document to get observer/observed for namespace
    stmt = select(models.Document).where(
        models.Document.id == document_id,
        models.Document.workspace_name == workspace_name,
        models.Document.deleted_at.is_(None),  # Only delete non-deleted docs
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if doc is None:
        raise ResourceNotFoundException(
            f"Document {document_id} not found or does not belong to workspace {workspace_name}"
        )

    # Try to delete from vector store first
    vector_store = get_vector_store()
    namespace = vector_store.get_document_namespace(
        workspace_name, doc.observer, doc.observed
    )
    vector_deleted = False

    try:
        await vector_store.delete_many(namespace, [document_id])
        vector_deleted = True
    except Exception as e:
        logger.warning(f"Failed to delete vector for document {document_id}: {e}")

    if vector_deleted:
        # Happy path: hard delete from DB
        delete_stmt = delete(models.Document).where(models.Document.id == document_id)
        await db.execute(delete_stmt)
    else:
        # Fallback: soft delete, let cleanup job handle vector
        update_stmt = (
            update(models.Document)
            .where(models.Document.id == document_id)
            .values(deleted_at=func.now())
        )
        await db.execute(update_stmt)

    await db.commit()


async def create_observations(
    db: AsyncSession,
    observations: list[schemas.ObservationCreate],
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
        sessions_to_validate.add(obs.session_id)
        peers_to_validate.add(obs.observer_id)
        peers_to_validate.add(obs.observed_id)
        collection_pairs.add((obs.observer_id, obs.observed_id))

    # Validate all sessions exist
    for session_name in sessions_to_validate:
        await get_session(db, session_name, workspace_name)

    # Validate all peers exist
    for peer_name in peers_to_validate:
        await get_peer(db, workspace_name, schemas.PeerCreate(name=peer_name))

    # Get or create all collections
    for observer, observed in collection_pairs:
        await get_or_create_collection(
            db, workspace_name, observer=observer, observed=observed
        )

    # Generate embeddings in batch
    contents = [obs.content for obs in observations]
    try:
        embeddings = await embedding_client.simple_batch_embed(contents)
    except ValueError as e:
        raise ValidationException(str(e)) from e

    # Create document objects and track embeddings for vector store
    honcho_documents: list[models.Document] = []
    # Group observations by collection (observer, observed) for vector store upserts
    collection_embeddings: dict[
        tuple[str, str], list[tuple[models.Document, list[float]]]
    ] = {}

    for obs, embedding in zip(observations, embeddings, strict=True):
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

        # Store embeddings in vector store after documents are committed (IDs now available)
        vector_store = get_vector_store()
        for (observer, observed), docs_with_embeddings in collection_embeddings.items():
            namespace = vector_store.get_document_namespace(
                workspace_name, observer, observed
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

            # Retry vector upsert with exponential backoff
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
                    reraise=True,
                ):
                    with attempt:
                        await vector_store.upsert_many(namespace, vector_records)
            except Exception as e:
                # Final attempt failed - log but don't raise
                # Documents exist in DB, vectors can be added manually later
                logger.error(
                    f"Failed to upsert vectors for {namespace} after retries: {e}"
                )

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


async def is_rejected_duplicate(
    db: AsyncSession,
    doc: schemas.DocumentCreate,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
) -> bool:
    """
    Check if a document is a duplicate of an existing document.

    Uses: 1) Cosine similarity (>=0.95), 2) Token diff for retention.

    Returns True if both:
    - the document is deemed a duplicate of an existing document
    - the existing document is deemed a superior duplicate

    If the document is not a duplicate, returns False.

    If the document is a duplicate AND the new document is superior,
    deletes the existing document and returns False.
    """
    # Step 1: Find potential duplicates using cosine similarity
    similar_docs = await query_documents(
        db=db,
        workspace_name=workspace_name,
        query=doc.content,
        observer=observer,
        observed=observed,
        max_distance=0.05,
        top_k=1,
        embedding=doc.embedding,
    )

    if not similar_docs:
        return False

    existing_doc = similar_docs[0]

    # Step 2: Determine which has more information using token set difference
    tokens_new = set(embedding_client.encoding.encode(doc.content))
    tokens_existing = set(embedding_client.encoding.encode(existing_doc.content))

    unique_new = len(tokens_new - tokens_existing)
    unique_existing = len(tokens_existing - tokens_new)

    score_new = len(tokens_new) + (unique_new * 10)
    score_existing = len(tokens_existing) + (unique_existing * 10)

    # If new document has more or equal information, keep it and delete existing
    if score_new >= score_existing:
        logger.warning(
            f"[DUPLICATE DETECTION] Deleting existing in favor of new. new='{doc.content}', existing='{existing_doc.content}'."
        )
        vector_store = get_vector_store()
        namespace = vector_store.get_document_namespace(
            workspace_name, observer, observed
        )
        vector_deleted = False
        try:
            await vector_store.delete_many(namespace, [existing_doc.id])
            vector_deleted = True
        except Exception:
            existing_doc.deleted_at = datetime.datetime.now(datetime.timezone.utc)
            await db.flush()

        if vector_deleted:
            await db.delete(existing_doc)
            await db.flush()  # Flush to make deletion visible in this transaction

        return False  # Don't reject the new document

    # Existing document has more information, reject the new one
    logger.warning(
        f"[DUPLICATE DETECTION] Rejecting new in favor of existing. new='{doc.content}', existing='{existing_doc.content}'."
    )
    return True


async def cleanup_soft_deleted_documents(
    db: AsyncSession,
    vector_store: VectorStore,
    batch_size: int = 100,
    older_than_minutes: int = 5,
) -> int:
    """
    Clean up soft-deleted documents by deleting from vector store and hard deleting from DB.

    This function is designed to be called periodically (e.g., every 5 minutes) to reconcile
    any documents that were soft-deleted when the vector store was unavailable.

    Steps:
    1. Find documents with deleted_at older than threshold
    2. Group by namespace (workspace/observer/observed)
    3. Delete from vector store (per namespace)
    4. Hard delete from DB only for documents where vector deletion succeeded

    If vector deletion fails for a namespace, those documents remain soft-deleted
    and will be retried on the next cleanup run.

    Args:
        db: Database session
        vector_store: Vector store instance
        batch_size: Maximum number of documents to process per call
        older_than_minutes: Only process documents soft-deleted more than this many minutes ago

    Returns:
        Count of documents cleaned up (only those where vector deletion succeeded)
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
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
        namespace = vector_store.get_document_namespace(
            doc.workspace_name, doc.observer, doc.observed
        )
        by_namespace.setdefault(namespace, []).append(doc.id)

    # Delete from vector store (per namespace) and track successful deletions
    successfully_deleted_ids: set[str] = set()
    for namespace, ids in by_namespace.items():
        try:
            await vector_store.delete_many(namespace, ids)
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
    return 0
