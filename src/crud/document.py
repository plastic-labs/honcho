from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.exceptions import ValidationException
from src.vector_store import VectorRecord, get_vector_store

logger = getLogger(__name__)


async def get_all_documents(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    limit: int = 1000,
) -> Sequence[models.Document]:
    """
    Get all documents in a collection.

    NOTE: Order is nondeterministic. Also this may return a massive amount of documents. Don't use this on large collections.
    TODO: add pagination and update dreaming logic to deduplicate more effectively
    """
    stmt = (
        select(models.Document)
        .limit(limit)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


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
    # No additional filtering needed since vector store already applied all supported filters
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
        .where(models.Document.id.in_(document_ids))
    )

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
    embeddings_to_store: list[tuple[str, list[float]]] = []  # [(doc_id, embedding)]

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

            # Track embedding for vector store (will use document's generated ID)
            if doc.embedding:
                embeddings_to_store.append((new_doc.id, doc.embedding))

        except Exception as e:
            logger.error(
                f"Error adding new document to {workspace_name}/{doc.session_name}/{observer}/{observed}: {e}"
            )
            continue

    try:
        db.add_all(honcho_documents)
        await db.commit()

        # Store embeddings in vector store after documents are committed
        if embeddings_to_store:
            vector_store = get_vector_store()
            namespace = vector_store.get_document_namespace(
                workspace_name, observer, observed
            )

            # Build vector records with metadata for filtering
            vector_records: list[VectorRecord] = []
            doc_lookup = {doc.id: doc for doc in honcho_documents}
            for doc_id, embedding in embeddings_to_store:
                doc = doc_lookup[doc_id]
                vector_records.append(
                    VectorRecord(
                        id=doc_id,
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
            await vector_store.upsert_many(namespace, vector_records)

    except IntegrityError as e:
        await db.rollback()
        raise ValidationException(
            "Failed to create documents due to integrity constraint violation"
        ) from e

    return len(honcho_documents)


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
        # Delete from database
        await db.delete(existing_doc)
        await db.flush()  # Flush to make deletion visible in this transaction

        # Delete from vector store
        vector_store = get_vector_store()
        namespace = vector_store.get_document_namespace(
            workspace_name, observer, observed
        )
        await vector_store.delete_many(namespace, [existing_doc.id])

        return False  # Don't reject the new document

    # Existing document has more information, reject the new one
    logger.warning(
        f"[DUPLICATE DETECTION] Rejecting new in favor of existing. new='{doc.content}', existing='{existing_doc.content}'."
    )
    return True
