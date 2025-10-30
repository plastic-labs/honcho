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
from src.utils.filter import apply_filter

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
        filters: Optional filters to apply
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

    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
    )
    if max_distance is not None:
        stmt = stmt.where(
            models.Document.embedding.cosine_distance(embedding) < max_distance
        )
    stmt = apply_filter(stmt, models.Document, filters)
    stmt = stmt.limit(top_k).order_by(
        models.Document.embedding.cosine_distance(embedding)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


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
            honcho_documents.append(
                models.Document(
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    content=doc.content,
                    internal_metadata=metadata_dict,
                    embedding=doc.embedding,
                    session_name=doc.session_name,
                )
            )
        except Exception as e:
            logger.error(
                f"Error adding new document to {workspace_name}/{doc.session_name}/{observer}/{observed}: {e}"
            )
            continue
    try:
        db.add_all(honcho_documents)
        await db.commit()
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
        await db.delete(existing_doc)
        await db.flush()  # Flush to make deletion visible in this transaction
        return False  # Don't reject the new document

    # Existing document has more information, reject the new one
    logger.warning(
        f"[DUPLICATE DETECTION] Rejecting new in favor of existing. new='{doc.content}', existing='{existing_doc.content}'."
    )
    return True
