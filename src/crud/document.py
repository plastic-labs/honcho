from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.exceptions import ValidationException
from src.utils.dynamic_tables import create_dynamic_document_model
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def get_all_documents(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    collection_name: str,
    collection_id: str,
) -> Sequence[Any]:
    """Get all documents in a collection."""
    DocumentModel = create_dynamic_document_model(collection_id)
    stmt = (
        select(DocumentModel)
        .where(DocumentModel.workspace_name == workspace_name)
        .where(DocumentModel.peer_name == peer_name)
    )
    try:
        result = await db.execute(stmt)
        return result.scalars().all()
    except Exception as e:
        # Table doesn't exist - return empty results instead of crashing
        if "does not exist" in str(e):
            logger.warning(
                f"Table for collection {collection_id} doesn't exist, returning empty results"
            )
            return []
        raise


async def query_documents(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    collection_name: str,
    collection_id: str,
    query: str,
    filters: dict[str, Any] | None = None,
    max_distance: float | None = None,
    top_k: int = 5,
    embedding: list[float] | None = None,
) -> Sequence[Any]:
    """
    Query documents using semantic similarity.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        collection_name: Name of the collection
        collection_id: ID of the collection
        query: Search query text
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

    DocumentModel = create_dynamic_document_model(collection_id)
    stmt = (
        select(DocumentModel)
        .where(DocumentModel.workspace_name == workspace_name)
        .where(DocumentModel.peer_name == peer_name)
    )
    if max_distance is not None:
        stmt = stmt.where(
            DocumentModel.embedding.cosine_distance(embedding) < max_distance
        )
    stmt = apply_filter(stmt, DocumentModel, filters)
    stmt = stmt.limit(top_k).order_by(
        DocumentModel.embedding.cosine_distance(embedding)
    )
    try:
        result = await db.execute(stmt)
        return result.scalars().all()
    except Exception as e:
        # Table doesn't exist - return empty results instead of crashing
        if "does not exist" in str(e):
            logger.warning(
                f"Table for collection {collection_id} doesn't exist, returning empty results"
            )
            return []
        raise


async def create_document(
    db: AsyncSession,
    document: schemas.DocumentCreate,
    collection: models.Collection,
    embedding: list[float],
    duplicate_threshold: float | None = None,
) -> tuple[Any, bool]:
    """
    Embed text as a vector and create a document.

    If duplicate_threshold is provided, we use the embedding to check for
    duplicates within the threshold. If a duplicate is found, instead of
    saving a new document, we increment the times_derived field of the duplicate document.

    Args:
        db: Database session
        document: Document creation schema
        collection: Collection to save the document to
        embedding: Optional pre-computed embedding for the document (avoids extra API call if possible)
        duplicate_threshold: Optional similarity threshold (0-1) for checking for duplicates.
                            Values closer to 1 require higher similarity (e.g., 0.99 = 99% similar)

    Returns:
        The created or updated document, and a boolean indicating whether a duplicate was found

    Raises:
        ResourceNotFoundException: If the collection does not exist
        ValidationException: If the document data is invalid
    """
    DocumentModel = create_dynamic_document_model(collection.id)

    if duplicate_threshold is not None:
        distance = 1 - duplicate_threshold
        stmt = (
            select(DocumentModel)
            .where(DocumentModel.workspace_name == collection.workspace_name)
            .where(DocumentModel.peer_name == collection.peer_name)
            .where(DocumentModel.embedding.cosine_distance(embedding) < distance)
            .order_by(DocumentModel.embedding.cosine_distance(embedding))
            .limit(1)
        )
        result = await db.execute(stmt)
        duplicate = result.scalar_one_or_none()  # Get the closest match if any exist
        if duplicate is not None:
            # Get the actual distance for debugging
            distance_stmt = select(
                DocumentModel.embedding.cosine_distance(embedding)
            ).where(DocumentModel.id == duplicate.id)
            distance_result = await db.execute(distance_stmt)
            actual_distance = distance_result.scalar()
            actual_similarity = (
                1 - actual_distance if actual_distance is not None else None
            )

            logger.info(
                f"Duplicate found: '{document.content}' matched with '{duplicate.content}'. "
                + f"Similarity: {actual_similarity:.4f}, Distance: {actual_distance:.4f}, "
                + f"Threshold: {duplicate_threshold}. Incrementing times_derived (was {duplicate.times_derived})."
            )
            duplicate.times_derived += 1
            await db.commit()
            await db.refresh(duplicate)
            return duplicate, True

    honcho_document = DocumentModel(
        workspace_name=collection.workspace_name,
        peer_name=collection.peer_name,
        content=document.content,
        internal_metadata=document.metadata.model_dump(exclude_none=True),
        embedding=embedding,
    )
    db.add(honcho_document)
    await db.commit()
    await db.refresh(honcho_document)
    return honcho_document, False


async def create_documents_bulk(
    db: AsyncSession,
    documents: list[schemas.DocumentCreate],
    collection: models.Collection,
    embeddings: list[list[float]],
) -> int:
    """
    Create multiple documents with NO duplicate detection.

    Args:
        db: Database session
        documents: List of document creation schemas
        collection: Collection to save documents to
        embeddings: Pre-computed embeddings for each document

    Returns:
        Count of new documents
    """
    if len(documents) != len(embeddings):
        raise ValidationException("Number of documents must match number of embeddings")

    DocumentModel = create_dynamic_document_model(collection.id)

    honcho_documents = [
        DocumentModel(
            workspace_name=collection.workspace_name,
            peer_name=collection.peer_name,
            content=doc.content,
            internal_metadata=doc.metadata.model_dump(exclude_none=True),
            embedding=embedding,
        )
        for doc, embedding in zip(documents, embeddings, strict=True)
    ]
    db.add_all(honcho_documents)
    await db.commit()

    return len(honcho_documents)
