from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import attributes

from src import models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.exceptions import ValidationException
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def query_documents(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    collection_name: str,
    query: str,
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
        peer_name: Name of the peer
        collection_name: Name of the collection
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

    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.peer_name == peer_name)
        .where(models.Document.collection_name == collection_name)
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


async def create_document(
    db: AsyncSession,
    document: schemas.DocumentCreate,
    collection: models.Collection,
    embedding: list[float],
    duplicate_threshold: float | None = None,
) -> tuple[models.Document, bool]:
    """
    Embed text as a vector and create a document.

    If duplicate_threshold is provided, we use the embedding to check for
    duplicates within the threshold. If a duplicate is found, instead of
    saving a new document, we increment the times_derived field in the
    `internal_metadata` of the duplicate document.

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
    if duplicate_threshold is not None:
        distance = 1 - duplicate_threshold
        stmt = (
            select(models.Document)
            .where(models.Document.workspace_name == collection.workspace_name)
            .where(models.Document.peer_name == collection.peer_name)
            .where(models.Document.collection_name == collection.name)
            .where(models.Document.embedding.cosine_distance(embedding) < distance)
            .order_by(models.Document.embedding.cosine_distance(embedding))
            .limit(1)
        )
        result = await db.execute(stmt)
        duplicate = result.scalar_one_or_none()  # Get the closest match if any exist
        if duplicate is not None:
            # Get the actual distance for debugging
            distance_stmt = select(
                models.Document.embedding.cosine_distance(embedding)
            ).where(models.Document.id == duplicate.id)
            distance_result = await db.execute(distance_stmt)
            actual_distance = distance_result.scalar()
            actual_similarity = (
                1 - actual_distance if actual_distance is not None else None
            )

            logger.info(
                f"Duplicate found: '{document.content}' matched with '{duplicate.content}'. "
                + f"Similarity: {actual_similarity:.4f}, Distance: {actual_distance:.4f}, "
                + f"Threshold: {duplicate_threshold}. Incrementing times_derived."
            )
            if "times_derived" not in duplicate.internal_metadata:
                duplicate.internal_metadata["times_derived"] = 2
            else:
                duplicate.internal_metadata["times_derived"] += 1
            attributes.flag_modified(duplicate, "internal_metadata")
            await db.commit()
            await db.refresh(duplicate)
            return duplicate, True

    honcho_document = models.Document(
        workspace_name=collection.workspace_name,
        peer_name=collection.peer_name,
        collection_name=collection.name,
        content=document.content,
        internal_metadata=document.metadata.model_dump(exclude_none=True),
        embedding=embedding,
    )
    db.add(honcho_document)
    await db.commit()
    await db.refresh(honcho_document)
    return honcho_document, False
