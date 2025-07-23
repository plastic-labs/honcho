from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.exceptions import ValidationException
from src.utils.filter import apply_filter

from .collection import get_collection

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
        embedding: Optional pre-computed embedding for the query (avoids API call)

    Returns:
        Sequence of matching documents
    """
    # Use provided embedding or generate one
    if embedding is None:
        # Using ModelClient for embeddings
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
        # .limit(top_k)
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
    workspace_name: str,
    peer_name: str,
    collection_name: str,
    duplicate_threshold: float | None = None,
) -> models.Document:
    """
    Embed text as a vector and create a document.

    Args:
        db: Database session
        document: Document creation schema
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        collection_name: Name of the collection

    Returns:
        The created document

    Raises:
        ResourceNotFoundException: If the collection does not exist
        ValidationException: If the document data is invalid
    """

    # This will raise ResourceNotFoundException if collection not found
    await get_collection(
        db,
        workspace_name=workspace_name,
        collection_name=collection_name,
        peer_name=peer_name,
    )

    # Using ModelClient for embeddings
    embedding = await embedding_client.embed(document.content)

    if duplicate_threshold is not None:
        # Check if there are duplicates within the threshold
        stmt = (
            select(models.Document)
            .where(models.Document.workspace_name == workspace_name)
            .where(models.Document.peer_name == peer_name)
            .where(models.Document.collection_name == collection_name)
            .where(
                models.Document.embedding.cosine_distance(embedding)
                < duplicate_threshold
            )
            .order_by(models.Document.embedding.cosine_distance(embedding))
            .limit(1)
        )
        result = await db.execute(stmt)
        duplicate = result.scalar_one_or_none()  # Get the closest match if any exist
        if duplicate is not None:
            logger.info(f"Duplicate found: {duplicate.content}. Ignoring new document.")
            return duplicate

    honcho_document = models.Document(
        workspace_name=workspace_name,
        peer_name=peer_name,
        collection_name=collection_name,
        content=document.content,
        internal_metadata=document.metadata,
        embedding=embedding,
    )
    db.add(honcho_document)
    await db.commit()
    await db.refresh(honcho_document)
    return honcho_document


async def get_duplicate_documents(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    collection_name: str,
    content: str,
    similarity_threshold: float = 0.85,
) -> list[models.Document]:
    """Check if a document with similar content already exists in the collection.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        collection_name: Name of the collection
        content: Document content to check for duplicates
        similarity_threshold: Similarity threshold (0-1) for considering documents as duplicates

    Returns:
        List of documents that are similar to the provided content
    """
    # Get embedding for the content
    # Using ModelClient for embeddings
    embedding = await embedding_client.embed(content)

    # Find documents with similar embeddings
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.peer_name == peer_name)
        .where(models.Document.collection_name == collection_name)
        .where(
            models.Document.embedding.cosine_distance(embedding)
            < (1 - similarity_threshold)
        )  # Convert similarity to distance
        .order_by(models.Document.embedding.cosine_distance(embedding))
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())  # Convert to list to match the return type
