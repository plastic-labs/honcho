from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from src import models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.exceptions import ResourceNotFoundException, ValidationException
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def get_all_documents(
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
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
) -> int:
    """
    Create multiple documents with NO duplicate detection.

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
    Delete a single document by ID.

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
    stmt = delete(models.Document).where(
        models.Document.id == document_id,
        models.Document.workspace_name == workspace_name,
        models.Document.observer == observer,
        models.Document.observed == observed,
    )

    # If session is specified, ensure document belongs to that session
    if session_name is not None:
        stmt = stmt.where(models.Document.session_name == session_name)

    result = await db.execute(stmt)
    await db.commit()

    if result.rowcount == 0:
        raise ResourceNotFoundException(
            f"Document {document_id} not found or does not belong to the specified collection/session"
        )
