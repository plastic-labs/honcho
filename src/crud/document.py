from collections.abc import Sequence
from logging import getLogger
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from src import models, schemas
from src.config import settings
from src.crud.collection import get_or_create_collection
from src.crud.peer import get_peer
from src.crud.session import get_session
from src.embedding_client import embedding_client
from src.exceptions import ResourceNotFoundException, ValidationException
from src.utils.filter import apply_filter

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
    stmt = select(models.Document).where(
        models.Document.workspace_name == workspace_name
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
    stmt = (
        select(models.Document)
        .limit(limit)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
        )
        .order_by(models.Document.created_at.desc())
    )

    if session_name is not None:
        stmt = stmt.where(models.Document.session_name == session_name)

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
        Sequence of documents ordered by times_derived descending
    """
    stmt = (
        select(models.Document)
        .limit(limit)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
        )
        .order_by(models.Document.times_derived.desc())
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
                    level=doc.level,
                    times_derived=doc.times_derived,
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


async def delete_document_by_id(
    db: AsyncSession,
    workspace_name: str,
    document_id: str,
) -> None:
    """
    Delete a single document by ID and workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        document_id: ID of the document to delete

    Raises:
        ResourceNotFoundException: If document not found or doesn't belong to the workspace
    """
    stmt = delete(models.Document).where(
        models.Document.id == document_id,
        models.Document.workspace_name == workspace_name,
    )

    result = await db.execute(stmt)
    await db.commit()

    if result.rowcount == 0:
        raise ResourceNotFoundException(
            f"Document {document_id} not found or does not belong to workspace {workspace_name}"
        )


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

    # Create document objects
    honcho_documents: list[models.Document] = []
    for obs, embedding in zip(observations, embeddings, strict=True):
        honcho_documents.append(
            models.Document(
                workspace_name=workspace_name,
                observer=obs.observer_id,
                observed=obs.observed_id,
                content=obs.content,
                level="explicit",  # Manually created observations are always explicit
                times_derived=1,
                internal_metadata={},  # No message_ids since not derived from messages
                embedding=embedding,
                session_name=obs.session_id,
            )
        )

    try:
        db.add_all(honcho_documents)
        await db.commit()
        # Refresh all documents to get generated IDs and timestamps
        for doc in honcho_documents:
            await db.refresh(doc)
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
        await db.delete(existing_doc)
        await db.flush()  # Flush to make deletion visible in this transaction
        return False  # Don't reject the new document

    # Existing document has more information, reject the new one
    logger.warning(
        f"[DUPLICATE DETECTION] Rejecting new in favor of existing. new='{doc.content}', existing='{existing_doc.content}'."
    )
    return True
