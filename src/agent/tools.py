import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.embedding_client import embedding_client
from src.models import Collection
from src.utils.types import DocumentLevel

logger = logging.getLogger(__name__)


async def create_observations(
    db: AsyncSession,
    observations: list[dict[str, str]],
    observer: str,
    observed: str,
    session_name: str,
    workspace_name: str,
    message_ids: list[int],
    message_created_at: str,
) -> None:
    """
    Create multiple observations (documents) in the memory system in a single call.

    Args:
        db: Database session
        observations: List of observations, each with 'content' and 'level' keys
        observer: The peer making the observation
        observed: The peer being observed
        session_name: Session identifier
        workspace_name: Workspace identifier
        message_ids: List of message IDs these observations are based on
        message_created_at: Timestamp of the message that triggered these observations
    """
    if not observations:
        logger.warning("create_observations called with empty list")
        return

    logger.info(f"Creating {len(observations)} observations")

    # Ensure collection exists for this observer/observed pair
    collection: Collection = await crud.get_or_create_collection(
        db,
        workspace_name,
        observer=observer,
        observed=observed,
    )

    logger.info(f"Using collection for {observer}/{observed}: {collection.id}")

    # Generate embeddings and create document objects for all observations
    documents: list[schemas.DocumentCreate] = []
    for obs in observations:
        content = obs.get("content", "")
        level_str = obs.get("level", "explicit")

        if not content:
            logger.warning("Skipping observation with empty content")
            continue

        # Validate and cast level
        level: DocumentLevel = "deductive" if level_str == "deductive" else "explicit"

        # Generate embedding for the observation
        embedding = await embedding_client.embed(content)

        # Create document
        doc = schemas.DocumentCreate(
            content=content,
            session_name=session_name,
            level=level,
            metadata=schemas.DocumentMetadata(
                message_ids=message_ids,
                message_created_at=message_created_at,
            ),
            embedding=embedding,
        )
        documents.append(doc)

    # Bulk create all documents
    if documents:
        await crud.create_documents(
            db,
            documents=documents,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            deduplicate=True,
        )
        logger.info(
            f"Created {len(documents)} observations for {observed} by {observer}"
        )


async def update_peer_card(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    content: list[str],
) -> None:
    """
    Update the peer card for an observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: The peer maintaining the card
        observed: The peer the card is about
        content: List of facts/information about the observed peer
    """
    await crud.set_peer_card(
        db,
        workspace_name=workspace_name,
        peer_card=content,
        observer=observer,
        observed=observed,
    )
    logger.info(f"Updated peer card for {observed} by {observer}")


async def get_recent_history(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    token_limit: int = 8192,
) -> list[models.Message]:
    """
    Retrieve recent conversation history.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier
        token_limit: Maximum tokens to retrieve

    Returns:
        List of messages in chronological order
    """
    messages = await crud.get_messages(
        workspace_name=workspace_name,
        session_name=session_name,
        token_limit=token_limit,
        reverse=True,  # Get most recent first
    )
    result = await db.execute(messages)

    messages = result.scalars().all()
    # Return in chronological order
    return list(reversed(messages))


async def search_memory(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    query: str,
    limit: int = 5,
) -> list[models.Document]:
    """
    Search for observations in memory using semantic similarity.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: The peer who made the observations
        observed: The peer who was observed
        query: Search query text
        limit: Maximum number of results

    Returns:
        List of documents ordered by relevance
    """
    # Generate embedding for the search query
    query_embedding = await embedding_client.embed(query)

    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
        .order_by(models.Document.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())
