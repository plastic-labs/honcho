from logging import getLogger
from typing import Any

from nanoid import generate as generate_nanoid
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.exceptions import DisabledException, ValidationException
from src.utils.filter import apply_filter

from .session import get_or_create_session

logger = getLogger(__name__)


async def create_messages(
    db: AsyncSession,
    messages: list[schemas.MessageCreate],
    workspace_name: str,
    session_name: str,
) -> list[models.Message]:
    """
    Bulk create messages for a session while maintaining order.

    Args:
        db: Database session
        messages: List of messages to create
        workspace_name: Name of the workspace
        session_name: Name of the session to create messages in

    Returns:
        List of created message objects
    """
    # Get or create session with peers in messages list
    peers = {message.peer_name: schemas.SessionPeerConfig() for message in messages}
    await get_or_create_session(
        db,
        session=schemas.SessionCreate(name=session_name, peers=peers),
        workspace_name=workspace_name,
    )

    # Create list of message objects (this will trigger the before_insert event)
    message_objects: list[models.Message] = []
    for message in messages:
        message_obj = models.Message(
            session_name=session_name,
            peer_name=message.peer_name,
            content=message.content,
            h_metadata=message.metadata or {},
            workspace_name=workspace_name,
            public_id=generate_nanoid(),
            token_count=len(message.encoded_message),
        )
        message_objects.append(message_obj)

    db.add_all(message_objects)
    await db.flush()

    if settings.EMBED_MESSAGES:
        encoded_message_lookup = {
            msg.public_id: orig_msg.encoded_message
            for msg, orig_msg in zip(message_objects, messages, strict=True)
        }
        id_resource_dict = {
            message.public_id: (
                message.content,
                encoded_message_lookup[message.public_id],
            )
            for message in message_objects
        }
        embedding_dict = await embedding_client.batch_embed(id_resource_dict)

        # Create MessageEmbedding entries for each embedded message
        embedding_objects: list[models.MessageEmbedding] = []
        for message_obj in message_objects:
            embeddings = embedding_dict.get(message_obj.public_id, [])
            for embedding in embeddings:
                embedding_obj = models.MessageEmbedding(
                    content=message_obj.content,
                    embedding=embedding,
                    message_id=message_obj.public_id,
                    workspace_name=workspace_name,
                    session_name=session_name,
                    peer_name=message_obj.peer_name,
                )
                embedding_objects.append(embedding_obj)

        # Add all embedding objects to the session
        if embedding_objects:
            db.add_all(embedding_objects)

    await db.commit()

    return message_objects


async def get_messages(
    workspace_name: str,
    session_name: str,
    reverse: bool | None = False,
    filters: dict[str, Any] | None = None,
    token_limit: int | None = None,
    message_count_limit: int | None = None,
) -> Select[tuple[models.Message]]:
    """
    Get messages from a session. If token_limit is provided, the n most recent messages
    with token count adding up to the limit will be returned. If message_count_limit is provided,
    the n most recent messages will be returned. If both are provided, message_count_limit will be
    used.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session
        reverse: Whether to reverse the order of messages
        filters: Filter to apply to the messages
        token_limit: Maximum number of tokens to include in the messages
        message_count_limit: Maximum number of messages to include

    Returns:
        Select statement for the messages
    """
    # Base query with workspace and session filters
    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]

    # Apply message count limit first (takes precedence over token limit)
    if message_count_limit is not None:
        stmt = select(models.Message).where(*base_conditions)
        stmt = apply_filter(stmt, models.Message, filters)
        # For message count limit, we want the most recent N messages
        # So we order by id desc to get most recent, then apply limit
        stmt = stmt.order_by(models.Message.id.desc()).limit(message_count_limit)

        # Apply final ordering based on reverse parameter
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())
    elif token_limit is not None:
        # Apply token limit logic
        # Create a subquery that calculates running sum of tokens for most recent messages
        token_subquery = (
            select(
                models.Message.id,
                func.sum(models.Message.token_count)
                .over(order_by=models.Message.id.desc())
                .label("running_token_sum"),
            )
            .where(*base_conditions)
            .subquery()
        )

        # Select Message objects where running sum doesn't exceed token_limit
        stmt = (
            select(models.Message)
            .join(token_subquery, models.Message.id == token_subquery.c.id)
            .where(token_subquery.c.running_token_sum <= token_limit)
        )
        stmt = apply_filter(stmt, models.Message, filters)

        # Apply final ordering based on reverse parameter
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())
    else:
        # Default case - no limits applied
        stmt = select(models.Message).where(*base_conditions)
        stmt = apply_filter(stmt, models.Message, filters)
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())

    return stmt


async def get_messages_id_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    start_id: int = 0,
    end_id: int | None = None,
) -> list[models.Message]:
    """
    Get messages from a session by primary key ID range.
    If end_id is not provided, all messages after and including start_id will be returned.
    If start_id is not provided, start will be beginning of session.

    Note: list is *inclusive* of the end_id message.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        start_id: Primary key ID of the first message to return
        end_id: Primary key ID of the last message (exclusive)

    Returns:
        List of messages
    """
    if start_id < 0 or (end_id is not None and (start_id >= end_id or end_id <= 0)):
        return []
    stmt = (
        select(models.Message)
        .where(
            models.Message.workspace_name == workspace_name,
        )
        .where(models.Message.session_name == session_name)
    )
    if end_id:
        stmt = stmt.where(models.Message.id.between(start_id, end_id))
    else:
        stmt = stmt.where(models.Message.id >= start_id)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_message_seq_in_session(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    message_id: int,
) -> int:
    """
    Get the sequence number of a message within a session.

    Args:
        db: Database session
        session_name: Name of the session
        message_id: Primary key ID of the message

    Returns:
        The sequence number of the message (1-indexed)
    """
    stmt = (
        select(func.count(models.Message.id))
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
        .where(models.Message.id < message_id)
    )
    result = await db.execute(stmt)
    count = result.scalar() or 0
    return count + 1


async def get_message(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    message_id: str,
) -> models.Message | None:
    stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
        .where(models.Message.public_id == message_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_message(
    db: AsyncSession,
    message: schemas.MessageUpdate,
    workspace_name: str,
    session_name: str,
    message_id: str,
) -> bool:
    honcho_message = await get_message(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        message_id=message_id,
    )
    if honcho_message is None:
        raise ValueError("Message not found or does not belong to user")
    if (
        message.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_message.h_metadata = message.metadata
    await db.commit()
    # await db.refresh(honcho_message)
    return honcho_message


async def search(
    query: str,
    *,
    workspace_name: str,
    session_name: str | None = None,
    peer_name: str | None = None,
    semantic: bool | None = None,
) -> Select[tuple[models.Message]]:
    """
    Search across message content using a hybrid approach:
    - Uses semantic search if embed_messages is set, else fall back to full text
    - Uses PostgreSQL full text search for natural language queries
    - Falls back to exact string matching for queries with special characters
    - Optionally uses semantic search with embeddings

    If a session or peer is provided, the search will be scoped to that
    session or peer. Otherwise, it will search across all messages in the workspace.

    Args:
        query: Search query to match against message content
        workspace_name: Name of the workspace
        session_name: Optional name of the session
        peer_name: Optional name of the peer
        semantic: Optional boolean to configure semantic search:
            - None: try semantic search if embed_messages is set, else fall back to full text
            - True: try semantic search if embed_messages is set, else throw error
            - False: use full text search

    Returns:
        List of messages that match the search query, ordered by relevance
    """
    import re

    from sqlalchemy import func, or_

    # Base query conditions
    base_conditions = [models.Message.workspace_name == workspace_name]

    should_use_semantic_search = False  # Default to full text search

    if semantic is None:
        # Try semantic search if embed_messages is set, else fall back to full text
        should_use_semantic_search = settings.EMBED_MESSAGES
    elif semantic is True:
        # Try semantic search if embed_messages is set, else throw error
        if settings.EMBED_MESSAGES:
            should_use_semantic_search = True
        else:
            raise DisabledException(
                "Semantic search requires EMBED_MESSAGES flag to be enabled"
            )

    if should_use_semantic_search:
        # Generate embedding for the search query
        try:
            embedding_query = await embedding_client.embed(query)
        except ValueError as e:
            raise ValidationException(
                f"Query exceeds maximum token limit of {settings.MAX_EMBEDDING_TOKENS}."
            ) from e

        # Use cosine distance for semantic search on MessageEmbedding table
        # Join with Message table to get the actual message data
        base_query = (
            select(models.Message)
            .join(
                models.MessageEmbedding,
                models.Message.public_id == models.MessageEmbedding.message_id,
            )
            .where(models.MessageEmbedding.workspace_name == workspace_name)
            .order_by(
                models.MessageEmbedding.embedding.cosine_distance(embedding_query)
            )
        )

        if session_name is not None:
            stmt = base_query.where(
                models.MessageEmbedding.session_name == session_name
            )
        elif peer_name is not None:
            stmt = base_query.where(models.MessageEmbedding.peer_name == peer_name)
        else:
            stmt = base_query

    else:
        # Check if query contains special characters that FTS might not handle well
        has_special_chars = bool(
            re.search(r'[~`!@#$%^&*()_+=\[\]{};\':"\\|,.<>/?-]', query)
        )

        if has_special_chars:
            # For queries with special characters, use exact string matching (ILIKE)
            # This ensures we can find exact matches like "~special-uuid~"
            search_condition = models.Message.content.ilike(f"%{query}%")

            base_query = (
                select(models.Message)
                .where(*base_conditions, search_condition)
                .order_by(models.Message.created_at.desc())
            )
        else:
            # For natural language queries, use full text search with ranking
            fts_condition = func.to_tsvector("english", models.Message.content).op(
                "@@"
            )(func.plainto_tsquery("english", query))

            # Combine FTS with ILIKE as fallback for better coverage
            combined_condition = or_(
                fts_condition, models.Message.content.ilike(f"%{query}%")
            )

            base_query = (
                select(models.Message)
                .where(*base_conditions, combined_condition)
                .order_by(
                    # Order by FTS relevance first, then by creation time
                    func.coalesce(
                        func.ts_rank(
                            func.to_tsvector("english", models.Message.content),
                            func.plainto_tsquery("english", query),
                        ),
                        0,
                    ).desc(),
                    models.Message.created_at.desc(),
                )
            )

        # Add additional filters based on parameters
        if session_name is not None:
            stmt = base_query.where(models.Message.session_name == session_name)
        elif peer_name is not None:
            stmt = base_query.where(models.Message.peer_name == peer_name)
        else:
            stmt = base_query

    return stmt
