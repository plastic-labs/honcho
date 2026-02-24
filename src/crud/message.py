from datetime import datetime
from logging import getLogger
from typing import Any

from nanoid import generate as generate_nanoid
from sqlalchemy import ColumnElement, Select, and_, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.utils.filter import apply_filter
from src.utils.formatting import ILIKE_ESCAPE_CHAR, escape_ilike_pattern
from src.vector_store import VectorRecord, get_external_vector_store, upsert_with_retry

from .session import get_or_create_session

logger = getLogger(__name__)


def peer_visibility_condition(
    workspace_name: str, peer_perspective: str
) -> ColumnElement[bool]:
    """
    Build a condition that enforces session membership visibility for a peer.

    A message is visible when the peer was a member of the message's session at the
    message timestamp (between joined_at and left_at).
    """
    return (
        select(models.session_peers_table.c.peer_name)
        .where(models.session_peers_table.c.workspace_name == workspace_name)
        .where(models.session_peers_table.c.session_name == models.Message.session_name)
        .where(models.session_peers_table.c.peer_name == peer_perspective)
        .where(models.Message.created_at >= models.session_peers_table.c.joined_at)
        .where(
            or_(
                models.session_peers_table.c.left_at.is_(None),
                models.Message.created_at <= models.session_peers_table.c.left_at,
            )
        )
        .exists()
    )


def _apply_token_limit(
    base_conditions: list[ColumnElement[Any]], token_limit: int
) -> Select[tuple[models.Message]]:
    """
    Helper function to apply token limit logic to a message query.

    Creates a subquery that calculates running sum of tokens for most recent messages
    and returns a select statement that joins with this subquery to limit results
    based on token count.

    Args:
        base_conditions: List of conditions to apply to the base query
        token_limit: Maximum number of tokens to include in the messages

    Returns:
        Select statement with token limit applied
    """
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
    return (
        select(models.Message)
        .join(token_subquery, models.Message.id == token_subquery.c.id)
        .where(token_subquery.c.running_token_sum <= token_limit)
    )


async def _build_merged_snippets(
    db: AsyncSession,
    workspace_name: str,
    matched_messages: list[models.Message],
    context_window: int,
    peer_perspective: str | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Group matched messages by session, merge overlapping context ranges, and fetch context.

    Takes a list of matched messages and builds conversation snippets by:
    1. Grouping matches by session name
    2. Sorting matches within each session by sequence number
    3. Merging overlapping context windows to avoid duplicate context
    4. Fetching the full context for each merged range from the database

    Args:
        db: Database session
        workspace_name: Name of the workspace
        matched_messages: List of messages that matched a search query
        context_window: Number of messages before/after each match to include

    Returns:
        List of tuples: (matched_messages_in_range, context_messages)
        Each tuple represents a snippet where context_messages includes all messages
        in the merged range (including the matched messages), ordered chronologically.
    """
    if not matched_messages:
        return []

    session_matches: dict[str, list[models.Message]] = {}
    for msg in matched_messages:
        session_matches.setdefault(msg.session_name, []).append(msg)

    snippets: list[tuple[list[models.Message], list[models.Message]]] = []

    for sess_name, matches in session_matches.items():
        matches.sort(key=lambda m: m.seq_in_session)

        merged_ranges: list[tuple[int, int, list[models.Message]]] = []

        for match in matches:
            start = match.seq_in_session - context_window
            end = match.seq_in_session + context_window

            if merged_ranges and start <= merged_ranges[-1][1] + 1:
                prev_start, prev_end, prev_matches = merged_ranges[-1]
                merged_ranges[-1] = (
                    prev_start,
                    max(prev_end, end),
                    [*prev_matches, match],
                )
            else:
                merged_ranges.append((start, end, [match]))

        # Batch all ranges into a single query using OR conditions.
        # NOTE: If callers ever pass a very high limit (many disjoint ranges),
        # consider chunking to avoid oversized SQL / planner issues.
        range_conditions = [
            models.Message.seq_in_session.between(start_seq, end_seq)
            for start_seq, end_seq, _ in merged_ranges
        ]
        context_stmt = (
            select(models.Message)
            .where(models.Message.workspace_name == workspace_name)
            .where(models.Message.session_name == sess_name)
            .where(or_(*range_conditions))
            .order_by(models.Message.seq_in_session.asc())
        )
        if peer_perspective:
            context_stmt = context_stmt.where(
                peer_visibility_condition(workspace_name, peer_perspective)
            )

        context_result = await db.execute(context_stmt)
        all_context_messages = list(context_result.scalars().all())

        # Partition results back into their respective ranges
        for start_seq, end_seq, range_matches in merged_ranges:
            context_messages = [
                msg
                for msg in all_context_messages
                if start_seq <= msg.seq_in_session <= end_seq
            ]
            snippets.append((range_matches, context_messages))

    return snippets


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

    await db.execute(text("SET LOCAL lock_timeout = '5s'"))
    await db.execute(
        text(
            "SELECT pg_advisory_xact_lock(hashtext(:workspace_name), hashtext(:session_name))"
        ),
        {"workspace_name": workspace_name, "session_name": session_name},
    )

    # Get the last sequence number on a session - uses (workspace_name, session_name, seq_in_session) index
    last_seq = (
        await db.scalar(
            select(models.Message.seq_in_session)
            .where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
            )
            .order_by(models.Message.seq_in_session.desc())
            .limit(1)
        )
        or 0
    )

    # Create list of message objects (this will trigger the before_insert event)
    message_objects: list[models.Message] = []
    for offset, message in enumerate(messages, start=1):
        message_seq_in_session = last_seq + offset
        message_obj = models.Message(
            session_name=session_name,
            peer_name=message.peer_name,
            content=message.content,
            h_metadata=message.metadata or {},
            workspace_name=workspace_name,
            public_id=generate_nanoid(),
            token_count=len(message.encoded_message),
            created_at=message.created_at,  # Use provided created_at if available
            seq_in_session=message_seq_in_session,
        )
        message_objects.append(message_obj)

    db.add_all(message_objects)

    # Commit here to release the advisory lock before generating embeddings
    await db.commit()
    try:
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

            external_vector_store = get_external_vector_store()

            # Determine if we need to persist embeddings to postgres
            # True when: TYPE=pgvector OR still migrating (dual-write to both stores)
            store_embeddings_in_postgres = (
                settings.VECTOR_STORE.TYPE == "pgvector"
                or not settings.VECTOR_STORE.MIGRATED
            )

            # Create MessageEmbedding entries
            embedding_objects: list[models.MessageEmbedding] = []
            # Maps emb index -> (chunk_position, embedding vector)
            pending_embedding_data: dict[int, tuple[int, list[float]]] = {}
            for message_obj in message_objects:
                embeddings = embedding_dict.get(message_obj.public_id, [])
                for chunk_position, embedding in enumerate(embeddings):
                    embedding_obj = models.MessageEmbedding(
                        content=message_obj.content,
                        message_id=message_obj.public_id,
                        workspace_name=workspace_name,
                        session_name=session_name,
                        peer_name=message_obj.peer_name,
                        sync_state="pending",
                        embedding=embedding if store_embeddings_in_postgres else None,
                    )
                    emb_idx = len(embedding_objects)
                    pending_embedding_data[emb_idx] = (chunk_position, embedding)
                    embedding_objects.append(embedding_obj)

            # Always create MessageEmbedding rows so reconciliation can track sync state
            # even when embeddings aren't stored in postgres
            embedding_ids: list[int] = []
            if embedding_objects:
                db.add_all(embedding_objects)
                await db.flush()
                embedding_ids = [emb.id for emb in embedding_objects]

            await db.commit()

            # If no external vector store (pgvector-only mode), mark as synced immediately
            if external_vector_store is None:
                if embedding_ids:
                    await db.execute(
                        update(models.MessageEmbedding)
                        .where(models.MessageEmbedding.id.in_(embedding_ids))
                        .values(
                            sync_state="synced",
                            last_sync_at=func.now(),
                            sync_attempts=0,
                        )
                    )
                    await db.commit()
            else:
                # External vector store - build and upsert vector records
                namespace = external_vector_store.get_vector_namespace(
                    "message", workspace_name
                )

                # Build vector records with {message_id}_{chunk_position} as vector ID
                vector_records: list[VectorRecord] = []
                for emb_idx, emb in enumerate(embedding_objects):
                    chunk_position, embedding = pending_embedding_data[emb_idx]
                    vector_id = f"{emb.message_id}_{chunk_position}"
                    vector_records.append(
                        VectorRecord(
                            id=vector_id,
                            embedding=list(embedding),
                            metadata={
                                "message_id": emb.message_id,
                                "session_name": emb.session_name,
                                "peer_name": emb.peer_name,
                            },
                        )
                    )

                # Upsert to external vector store with retry and update sync state
                if vector_records:
                    try:
                        await upsert_with_retry(
                            external_vector_store, namespace, vector_records
                        )
                        # Success: mark as synced if we have DB rows
                        if embedding_ids:
                            await db.execute(
                                update(models.MessageEmbedding)
                                .where(models.MessageEmbedding.id.in_(embedding_ids))
                                .values(
                                    sync_state="synced",
                                    last_sync_at=func.now(),
                                    sync_attempts=0,
                                )
                            )
                            await db.commit()

                    except Exception:
                        # Failed after retries - increment sync_attempts for reconciliation
                        logger.exception(
                            "Failed to upsert message vectors after retries"
                        )
                        if embedding_ids:
                            await db.execute(
                                update(models.MessageEmbedding)
                                .where(models.MessageEmbedding.id.in_(embedding_ids))
                                .values(
                                    sync_attempts=models.MessageEmbedding.sync_attempts
                                    + 1,
                                    last_sync_at=func.now(),
                                )
                            )
                            await db.commit()

    except Exception:
        logger.exception(
            "Failed to generate message embeddings for %s messages in workspace %s and session %s.",
            len(message_objects),
            workspace_name,
            session_name,
        )

    return message_objects


async def get_messages(
    workspace_name: str,
    session_name: str,
    reverse: bool | None = False,
    filters: dict[str, Any] | None = None,
    token_limit: int | None = None,
    message_count_limit: int | None = None,
    peer_perspective: str | None = None,
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
        peer_perspective: Optional peer name to enforce session membership visibility

    Returns:
        Select statement for the messages
    """
    # Base query with workspace and session filters
    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]
    if peer_perspective:
        base_conditions.append(
            peer_visibility_condition(workspace_name, peer_perspective)
        )

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
        # Apply token limit logic using helper function
        stmt = _apply_token_limit(base_conditions, token_limit)
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
    token_limit: int | None = None,
) -> list[models.Message]:
    """
    Get messages from a session by primary key ID range.
    If end_id is not provided, all messages after and including start_id will be returned.
    If start_id is not provided, start will be beginning of session.

    Note: list is *inclusive* of the end_id message and start_id message.

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

    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]
    if end_id:
        base_conditions.append(
            and_(models.Message.id >= start_id, models.Message.id < end_id)
        )
    else:
        base_conditions.append(models.Message.id >= start_id)

    if token_limit:
        # Apply token limit logic using helper function
        stmt = _apply_token_limit(base_conditions, token_limit)
        stmt = stmt.order_by(models.Message.id)
    else:
        stmt = select(models.Message).where(*base_conditions)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_messages_by_seq_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    start_seq: int = 1,
    end_seq: int | None = None,
) -> list[models.Message]:
    """
    Get messages from a session by seq_in_session range.

    This is useful for getting the last N messages in a session.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        start_seq: Sequence number of the first message to return (inclusive)
        end_seq: Sequence number of the last message to return (inclusive)

    Returns:
        List of messages ordered by seq_in_session
    """
    if start_seq < 1 or (end_seq is not None and start_seq > end_seq):
        return []

    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]

    if end_seq is not None:
        base_conditions.append(
            and_(
                models.Message.seq_in_session >= start_seq,
                models.Message.seq_in_session <= end_seq,
            )
        )
    else:
        base_conditions.append(models.Message.seq_in_session >= start_seq)

    stmt = (
        select(models.Message)
        .where(*base_conditions)
        .order_by(models.Message.seq_in_session.asc())
    )

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
        select(models.Message.seq_in_session)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
        .where(models.Message.id == message_id)
    )
    seq: int | None = await db.scalar(stmt)
    return int(seq) if seq is not None else 0


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


async def search_messages(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    query: str,
    limit: int = 10,
    context_window: int = 2,
    peer_perspective: str | None = None,
    embedding: list[float] | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages using semantic similarity and return conversation snippets.

    Each result includes matched messages plus surrounding context. Overlapping
    snippets within the same session are merged to avoid repetition.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session (optional)
        query: Search query text
        limit: Maximum number of matching messages to return
        context_window: Number of messages before/after each match to include
        peer_perspective: Optional peer name to enforce session membership visibility
        embedding: Optional pre-computed embedding

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
        Context messages are ordered chronologically and include the matched messages.
    """
    # Use provided embedding or generate one
    query_embedding = (
        embedding if embedding is not None else await embedding_client.embed(query)
    )

    # First, find the top matching messages
    match_stmt = (
        select(models.Message)
        .join(
            models.MessageEmbedding,
            models.Message.public_id == models.MessageEmbedding.message_id,
        )
        .where(models.MessageEmbedding.workspace_name == workspace_name)
        .order_by(models.MessageEmbedding.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )

    if session_name:
        match_stmt = match_stmt.where(
            models.MessageEmbedding.session_name == session_name
        )
    if peer_perspective:
        match_stmt = match_stmt.where(
            peer_visibility_condition(workspace_name, peer_perspective)
        )

    result = await db.execute(match_stmt)
    matched_messages = list(result.scalars().all())

    return await _build_merged_snippets(
        db, workspace_name, matched_messages, context_window, peer_perspective
    )


async def grep_messages(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    text: str,
    limit: int = 10,
    context_window: int = 2,
    peer_perspective: str | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages containing specific text (case-insensitive substring match).

    Unlike semantic search, this finds EXACT text matches. Useful for finding
    specific names, dates, phrases, or keywords.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session (optional - searches all sessions if None)
        text: Text to search for (case-insensitive)
        limit: Maximum number of matching messages to return
        context_window: Number of messages before/after each match to include
        peer_perspective: Optional peer name to enforce session membership visibility

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
    """
    # Build the base query with ILIKE for case-insensitive text search
    escaped_text = escape_ilike_pattern(text)
    match_stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(
            models.Message.content.ilike(f"%{escaped_text}%", escape=ILIKE_ESCAPE_CHAR)
        )
        .order_by(models.Message.created_at.desc())
        .limit(limit)
    )

    if session_name:
        match_stmt = match_stmt.where(models.Message.session_name == session_name)
    if peer_perspective:
        match_stmt = match_stmt.where(
            peer_visibility_condition(workspace_name, peer_perspective)
        )

    result = await db.execute(match_stmt)
    matched_messages = list(result.scalars().all())

    return await _build_merged_snippets(
        db, workspace_name, matched_messages, context_window, peer_perspective
    )


async def get_messages_by_date_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 20,
    order: str = "desc",
    peer_perspective: str | None = None,
) -> list[models.Message]:
    """
    Get messages within a date range.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session (optional - searches all sessions if None)
        after_date: Return messages after this datetime
        before_date: Return messages before this datetime
        limit: Maximum messages to return
        order: Sort order - 'asc' for oldest first, 'desc' for newest first
        peer_perspective: Optional peer name to enforce session membership visibility

    Returns:
        List of messages within the date range
    """
    stmt = select(models.Message).where(models.Message.workspace_name == workspace_name)

    if session_name:
        stmt = stmt.where(models.Message.session_name == session_name)
    if peer_perspective:
        stmt = stmt.where(peer_visibility_condition(workspace_name, peer_perspective))
    if after_date:
        stmt = stmt.where(models.Message.created_at >= after_date)
    if before_date:
        stmt = stmt.where(models.Message.created_at <= before_date)

    if order == "asc":
        stmt = stmt.order_by(models.Message.created_at.asc())
    else:
        stmt = stmt.order_by(models.Message.created_at.desc())

    stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def search_messages_temporal(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    query: str,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 10,
    context_window: int = 2,
    peer_perspective: str | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages using semantic similarity with optional date filtering.

    Combines the power of semantic search with time constraints. Use after_date
    to find recent mentions, or before_date to find what was said before a certain point.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session (optional)
        query: Search query text
        after_date: Only return messages after this datetime
        before_date: Only return messages before this datetime
        limit: Maximum number of matching messages to return
        context_window: Number of messages before/after each match to include
        peer_perspective: Optional peer name to enforce session membership visibility

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
    """
    # Generate embedding for the search query
    query_embedding = await embedding_client.embed(query)

    # Build query with date filters
    match_stmt = (
        select(models.Message)
        .join(
            models.MessageEmbedding,
            models.Message.public_id == models.MessageEmbedding.message_id,
        )
        .where(models.MessageEmbedding.workspace_name == workspace_name)
    )

    if session_name:
        match_stmt = match_stmt.where(
            models.MessageEmbedding.session_name == session_name
        )
    if peer_perspective:
        match_stmt = match_stmt.where(
            peer_visibility_condition(workspace_name, peer_perspective)
        )

    # Apply date filters on the Message table
    if after_date:
        match_stmt = match_stmt.where(models.Message.created_at >= after_date)
    if before_date:
        match_stmt = match_stmt.where(models.Message.created_at <= before_date)

    # Order by similarity and limit
    match_stmt = match_stmt.order_by(
        models.MessageEmbedding.embedding.cosine_distance(query_embedding)
    ).limit(limit)

    result = await db.execute(match_stmt)
    matched_messages = list(result.scalars().all())

    return await _build_merged_snippets(
        db, workspace_name, matched_messages, context_window, peer_perspective
    )
