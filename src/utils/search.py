"""
Reciprocal Rank Fusion (RRF) utilities for combining search results.

RRF is a method to combine multiple ranked lists by computing the reciprocal
of each item's rank in each list, then summing these reciprocal ranks.
"""

import re
from typing import Any, TypeVar

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.embedding_client import embedding_client
from src.exceptions import ValidationException
from src.models import session_peers_table
from src.utils.filter import apply_filter
from src.vector_store import get_vector_store

T = TypeVar("T")


def reciprocal_rank_fusion(*ranked_lists: list[T], k: int = 60, limit: int) -> list[T]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF assigns a score to each item based on the formula:
    RRF_score = sum(1 / (k + rank_i)) for all lists where the item appears

    Where:
    - k is a constant (typically 60) that controls the impact of high-ranked items
    - rank_i is the rank of the item in list i (1-indexed)

    Args:
        *ranked_lists: Variable number of ranked lists to combine
        k: RRF constant parameter (default: 60)
        limit: Maximum number of results to return

    Returns:
        list of items ranked by RRF score (highest score first)
    """
    if not ranked_lists:
        return []

    # dictionary to store RRF scores for each item
    rrf_scores: dict[T, float] = {}

    # Process each ranked list
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):  # 1-indexed ranking
            if item not in rrf_scores:
                rrf_scores[item] = 0.0
            # Add reciprocal rank contribution from this list
            rrf_scores[item] += 1.0 / (k + rank)

    # Sort items by RRF score (descending order)
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Extract just the items (not the scores)
    result = [item for item, _ in sorted_items]

    return result[:limit]


async def _semantic_search(
    db: AsyncSession,
    query: str,
    workspace_name: str,
    limit: int,
    filters: dict[str, Any] | None = None,
) -> list[models.Message]:
    """
    Perform semantic search using external vector store for message embeddings.

    Args:
        db: Database session
        query: Search query
        workspace_name: Name of the workspace to search in
        limit: Maximum number of results to return
        filters: Optional filters to apply at vector store level (supports: session_id, peer_id)

    Returns:
        list of messages ordered by semantic similarity
    """
    try:
        embedding_query = await embedding_client.embed(query)
    except ValueError as e:
        raise ValidationException(
            f"Query exceeds maximum token limit of {settings.MAX_EMBEDDING_TOKENS}."
        ) from e

    # Get vector store and namespace for this workspace's messages
    vector_store = get_vector_store()
    namespace = vector_store.get_message_namespace(workspace_name)

    # Build vector store filters from the provided filters
    vector_filters: dict[str, Any] = {}
    if filters:
        # Map external filter keys to vector store metadata keys
        if "session_id" in filters:
            vector_filters["session_name"] = filters["session_id"]
        if "peer_id" in filters:
            vector_filters["peer_name"] = filters["peer_id"]

    # Query vector store for similar message embeddings
    # Since all filters are applied at the vector store level, we don't need to oversample
    vector_results = await vector_store.query(
        namespace,
        embedding_query,
        top_k=limit,
        filters=vector_filters if vector_filters else None,
    )

    if not vector_results:
        return []

    # Extract message IDs from vector results (vector ID format: {message_public_id}_{chunk_index})
    # Use dict to deduplicate while preserving order (dict keys maintain insertion order in Python 3.7+)
    seen_message_ids: dict[str, None] = {}

    for result in vector_results:
        # Vector ID format: {message_public_id}_{chunk_index}
        parts = result.id.rsplit("_", 1)
        if len(parts) >= 1:
            message_id = parts[0]
            if message_id not in seen_message_ids:
                seen_message_ids[message_id] = None

    message_ids = list(seen_message_ids.keys())

    # Fetch messages from database by the IDs from vector search
    # No additional filtering needed since vector store already applied all filters
    semantic_query = select(models.Message).where(
        models.Message.public_id.in_(message_ids)
    )

    result = await db.execute(semantic_query)
    messages = {msg.public_id: msg for msg in result.scalars().all()}

    # Return messages in order of similarity (preserving vector store order)
    ordered_messages: list[models.Message] = []
    for msg_id in message_ids:
        if msg_id in messages:
            ordered_messages.append(messages[msg_id])

    return ordered_messages


async def _filter_by_peer_perspective(
    db: AsyncSession,
    messages: list[models.Message],
    workspace_name: str,
    peer_name: str,
) -> list[models.Message]:
    """
    Filter messages by peer perspective (temporal session membership).

    Only keeps messages from sessions where the peer was a member at the time
    the message was created (between joined_at and left_at).

    Args:
        db: Database session
        messages: List of messages to filter
        workspace_name: Name of the workspace
        peer_name: Name of the peer whose perspective to use

    Returns:
        Filtered list of messages
    """
    if not messages:
        return []

    # Get all session memberships for this peer in this workspace
    session_memberships_query = (
        select(session_peers_table)
        .where(session_peers_table.c.workspace_name == workspace_name)
        .where(session_peers_table.c.peer_name == peer_name)
    )
    result = await db.execute(session_memberships_query)
    memberships = result.all()

    # Build a lookup of session -> time windows
    session_windows: dict[str, list[tuple[Any, Any]]] = {}
    for membership in memberships:
        session_name = membership.session_name
        if session_name not in session_windows:
            session_windows[session_name] = []
        session_windows[session_name].append((membership.joined_at, membership.left_at))

    # Filter messages
    filtered_messages: list[models.Message] = []
    for msg in messages:
        if msg.session_name not in session_windows:
            continue

        # Check if message was created during any of the peer's active windows in this session
        for joined_at, left_at in session_windows[msg.session_name]:
            if msg.created_at >= joined_at and (
                left_at is None or msg.created_at <= left_at
            ):
                filtered_messages.append(msg)
                break  # Don't add the same message twice

    return filtered_messages


async def _fulltext_search(
    db: AsyncSession,
    query: str,
    stmt: Select[tuple[models.Message]],
    limit: int,
) -> list[models.Message]:
    """
    Perform full-text search using PostgreSQL FTS and ILIKE fallback.

    Args:
        db: Database session
        query: Search query
        stmt: Base SQL query conditions
        limit: Maximum number of results to return

    Returns:
        list of messages ordered by text search relevance
    """
    # Check if query contains special characters that FTS might not handle well
    has_special_chars = bool(
        re.search(r'[~`!@#$%^&*()_+=\[\]{};\':"\\|,.<>/?-]', query)
    )

    if has_special_chars:
        # For queries with special characters, use exact string matching (ILIKE)
        search_condition = models.Message.content.ilike(f"%{query}%")
        fulltext_query = stmt.where(search_condition).order_by(
            models.Message.created_at.desc()
        )
    else:
        # For natural language queries, use full text search with ranking
        fts_condition = func.to_tsvector("english", models.Message.content).op("@@")(
            func.plainto_tsquery("english", query)
        )

        # Combine FTS with ILIKE as fallback for better coverage
        combined_condition = or_(
            fts_condition, models.Message.content.ilike(f"%{query}%")
        )

        fulltext_query = stmt.where(combined_condition).order_by(
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

    fulltext_query = fulltext_query.limit(limit)

    result = await db.execute(fulltext_query)
    return list(result.scalars().all())


async def search(
    db: AsyncSession,
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    limit: int = 10,
) -> list[models.Message]:
    """
    Search across message content using a hybrid approach with Reciprocal Rank Fusion (RRF).

    This function combines semantic search and full-text search results using RRF when both
    are available, providing better search results than either method alone.

    Args:
        db: Database session
        query: Search query to match against message content
        filters: Optional filters to scope search (must include workspace_id for semantic search).
            Special filter 'peer_perspective' will search across all messages from sessions that the peer is/was a member of,
            filtered by the time window when they were actually in the session.
        limit: Maximum number of results to return

    Returns:
        list of messages that match the search query, ordered by RRF relevance or individual search relevance

    Raises:
        ValidationException: If query exceeds maximum token limit for embeddings
    """
    # Base query conditions
    stmt = select(models.Message)

    # Handle special peer_perspective filter
    peer_perspective_name: str | None = None
    if filters and "peer_perspective" in filters:
        peer_perspective_name = filters["peer_perspective"]
        # Remove from filters dict so apply_filter doesn't try to handle it
        filters = {k: v for k, v in filters.items() if k != "peer_perspective"}
        # Safety: peer_perspective must be scoped to a workspace
        if not filters or (
            "workspace_id" not in filters and "workspace_name" not in filters
        ):
            raise ValidationException(
                "peer_perspective requires a workspace scope (workspace_id or workspace_name)."
            )

        # Join with session_peers_table to get messages from sessions the peer was in
        # Only include messages created during the time window the peer was active
        stmt = stmt.join(
            session_peers_table,
            and_(
                models.Message.session_name == session_peers_table.c.session_name,
                models.Message.workspace_name == session_peers_table.c.workspace_name,
                models.Message.created_at >= session_peers_table.c.joined_at,
                or_(
                    session_peers_table.c.left_at.is_(None),
                    models.Message.created_at <= session_peers_table.c.left_at,
                ),
            ),
        ).where(session_peers_table.c.peer_name == peer_perspective_name)

    stmt = apply_filter(stmt, models.Message, filters)

    search_results: list[list[models.Message]] = []

    # Perform semantic search if enabled and we have workspace context
    # workspace_id is required for semantic search to determine the vector namespace
    workspace_name: str | None = filters.get("workspace_id") if filters else None
    if settings.EMBED_MESSAGES and isinstance(workspace_name, str):
        # Type narrowing: workspace_name is guaranteed to be str in this block
        # Get more results for fusion (increase if peer_perspective filtering is applied post-search)
        semantic_limit = limit * 4 if peer_perspective_name else limit * 2
        semantic_results = await _semantic_search(
            db=db,
            query=query,
            workspace_name=workspace_name,
            limit=semantic_limit,
            filters=filters,
        )

        # Apply peer_perspective filtering to semantic results if needed
        # Vector store can't handle temporal filtering (joined_at/left_at), so filter post-search
        if peer_perspective_name:
            semantic_results = await _filter_by_peer_perspective(
                db, semantic_results, workspace_name, peer_perspective_name
            )

        search_results.append(semantic_results)

    # Perform full-text search
    # Get more results for fusion
    fulltext_limit = limit * 2
    fulltext_results = await _fulltext_search(
        db=db, query=query, stmt=stmt, limit=fulltext_limit
    )
    search_results.append(fulltext_results)

    # Combine results using RRF if we have multiple search methods
    if len(search_results) > 1:
        # Use RRF to combine semantic and full-text results
        combined_results = reciprocal_rank_fusion(*search_results, limit=limit)
    elif len(search_results) == 1:
        # Single search method - apply limit directly
        combined_results = search_results[0]
        combined_results = combined_results[:limit]
    else:
        # No search results
        combined_results = []

    return combined_results
