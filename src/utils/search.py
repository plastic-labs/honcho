"""
Reciprocal Rank Fusion (RRF) utilities for combining search results.

RRF is a method to combine multiple ranked lists by computing the reciprocal
of each item's rank in each list, then summing these reciprocal ranks.
"""

import re
from typing import Any, TypeVar

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.embedding_client import embedding_client
from src.exceptions import ValidationException
from src.utils.filter import apply_filter

T = TypeVar("T")


def reciprocal_rank_fusion(
    *ranked_lists: list[T], k: int = 60, limit: int | None = None
) -> list[T]:
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
        limit: Maximum number of results to return (default: None for all results)

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

    # Apply limit if specified
    if limit is not None:
        result = result[:limit]

    return result


async def _semantic_search(
    db: AsyncSession,
    query: str,
    stmt: Select[tuple[models.Message]],
    limit: int | None = None,
) -> list[models.Message]:
    """
    Perform semantic search using message embeddings.

    Args:
        db: Database session
        query: Search query
        base_conditions: Base SQL query conditions
        workspace_name: Name of the workspace
        session_name: Optional session name filter
        peer_name: Optional peer name filter
        limit: Maximum number of results to return

    Returns:
        list of messages ordered by semantic similarity
    """
    try:
        embedding_query = await embedding_client.embed(query)
    except ValueError as e:
        raise ValidationException(
            f"Query exceeds maximum token limit of {settings.MAX_EMBEDDING_TOKENS}."
        ) from e

    # Use cosine distance for semantic search on MessageEmbedding table
    semantic_query = stmt.join(
        models.MessageEmbedding,
        models.Message.public_id == models.MessageEmbedding.message_id,
    ).order_by(models.MessageEmbedding.embedding.cosine_distance(embedding_query))

    if limit is not None:
        semantic_query = semantic_query.limit(limit)

    result = await db.execute(semantic_query)
    return list(result.scalars().all())


async def _fulltext_search(
    db: AsyncSession,
    query: str,
    stmt: Select[tuple[models.Message]],
    limit: int | None = None,
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

    if limit is not None:
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
        filters: Optional filters to scope search
        semantic: Optional boolean to configure search strategy:
            - None: use both semantic and full-text search with RRF if embed_messages is set, else full text only
            - True: use semantic search only if embed_messages is set, else throw error
            - False: use full text search only
        limit: Maximum number of results to return

    Returns:
        list of messages that match the search query, ordered by RRF relevance or individual search relevance

    Raises:
        DisabledException: If semantic search is requested but EMBED_MESSAGES is not enabled
        ValidationException: If query exceeds maximum token limit for embeddings
    """
    # Base query conditions
    stmt = select(models.Message)
    stmt = apply_filter(stmt, models.Message, filters)

    search_results: list[list[models.Message]] = []

    # Perform semantic search if enabled
    if settings.EMBED_MESSAGES:
        # Get more results for fusion
        semantic_limit = limit * 2
        semantic_results = await _semantic_search(
            db=db,
            query=query,
            stmt=stmt,
            limit=semantic_limit or 100,  # Default limit for fusion
        )
        search_results.append(semantic_results)

    # Perform full-text search
    # Get more results for fusion
    fulltext_limit = limit * 2
    fulltext_results = await _fulltext_search(
        db=db,
        query=query,
        stmt=stmt,
        limit=fulltext_limit or 100,  # Default limit for fusion
    )
    search_results.append(fulltext_results)

    # Combine results using RRF if we have multiple search methods
    if len(search_results) > 1:
        # Use RRF to combine semantic and full-text results
        combined_results = reciprocal_rank_fusion(*search_results, limit=limit)
    elif len(search_results) == 1:
        # Single search method - apply limit directly
        combined_results = search_results[0]
        if limit:
            combined_results = combined_results[:limit]
    else:
        # No search results
        combined_results = []

    return combined_results
