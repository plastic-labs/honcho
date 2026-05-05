"""
Reciprocal Rank Fusion (RRF) utilities for combining search results.

RRF is a method to combine multiple ranked lists by computing the reciprocal
of each item's rank in each list, then summing these reciprocal ranks.
"""

import re
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, cast

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.exceptions import ValidationException
from src.models import session_peers_table
from src.utils.filter import apply_filter
from src.utils.formatting import ILIKE_ESCAPE_CHAR, escape_ilike_pattern
from src.vector_store import get_external_vector_store

T = TypeVar("T")


def _uses_pgvector_message_search() -> bool:
    """Return True when semantic message search can stay entirely in Postgres."""
    return (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )


def _build_fts_ranked_query(
    stmt: Select[Any],
    tsquery: Any,
    escaped_query: str,
    content_column: Any,
    created_at_column: Any,
) -> Select[Any]:
    """Build FTS query with ranking using ts_rank for pgvector search."""
    fts_condition = func.to_tsvector("english", content_column).op("@@")(
        tsquery
    )
    combined_condition = or_(
        fts_condition,
        content_column.ilike(
            f"%{escaped_query}%", escape=ILIKE_ESCAPE_CHAR
        ),
    )
    return stmt.where(combined_condition).order_by(
        func.coalesce(
            func.ts_rank(
                func.to_tsvector("english", content_column),
                tsquery,
            ),
            0,
        ),
        created_at_column.desc(),
    )


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


# =============================================================================
# Post-fusion quality utilities: MMR, lexical rerank, score thresholds
# =============================================================================


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class _HasId(Protocol):
    """Protocol for objects with an ``id`` attribute."""

    id: str


def maximal_marginal_relevance(
    documents: Sequence[_HasId],
    query_embedding: list[float],
    document_embeddings: dict[str, list[float]],
    lambda_param: float,
    top_k: int,
) -> list[_HasId]:
    """
    Re-rank items using Maximal Marginal Relevance (MMR).

    MMR balances relevance to the query with diversity among selected results:
        MMR_score = lambda * rel(d, q) - (1 - lambda) * max_sim(d, selected)

    Args:
        documents: Candidate items (already ranked by relevance). Each item must
            have an ``id`` attribute that maps into *document_embeddings*.
        query_embedding: Embedding of the search query.
        document_embeddings: Mapping from item ID to embedding vector.
        lambda_param: Trade-off between relevance and diversity
            (0.0 = max diversity, 1.0 = max relevance).
        top_k: Number of items to return.

    Returns:
        Re-ranked list of items with MMR applied.
    """
    if not documents or lambda_param >= 1.0:
        return list(documents)[:top_k]

    selected: list[_HasId] = []
    remaining = list(documents)

    while remaining and len(selected) < top_k:
        best_doc: _HasId | None = None
        best_score = -float("inf")

        for doc in remaining:
            doc_emb = document_embeddings.get(doc.id)
            if doc_emb is None:
                continue
            rel = _cosine_similarity(query_embedding, doc_emb)
            max_sim = 0.0
            for sel in selected:
                sel_emb = document_embeddings.get(sel.id)
                if sel_emb is not None:
                    sim = _cosine_similarity(doc_emb, sel_emb)
                    if sim > max_sim:
                        max_sim = sim
            score = lambda_param * rel - (1.0 - lambda_param) * max_sim
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is None:
            break
        selected.append(best_doc)
        remaining.remove(best_doc)

    # Append any remaining documents if MMR couldn't fill the quota
    for doc in documents:
        if doc not in selected:
            selected.append(doc)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def _normalize_query_terms(query: str) -> list[str]:
    """Extract normalized query terms for lexical reranking."""
    return [t.lower() for t in re.findall(r"\b\w+\b", query) if len(t) > 2]


def lexical_rerank(
    items: list[T],
    query: str,
) -> list[T]:
    """
    Lightweight lexical reranker using token overlap.

    Scores each item by the number of query terms its content contains,
    with a configurable boost for exact substring matches.

    Args:
        items: Candidate items (must have a ``content`` attribute).
        query: Original search query.

    Returns:
        Items reordered by lexical overlap score.
    """
    terms = _normalize_query_terms(query)
    if not terms:
        return items

    scored: list[tuple[T, float]] = []
    query_lower = query.lower()
    for item in items:
        content = getattr(item, "content", "")
        content_lower = content.lower()
        score = sum(1.0 for term in terms if term in content_lower)
        if query_lower in content_lower:
            score += len(terms) * settings.RETRIEVAL.EXACT_MATCH_BOOST
        scored.append((item, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored]


def _apply_score_threshold(
    ranked_items: list[T],
    *source_lists: list[T],
    threshold: float,
    k: int = 60,
) -> list[T]:
    """
    Filter RRF-ranked items by a minimum composite score.

    Args:
        ranked_items: Items already ordered by RRF.
        *source_lists: The original ranked lists fed into RRF.
        threshold: Minimum RRF score required to keep an item.
        k: RRF constant used during fusion.

    Returns:
        Items with RRF score >= threshold.
    """
    if threshold <= 0.0 or not source_lists:
        return ranked_items

    # Recompute RRF scores
    rrf_scores: dict[T, float] = {}
    for ranked_list in source_lists:
        for rank, item in enumerate(ranked_list, 1):
            if item in ranked_items:
                rrf_scores[item] = rrf_scores.get(item, 0.0) + 1.0 / (k + rank)

    filtered = [item for item in ranked_items if rrf_scores.get(item, 0.0) >= threshold]
    return filtered


# =============================================================================
# Fulltext helpers
# =============================================================================


def _tsquery_func(query: str):
    """Return the appropriate PostgreSQL tsquery function based on settings."""
    if settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH:
        return func.websearch_to_tsquery("english", query)
    return func.plainto_tsquery("english", query)


async def query_external_vector_message_ids(
    workspace_name: str,
    embedding_query: list[float],
    limit: int,
    filters: dict[str, Any] | None = None,
) -> list[str]:
    """Query the external vector store and return ordered message IDs."""
    external_vector_store = get_external_vector_store()
    if external_vector_store is None:
        return []

    namespace = external_vector_store.get_vector_namespace("message", workspace_name)

    vector_filters: dict[str, Any] = {}
    if filters:
        if "session_id" in filters:
            vector_filters["session_name"] = filters["session_id"]
        if "peer_id" in filters:
            vector_filters["peer_name"] = filters["peer_id"]

    # Oversample: multiple chunk-level hits can map to the same message,
    # so fetch extra to ensure enough unique messages after deduplication.
    vector_results = await external_vector_store.query(
        namespace,
        embedding_query,
        top_k=limit * 3,
        filters=vector_filters if vector_filters else None,
    )

    if not vector_results:
        return []

    seen_message_ids: dict[str, None] = {}
    for result in vector_results:
        message_id = result.metadata.get("message_id")
        if message_id and message_id not in seen_message_ids:
            seen_message_ids[message_id] = None

    return list(seen_message_ids.keys())


async def fetch_messages_by_ids(
    db: AsyncSession,
    message_ids: list[str],
    filters: dict[str, Any] | None = None,
) -> list[models.Message]:
    """Fetch messages by ID and preserve the input ordering."""
    if not message_ids:
        return []

    stmt = select(models.Message).where(models.Message.public_id.in_(message_ids))
    stmt = apply_filter(stmt, models.Message, filters)

    result = await db.execute(stmt)
    messages = {msg.public_id: msg for msg in result.scalars().all()}

    return [messages[msg_id] for msg_id in message_ids if msg_id in messages]


async def _semantic_search_pgvector(
    db: AsyncSession,
    workspace_name: str,
    embedding_query: list[float],
    limit: int,
    filters: dict[str, Any] | None = None,
) -> list[models.Message]:
    """
    Perform semantic message search using pgvector in Postgres.

    Args:
        db: Database session
        workspace_name: Name of the workspace to search in
        embedding_query: Pre-computed embedding for the search query
        limit: Maximum number of results to return
        filters: Optional filters to apply to the message query

    Returns:
        list of messages ordered by semantic similarity
    """
    distance_expr = models.MessageEmbedding.embedding.cosine_distance(embedding_query)

    stmt = (
        select(models.Message)
        .join(
            models.MessageEmbedding,
            models.Message.public_id == models.MessageEmbedding.message_id,
        )
        .where(models.MessageEmbedding.embedding.isnot(None))
        .where(models.MessageEmbedding.workspace_name == workspace_name)
    )

    if filters:
        internal_filters = filters.copy()
        internal_filters["workspace_id"] = workspace_name
        stmt = apply_filter(stmt, models.Message, internal_filters)

    # Oversample because a message with multiple embedding chunks can
    # produce duplicate rows; we deduplicate in Python to preserve HNSW
    # index usage (a DISTINCT ON subquery would prevent the index scan).
    stmt = stmt.order_by(distance_expr).limit(limit * 2)

    result = await db.execute(stmt)
    seen: set[str] = set()
    deduped: list[models.Message] = []
    for msg in result.scalars().all():
        if msg.public_id not in seen:
            seen.add(msg.public_id)
            deduped.append(msg)
    return deduped[:limit]


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
    # Escape ILIKE pattern characters to treat user input literally
    escaped_query = escape_ilike_pattern(query)

    if settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH:
        # websearch_to_tsquery handles special chars, quoted phrases, and boolean operators
        tsquery = _tsquery_func(query)
        fulltext_query = _build_fts_ranked_query(
            stmt, tsquery, escaped_query, models.Message.content, models.Message.created_at
        )
    else:
        # Check for special chars that plainto_tsquery can't handle
        if bool(re.search(r'[~`!@#$%^&*()_+=\[\]{};\':"\\|,.<>/?-]', query)):
            # Use ILIKE for queries with chars plainto_tsquery can't handle
            search_condition = models.Message.content.ilike(
                f"%{escaped_query}%", escape=ILIKE_ESCAPE_CHAR
            )
            fulltext_query = stmt.where(search_condition).order_by(
                models.Message.created_at.desc()
            )
        else:
            tsquery = _tsquery_func(query)
            fulltext_query = _build_fts_ranked_query(
                stmt, tsquery, escaped_query, models.Message.content, models.Message.created_at
            )

    fulltext_query = fulltext_query.limit(limit)

    result = await db.execute(fulltext_query)
    return list(result.scalars().all())


async def search(
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
        query: Search query to match against message content
        filters: Optional filters to scope search (must include workspace_id for semantic search).
            Special filter 'peer_perspective' will search across all messages from sessions that the peer is/was a member of,
            filtered by the time window when they were actually in the session.
        limit: Maximum number of results to return

    Returns:
        list of messages that match the search query, ordered by RRF relevance or individual search relevance

    Raises:
        ValidationException: If query exceeds maximum token limit for embeddings, or hybrid search is used with non-pgvector vector store
    """
    # Hybrid search requires pgvector vector store
    if not _uses_pgvector_message_search():
        raise ValidationException(
            "Hybrid retrieval is only supported with pgvector (VECTOR_STORE_TYPE=pgvector)."
            " Set RETRIEVAL_HYBRID_ENABLED=false or switch to pgvector."
        )

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

    workspace_name: str | None = None
    if filters:
        workspace_value = filters.get("workspace_id") or filters.get("workspace_name")
        if isinstance(workspace_value, str):
            workspace_name = workspace_value

    semantic_limit = limit * 4 if peer_perspective_name else limit * 2
    query_embedding: list[float] | None = None
    semantic_message_ids: list[str] | None = None

    if settings.EMBED_MESSAGES and isinstance(workspace_name, str):
        try:
            query_embedding = await embedding_client.embed(query)
        except ValueError as e:
            raise ValidationException(
                f"Query exceeds maximum token limit of {settings.EMBEDDING.MAX_INPUT_TOKENS}."
            ) from e

        if not _uses_pgvector_message_search():
            semantic_message_ids = await query_external_vector_message_ids(
                workspace_name=workspace_name,
                embedding_query=query_embedding,
                limit=semantic_limit,
                filters=filters,
            )

    async def _run_search(active_db: AsyncSession) -> list[models.Message]:
        search_results: list[list[models.Message]] = []

        if (
            settings.EMBED_MESSAGES
            and isinstance(workspace_name, str)
            and query_embedding is not None
        ):
            if _uses_pgvector_message_search():
                semantic_results = await _semantic_search_pgvector(
                    db=active_db,
                    workspace_name=workspace_name,
                    embedding_query=query_embedding,
                    limit=semantic_limit,
                    filters=filters,
                )
            else:
                semantic_results = await fetch_messages_by_ids(
                    db=active_db,
                    message_ids=semantic_message_ids or [],
                    filters=filters,
                )

            if peer_perspective_name:
                semantic_results = await _filter_by_peer_perspective(
                    active_db,
                    semantic_results,
                    workspace_name,
                    peer_perspective_name,
                )

            search_results.append(semantic_results)

        fulltext_results = await _fulltext_search(
            db=active_db,
            query=query,
            stmt=stmt,
            limit=limit * 2,
        )
        search_results.append(fulltext_results)

        if len(search_results) > 1:
            fused = reciprocal_rank_fusion(
                *search_results,
                k=settings.RETRIEVAL.RRF_K,
                limit=limit,
            )
            if settings.RETRIEVAL.SCORE_THRESHOLD is not None:
                fused = _apply_score_threshold(
                    fused,
                    *search_results,
                    threshold=settings.RETRIEVAL.SCORE_THRESHOLD,
                    k=settings.RETRIEVAL.RRF_K,
                )
            if settings.RETRIEVAL.RERANK_ENABLED and fused:
                fused = lexical_rerank(
                    fused[: settings.RETRIEVAL.RERANK_TOP_K], query
                ) + fused[settings.RETRIEVAL.RERANK_TOP_K :]
            return fused
        if len(search_results) == 1:
            return search_results[0][:limit]
        return []

    async with tracked_db("search.messages") as managed_db:
        combined_results = await _run_search(managed_db)
        for message in combined_results:
            managed_db.expunge(message)
        return combined_results


# =============================================================================
# Document hybrid search
# =============================================================================


async def _fulltext_search_documents(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    query: str,
    filters: dict[str, Any] | None,
    limit: int,
    embedding: list[float] | None = None,
    max_distance: float | None = None,
) -> list[models.Document]:
    """
    Perform full-text search over documents using PostgreSQL FTS + ILIKE fallback.

    Args:
        db: Database session.
        workspace_name: Workspace to search in.
        observer: Observing peer.
        observed: Observed peer.
        query: Search query text.
        filters: Optional additional filters.
        limit: Maximum results to return.
        embedding: Query embedding for optional distance filtering.
        max_distance: Optional cosine distance cutoff. When set, only documents
            whose embedding is within this distance are returned.

    Returns:
        Documents ordered by text search relevance.
    """
    escaped_query = escape_ilike_pattern(query)

    # Base conditions scoped to the collection
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.observer == observer)
        .where(models.Document.observed == observed)
        .where(models.Document.deleted_at.is_(None))
    )

    if max_distance is not None and embedding is not None:
        stmt = stmt.where(
            models.Document.embedding.cosine_distance(embedding) <= max_distance
        )

    if filters:
        stmt = apply_filter(stmt, models.Document, filters)

    if settings.RETRIEVAL.FULLTEXT_USE_WEBSEARCH:
        # websearch_to_tsquery handles special chars, quoted phrases, and boolean operators
        tsquery = _tsquery_func(query)
        fulltext_query = _build_fts_ranked_query(
            stmt, tsquery, escaped_query, models.Document.content, models.Document.created_at
        )
    else:
        # Check for special chars that plainto_tsquery can't handle
        if bool(re.search(r'[~`!@#$%^&*()_+=\[\]{};\':"\\|,.<>/?-]', query)):
            # Use ILIKE for queries with chars plainto_tsquery can't handle
            search_condition = models.Document.content.ilike(
                f"%{escaped_query}%", escape=ILIKE_ESCAPE_CHAR
            )
            fulltext_query = stmt.where(search_condition).order_by(
                models.Document.created_at.desc()
            )
        else:
            tsquery = _tsquery_func(query)
            fulltext_query = _build_fts_ranked_query(
                stmt, tsquery, escaped_query, models.Document.content, models.Document.created_at
            )

    fulltext_query = fulltext_query.limit(limit)
    result = await db.execute(fulltext_query)
    return list(result.scalars().all())


async def search_documents_hybrid(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    query: str,
    embedding: list[float],
    filters: dict[str, Any] | None,
    top_k: int,
    max_distance: float | None = None,
) -> list[models.Document]:
    """
    Hybrid document search combining semantic similarity and full-text search.

    Results are fused with Reciprocal Rank Fusion (RRF), then optionally
    re-ranked with MMR and lexical overlap boosting.

    Args:
        db: Database session.
        workspace_name: Workspace to search in.
        observer: Observing peer.
        observed: Observed peer.
        query: Search query text.
        embedding: Pre-computed query embedding.
        filters: Optional additional filters.
        top_k: Number of results to return.
        max_distance: Optional cosine distance cutoff for semantic search.

    Returns:
        Ranked list of matching documents.
    """
    from src.crud.document import _query_documents_pgvector

    # Semantic search (oversample to leave headroom for fusion)
    semantic_results = await _query_documents_pgvector(
        db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        embedding=embedding,
        filters=filters,
        max_distance=max_distance,
        top_k=top_k * 2,
    )

    # Full-text search
    fulltext_results = await _fulltext_search_documents(
        db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        query=query,
        filters=filters,
        limit=top_k * 2,
        embedding=embedding,
        max_distance=max_distance,
    )

    # RRF fusion
    fused = reciprocal_rank_fusion(
        semantic_results,
        fulltext_results,
        k=settings.RETRIEVAL.RRF_K,
        limit=top_k,
    )

    # Score threshold filter
    if settings.RETRIEVAL.SCORE_THRESHOLD is not None:
        fused = _apply_score_threshold(
            fused,
            semantic_results,
            fulltext_results,
            threshold=settings.RETRIEVAL.SCORE_THRESHOLD,
            k=settings.RETRIEVAL.RRF_K,
        )

    # Lightweight lexical rerank (only on top results for speed)
    if settings.RETRIEVAL.RERANK_ENABLED and fused:
        fused = lexical_rerank(
            fused[: settings.RETRIEVAL.RERANK_TOP_K], query
        ) + fused[settings.RETRIEVAL.RERANK_TOP_K :]

    # MMR diversity re-ranking (requires document embeddings)
    if settings.RETRIEVAL.MMR_ENABLED and fused:
        # Fetch embeddings for the fused subset from the DB
        fused_ids = [doc.id for doc in fused]
        if fused_ids:
            emb_stmt = (
                select(models.Document.id, models.Document.embedding)
                .where(models.Document.id.in_(fused_ids))
                .where(models.Document.embedding.isnot(None))
            )
            emb_result = await db.execute(emb_stmt)
            document_embeddings = {
                row.id: list(row.embedding) for row in emb_result.all()
            }
            fused = cast(
                list[models.Document],
                maximal_marginal_relevance(
                    fused,
                    query_embedding=embedding,
                    document_embeddings=document_embeddings,
                    lambda_param=settings.RETRIEVAL.MMR_LAMBDA,
                    top_k=top_k,
                ),
            )

    return fused[:top_k]
