import asyncio
import json
import logging
from typing import Any

from langfuse.decorators import langfuse_context, observe  # pyright: ignore

from src.config import settings
from src.models import Document
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
from src.utils.formatting import (
    format_premises_for_display,
)
from src.utils.shared_models import SemanticQueries

from .prompts import query_generation_prompt

# Configure logging
logger = logging.getLogger(__name__)


@observe()
async def get_observations(
    query: str,
    embedding_store: EmbeddingStore,
    *,
    include_premises: bool = False,
    exclude_session_name: str | None = None,
    include_session_name: str | None = None,
    peer_name: str | None = None,
) -> str:
    """
    Generate queries based on the dialectic query and retrieve relevant observations.

    Uses semantic search to find additional relevant historical context beyond
    what's already in the working representation.

    Args:
        query: The user query
        embedding_store: The embedding store to search
        include_premises: Whether to include premises from document metadata
        exclude_session_name: Current session name to exclude from results
        include_session_name: Current session name to exclusively include in results

    Returns:
        String containing additional relevant observations from semantic search
    """
    logger.info("Starting observation retrieval for query: %s", query)
    logger.info("exclude_session_name: %s", exclude_session_name)
    logger.info("include_session_name: %s", include_session_name)

    # Generate search queries with multiple fallback strategies
    search_queries_result = None

    logger.debug(
        "Attempting to generate semantic queries using %s",
        settings.DIALECTIC.QUERY_GENERATION_PROVIDER,
    )
    search_queries_result = await generate_semantic_queries(query, peer_name)
    logger.debug(
        "Successfully generated queries via %s: %s",
        settings.DIALECTIC.QUERY_GENERATION_PROVIDER,
        search_queries_result,
    )

    # search_queries_result should never be None based on function return types

    search_queries = search_queries_result.queries
    # Include the original query in the search queries
    search_queries.append(query)
    logger.info(
        "Generated %s search queries: \n%s",
        len(search_queries),
        json.dumps(search_queries, indent=2),
    )

    # Execute all queries in parallel
    tasks = [_execute_single_query(q, embedding_store) for q in search_queries]
    all_results = await asyncio.gather(*tasks)

    # Combine and deduplicate results
    unique_observations = _deduplicate_observations(all_results)

    langfuse_context.update_current_observation(
        input={
            "query": query,
            "include_premises": include_premises,
        },
        output={
            "search_queries": search_queries_result,
            "all_results": all_results,
            "unique_observations": unique_observations,
        },
    )

    langfuse_context.update_current_trace(
        metadata={
            "search_queries": search_queries,
            "observations_retrieved": unique_observations,
        }
    )

    logger.info(
        "Retrieved %s unique observations before filtering", len(unique_observations)
    )

    # Filter out current session observations to get only historical context
    original_count = len(unique_observations)
    if exclude_session_name:
        filtered_observations = _filter_current_session_observations(
            unique_observations, exclude_session_name
        )
        unique_observations = filtered_observations
    elif include_session_name:
        filtered_observations = _filter_all_but_current_session_observations(
            unique_observations, include_session_name
        )
        unique_observations = filtered_observations
    else:
        filtered_observations = unique_observations

    logger.info(
        "After session filtering: %s observations (removed %s observations)",
        len(unique_observations),
        original_count - len(unique_observations),
    )

    # Format observations
    if not unique_observations:
        logger.info("No unique historical observations found after filtering")
        return "No additional relevant context found."

    # Log a summary of what was retrieved
    logger.info(
        f"Final retrieval summary: {len(unique_observations)} observations retrieved across search queries: \n{json.dumps(unique_observations, indent=2)}"
    )

    return _format_observations(unique_observations, include_premises=include_premises)


async def _execute_single_query(
    query: str, embedding_store: EmbeddingStore
) -> list[tuple[str, str, dict[str, Any]]]:
    """Execute a single semantic search query and return formatted results."""
    documents: list[Document] = await embedding_store.get_relevant_observations(
        query,
        top_k=settings.DIALECTIC.SEMANTIC_SEARCH_TOP_K,
        max_distance=settings.DIALECTIC.SEMANTIC_SEARCH_MAX_DISTANCE,
        for_reasoning=False,
    )

    # Extract data to avoid DetachedInstanceError
    return [
        (
            doc.content,
            doc.created_at.strftime("%Y-%m-%d-%H:%M:%S"),
            doc.internal_metadata or {},
        )
        for doc in documents
    ]


def _deduplicate_observations(
    all_results: list[list[tuple[str, str, dict[str, Any]]]],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Deduplicate observations based on content."""
    unique_observations: list[tuple[str, str, dict[str, Any]]] = []
    seen_content: set[str] = set()

    for results in all_results:
        for content, timestamp, metadata in results:
            if content not in seen_content:
                unique_observations.append((content, timestamp, metadata))
                seen_content.add(content)

    return unique_observations


def _format_observations(
    observations: list[tuple[str, str, dict[str, Any]]], *, include_premises: bool
) -> str:
    """Format observations grouped by level and date, including access metadata."""
    grouped: dict[str, dict[str, list[str]]] = {}

    for content, timestamp, metadata in observations:
        level: str = metadata.get("level", "unknown")
        date_str: str = timestamp[:10]  # Extract YYYY-MM-DD

        if level not in grouped:
            grouped[level] = {}
        if date_str not in grouped[level]:
            grouped[level][date_str] = []

        # Build formatted content with premises and access metadata
        formatted_content: str = content

        # Add premises if requested and available
        if include_premises and metadata.get("premises"):
            premises_text: str = format_premises_for_display(metadata["premises"])
            formatted_content = f"{content}{premises_text}"

        # Prefix with full timestamp for clarity
        if timestamp:
            formatted_content = f"{timestamp}: {formatted_content}"

        # Add access metadata if available
        access_parts: list[str] = []
        access_count: int = metadata.get("access_count", 0)
        last_accessed: Any = metadata.get("last_accessed")

        if access_count > 0:
            access_parts.append(f"accessed {access_count}x")

        if last_accessed:
            # Format the last_accessed datetime for display
            try:
                from datetime import datetime

                if isinstance(last_accessed, str):
                    # Parse ISO format datetime string
                    dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                    formatted_last_accessed: str = dt.strftime("%Y-%m-%d %H:%M")
                    access_parts.append(f"last accessed {formatted_last_accessed}")
            except (ValueError, AttributeError):
                # If parsing fails, just show the raw value
                access_parts.append(f"last accessed {last_accessed}")

        # Append access metadata to the formatted content
        if access_parts:
            access_info: str = ", ".join(access_parts)
            formatted_content = f"{formatted_content} [{access_info}]"

        grouped[level][date_str].append(formatted_content)

    # Build output
    parts: list[str] = []
    for level in sorted(grouped.keys()):
        header: str = (
            f"\n{level.upper()} OBSERVATIONS:"
            if level != "unknown"
            else "\nOBSERVATIONS:"
        )
        parts.append(header)

        for date_str in sorted(
            grouped[level].keys(), reverse=True
        ):  # Most recent first
            parts.append(f"\n{date_str}:")
            for obs in grouped[level][date_str]:
                parts.append(f"  • {obs}")

    return "\n".join(parts).strip()


@honcho_llm_call(
    provider=settings.DIALECTIC.QUERY_GENERATION_PROVIDER,
    model=settings.DIALECTIC.QUERY_GENERATION_MODEL,
    response_model=SemanticQueries,
    enable_retry=True,
    retry_attempts=3,
)
async def generate_semantic_queries(query: str, peer_name: str | None = None):
    """Generate semantic search queries for observation retrieval."""
    return query_generation_prompt(query, peer_name or "")


def _filter_current_session_observations(
    observations: list[tuple[str, str, dict[str, Any]]], session_name: str
) -> list[tuple[str, str, dict[str, Any]]]:
    """Filter out observations from the current session."""
    filtered: list[tuple[str, str, dict[str, Any]]] = []
    current_session_count: int = 0

    for content, timestamp, metadata in observations:
        obs_session_name: str | None = metadata.get(
            "session_name"
        )  # Changed from session_id to session_name
        if obs_session_name != session_name:
            filtered.append((content, timestamp, metadata))
        else:
            current_session_count += 1
            logger.debug(
                "Filtered out current session observation: %s...", content[:50]
            )

    if current_session_count > 0:
        logger.info(
            "Filtered out %s observations from current session %s",
            current_session_count,
            session_name,
        )

    return filtered


def _filter_all_but_current_session_observations(
    observations: list[tuple[str, str, dict[str, Any]]], session_name: str
) -> list[tuple[str, str, dict[str, Any]]]:
    """Filter to keep only observations from the current session."""
    filtered: list[tuple[str, str, dict[str, Any]]] = []

    for content, timestamp, metadata in observations:
        obs_session_name: str | None = metadata.get(
            "session_name"
        )  # Changed from session_id to session_name
        if obs_session_name == session_name:
            filtered.append((content, timestamp, metadata))

    return filtered
