import asyncio
import json
import logging
from typing import Any

from langfuse import get_client

from src.config import settings
from src.models import Document
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
from src.utils.formatting import (
    format_premises_for_display,
    parse_datetime_iso,
)
from src.utils.logging import conditional_observe
from src.utils.shared_models import SemanticQueries

from .prompts import query_generation_prompt

# Configure logging
logger = logging.getLogger(__name__)

lf = get_client()


@conditional_observe
async def get_observations(
    query: str,
    target_peer_name: str,
    embedding_store: EmbeddingStore,
    *,
    include_premises: bool = False,
) -> str:
    """
    Generate queries based on the dialectic query and retrieve relevant observations.

    Uses semantic search to find additional relevant historical context beyond
    what's already in the working representation.

    Args:
        query: The user query
        embedding_store: The embedding store to search
        include_premises: Whether to include premises from document metadata

    Returns:
        String containing additional relevant observations from semantic search
    """
    logger.info("Starting observation retrieval for query: %s", query)

    if settings.DIALECTIC.PERFORM_QUERY_GENERATION:
        logger.debug(
            "Attempting to generate semantic queries using %s",
            settings.DIALECTIC.QUERY_GENERATION_PROVIDER,
        )
        search_queries_result = await generate_semantic_queries(query, target_peer_name)
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
        unique_observations = _deduplicate_observations(all_results)

        if settings.LANGFUSE_PUBLIC_KEY:
            lf.update_current_generation(
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

            lf.update_current_trace(
                metadata={
                    "search_queries": search_queries,
                    "observations_retrieved": unique_observations,
                }
            )
    else:
        all_results = [await _execute_single_query(query, embedding_store)]
        unique_observations = _deduplicate_observations(all_results)

    # Format observations
    if not unique_observations:
        logger.info("No unique historical observations found after filtering")
        return "No additional relevant context found."

    return _format_observations(unique_observations, include_premises=include_premises)


async def _execute_single_query(
    query: str,
    embedding_store: EmbeddingStore,
) -> list[tuple[str, str, dict[str, Any]]]:
    """
    Execute a single semantic search query and return formatted results.

    Args:
        query: The query to search for
        embedding_store: The embedding store to use.

    Returns:
        A list of tuples containing the content, timestamp, and metadata of the retrieved observations.
    """
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
                if isinstance(last_accessed, str):
                    # Parse ISO format datetime string
                    dt = parse_datetime_iso(last_accessed)
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


async def generate_semantic_queries(query: str, target_peer_name: str):
    """Generate semantic search queries for observation retrieval."""
    prompt = query_generation_prompt(query, target_peer_name)
    response = await honcho_llm_call(
        provider=settings.DIALECTIC.QUERY_GENERATION_PROVIDER,
        model=settings.DIALECTIC.QUERY_GENERATION_MODEL,
        prompt=prompt,
        max_tokens=settings.LLM.DEFAULT_MAX_TOKENS,
        response_model=SemanticQueries,
        enable_retry=True,
        retry_attempts=3,
    )
    return json.loads(response.content)
