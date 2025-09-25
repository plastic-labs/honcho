import asyncio
import json
import logging

from langfuse import get_client

from src.config import settings
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
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
        tasks = [
            embedding_store.get_relevant_observations(
                query,
                top_k=settings.DIALECTIC.SEMANTIC_SEARCH_TOP_K,
                max_distance=settings.DIALECTIC.SEMANTIC_SEARCH_MAX_DISTANCE,
            )
            for query in search_queries
        ]
        sub_representations = await asyncio.gather(*tasks)

        # merge all sub_representations into one
        representation = sub_representations.pop()
        for sub_rep in sub_representations:
            representation.merge_representation(sub_rep)

    else:
        representation = await embedding_store.get_relevant_observations(
            query,
            top_k=settings.DIALECTIC.SEMANTIC_SEARCH_TOP_K,
            max_distance=settings.DIALECTIC.SEMANTIC_SEARCH_MAX_DISTANCE,
        )

    return str(representation)


async def generate_semantic_queries(
    query: str, target_peer_name: str
) -> SemanticQueries:
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
    return response.content
