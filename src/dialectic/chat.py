"""
Main dialectic system for AI-powered context synthesis and user representation.

The Dialectic class provides a natural language API for AI applications to query
and understand users through context synthesis of working representations and
historical observations.
"""

import asyncio
import logging

from dotenv import load_dotenv
from langfuse.decorators import langfuse_context
from mirascope.llm import Stream

from src import crud
from src.config import settings
from src.dependencies import tracked_db
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore

from .prompts import dialectic_prompt
from .utils import get_observations

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@honcho_llm_call(
    provider=settings.DIALECTIC.PROVIDER,
    model=settings.DIALECTIC.MODEL,
    track_name="Dialectic Call",
    max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
    thinking_budget_tokens=settings.DIALECTIC.THINKING_BUDGET_TOKENS
    if settings.DIALECTIC.PROVIDER == "anthropic"
    else None,
    enable_retry=True,
    retry_attempts=3,
)
async def dialectic_call(
    query: str,
    working_representation: str,
    additional_context: str | None,
    peer_name: str,
):
    """
    Make a direct call to the dialectic model for context synthesis.

    Args:
        query: The user query
        working_representation: Current session conclusions
        additional_context: Historical context from semantic search
        peer_name: Name of the user/peer

    Returns:
        Model response
    """
    # Generate the prompt and log it
    prompt_result = dialectic_prompt(
        query, working_representation, additional_context, peer_name
    )

    # Pretty print the prompt content
    if len(prompt_result) > 0:
        # Extract content from the first BaseMessageParam
        prompt_content = prompt_result[0].content
    else:
        prompt_content = str(prompt_result)

    logger.debug("=== DIALECTIC PROMPT ===")
    logger.debug(prompt_content)
    logger.debug("=== END DIALECTIC PROMPT ===")

    return prompt_result


@honcho_llm_call(
    provider=settings.DIALECTIC.PROVIDER,
    model=settings.DIALECTIC.MODEL,
    track_name="Dialectic Stream",
    max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
    thinking_budget_tokens=settings.DIALECTIC.THINKING_BUDGET_TOKENS
    if settings.DIALECTIC.PROVIDER == "anthropic"
    else None,
    enable_retry=True,
    retry_attempts=3,
    stream=True,
)
async def dialectic_stream(
    query: str,
    working_representation: str,
    additional_context: str | None,
    peer_name: str,
):
    """
    Make a streaming call to the dialectic model for context synthesis.

    Args:
        query: The user query
        working_representation: Current session conclusions
        additional_context: Historical context from semantic search
        peer_name: Name of the user/peer

    Returns:
        Streaming model response
    """
    # Generate the prompt and log it
    prompt_result = dialectic_prompt(
        query, working_representation, additional_context, peer_name
    )

    # Pretty print the prompt content
    if len(prompt_result) > 0:
        # Extract content from the first BaseMessageParam
        prompt_content = prompt_result[0].content
    else:
        prompt_content = str(prompt_result)

    logger.debug("=== DIALECTIC PROMPT (STREAM) ===")
    logger.debug(prompt_content)
    logger.debug("=== END DIALECTIC PROMPT ===")

    return prompt_result


async def chat(
    workspace_name: str,
    peer_name: str,
    target_name: str | None,
    session_name: str | None,
    query: str,
    *,
    stream: bool = False,
) -> Stream | str:
    """
    Chat with the Dialectic API that builds on-demand user representations.

    Steps:
    1. Get working representation from deriver trace
    2. Retrieve additional relevant context via semantic search
    3. (New) Append observations from latest deriver trace into that context
    4. Call Dialectic to synthesize an answer

    Args:
        workspace_name: Name of the workspace
        peer_name: Name of the peer making the query
        target_name: Optional name of the peer being queried about
        session_name: Optional session name for scoping
        query: Input Dialectic Query
        stream: Whether to stream the response

    Returns:
        Dialectic response (streaming or complete)
    """

    langfuse_context.update_current_trace(
        metadata={
            "query_generation_model": settings.DIALECTIC.QUERY_GENERATION_MODEL,
            "query_generation_provider": settings.DIALECTIC.QUERY_GENERATION_PROVIDER,
            "dialectic_model": settings.DIALECTIC.MODEL,
        }
    )
    logger.debug(f"Received query: {query} for session {session_name}")
    start_time = asyncio.get_event_loop().time()

    # 1. Working representation (short-term) -----------------------------------
    # Only useful for session-scoped queries, not global queries
    if session_name:
        async with tracked_db("chat.get_working_representation") as db:
            # If no target specified, get global representation (peer observing themselves)
            target_peer = target_name if target_name is not None else peer_name

            working_representation = await crud.get_working_representation(
                db, workspace_name, peer_name, target_peer, session_name
            )
    else:
        # For global queries, working representation isn't useful - use historical context instead
        working_representation = ""

    logger.debug(f"Working representation length: {len(working_representation)}")

    # 2. Additional context (long-term semantic search) ------------------------
    # If the query is globally-scoped but not targeted, get global_representation facts from other sessions
    # If the query is globally-scoped and targeted, get facts from other sessions for our target
    # If the query is session-scoped but not targeted, skip this step
    # If the query is session-scoped and targeted, get facts from *only* this session for our target
    if not session_name:
        async with tracked_db("chat.get_additional_context") as db:
            embedding_store = EmbeddingStore(
                workspace_name=workspace_name,
                peer_name=target_name if target_name else peer_name,
                collection_name="global_representation"
                if not target_name
                else crud.construct_collection_name(
                    observer=peer_name, observed=target_name
                ),
            )
            additional_context = await get_observations(
                query,
                embedding_store,
                include_premises=True,
                exclude_session_name=session_name if not target_name else None,
                peer_name=peer_name,
            )
            logger.debug(
                f"Retrieved additional context: {len(additional_context)} characters"
            )
    else:
        if not target_name:
            additional_context = None
        else:
            async with tracked_db("chat.get_additional_context") as db:
                embedding_store = EmbeddingStore(
                    workspace_name=workspace_name,
                    peer_name=target_name,
                    collection_name=crud.construct_collection_name(
                        observer=peer_name, observed=target_name
                    ),
                )
                additional_context = await get_observations(
                    query,
                    embedding_store,
                    include_premises=True,
                    include_session_name=session_name,
                    peer_name=peer_name,
                )
                logger.debug(
                    f"Retrieved additional context: {len(additional_context)} characters"
                )

    # 3. Dialectic call --------------------------------------------------------
    if stream:
        return await dialectic_stream(
            query, working_representation, additional_context, peer_name
        )

    response = await dialectic_call(
        query, working_representation, additional_context, peer_name
    )
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.debug(f"Dialectic answered in {elapsed:.2f}s")
    # Convert AnthropicCallResponse to string for compatibility
    return str(response)
