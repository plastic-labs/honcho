"""
Main dialectic system for AI-powered context synthesis and user representation.

The Dialectic class provides a natural language API for AI applications to query
and understand users through context synthesis of working representations and
historical observations.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator

import tiktoken
from dotenv import load_dotenv

from src import crud
from src.config import settings
from src.dependencies import tracked_db
from src.utils import summarizer
from src.utils.clients import HonchoLLMCallStreamChunk, honcho_llm_call
from src.utils.langfuse_client import get_langfuse_client
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
)
from src.utils.representation import Representation

from .prompts import dialectic_prompt

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create langfuse client
lf = get_langfuse_client() if settings.LANGFUSE_PUBLIC_KEY else None


async def dialectic_call(
    query: str,
    working_representation: str,
    recent_conversation_history: str | None,
    peer_card: list[str] | None,
    observed_peer_card: list[str] | None = None,
    *,
    observer: str,
    observed: str,
):
    """
    Make a direct call to the dialectic model for context synthesis.

    Args:
        query: The user query
        working_representation: Current session conclusions AND historical conclusions from the user's global representation
        recent_conversation_history: Recent conversation history
        peer_name: Name of the user/peer
        peer_card: Known biographical information about the user
        observed: Name of the user/peer being queried about
        observed_peer_card: Known biographical information about the target, if applicable

    Returns:
        Model response
    """
    # Generate the prompt and log it
    prompt = dialectic_prompt(
        query,
        working_representation,
        recent_conversation_history,
        peer_card,
        observed_peer_card,
        observer=observer,
        observed=observed,
    )

    response = await honcho_llm_call(
        provider=settings.DIALECTIC.PROVIDER,
        model=settings.DIALECTIC.MODEL,
        prompt=prompt,
        max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
        track_name="Dialectic Call",
        thinking_budget_tokens=settings.DIALECTIC.THINKING_BUDGET_TOKENS
        if settings.DIALECTIC.PROVIDER == "anthropic"
        else None,
        enable_retry=True,
        retry_attempts=3,
    )

    logger.debug("=== DIALECTIC PROMPT ===")
    logger.debug(prompt)
    logger.debug("=== END DIALECTIC PROMPT ===")

    return response.content


async def dialectic_stream(
    query: str,
    working_representation: str,
    recent_conversation_history: str | None,
    peer_card: list[str] | None,
    observed_peer_card: list[str] | None = None,
    *,
    observer: str,
    observed: str,
):
    """
    Make a streaming call to the dialectic model for context synthesis.

    Args:
        query: The user query
        working_representation: Current session conclusions AND historical conclusions from the user's global representation
        recent_conversation_history: Recent conversation history
        peer_name: Name of the user/peer
        peer_card: Known biographical information about the user
        observed: Name of the user/peer being queried about
        observed_peer_card: Known biographical information about the target, if applicable

    Returns:
        Streaming model response
    """
    # Generate the prompt and log it
    prompt = dialectic_prompt(
        query,
        working_representation,
        recent_conversation_history,
        peer_card,
        observed_peer_card,
        observer=observer,
        observed=observed,
    )

    response = await honcho_llm_call(
        provider=settings.DIALECTIC.PROVIDER,
        model=settings.DIALECTIC.MODEL,
        prompt=prompt,
        max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
        track_name="Dialectic Stream",
        thinking_budget_tokens=settings.DIALECTIC.THINKING_BUDGET_TOKENS
        if settings.DIALECTIC.PROVIDER == "anthropic"
        else None,
        enable_retry=True,
        retry_attempts=3,
        stream=True,
    )

    logger.debug("=== DIALECTIC PROMPT (STREAM) ===")
    logger.debug(prompt)
    logger.debug("=== END DIALECTIC PROMPT ===")

    return response


async def chat(
    workspace_name: str,
    session_name: str | None,
    query: str,
    *,
    observer: str,
    observed: str,
    stream: bool = False,
) -> str | AsyncIterator[HonchoLLMCallStreamChunk]:
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
        observed: Optional name of the peer being queried about
        session_name: Optional session name for scoping
        query: Input Dialectic Query
        stream: Whether to stream the response

    Returns:
        Dialectic response (streaming or complete)
    """

    dialectic_chat_uuid = str(uuid.uuid4())

    tokenizer = tiktoken.get_encoding("cl100k_base")

    context_window_size = (
        settings.DIALECTIC.CONTEXT_WINDOW_SIZE - 750
    )  # this is a hardcoded (accurate, slightly conservative) estimate of system prompt

    context_window_size -= len(tokenizer.encode(query))

    if lf:
        lf.update_current_trace(
            metadata={
                "query_generation_model": settings.DIALECTIC.QUERY_GENERATION_MODEL,
                "query_generation_provider": settings.DIALECTIC.QUERY_GENERATION_PROVIDER,
                "dialectic_model": settings.DIALECTIC.MODEL,
            }
        )
    logger.info(
        "Received query:\n'%s'\nobserver: %s, observed: %s%s\n",
        query,
        observer,
        observed,
        f", session: {session_name}" if session_name else "",
    )
    start_time = time.perf_counter()

    # 1. Working representation (short-term) -----------------------------------
    working_rep_start_time = time.perf_counter()
    # If no target specified, get global representation (peer observing themselves)
    working_representation: Representation = await crud.get_working_representation(
        workspace_name,
        observer=observer,
        observed=observed,
        session_name=session_name,
        include_semantic_query=query,
        semantic_search_top_k=settings.DIALECTIC.SEMANTIC_SEARCH_TOP_K,
        semantic_search_max_distance=settings.DIALECTIC.SEMANTIC_SEARCH_MAX_DISTANCE,
        include_most_derived=True,
    )
    working_rep_duration = (time.perf_counter() - working_rep_start_time) * 1000
    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "retrieve_working_rep",
        working_rep_duration,
        "ms",
    )
    logger.info(
        "Retrieved working representation with %s explicit, %s deductive observations",
        len(working_representation.explicit),
        len(working_representation.deductive),
    )

    working_representation_str = str(working_representation)

    context_window_size -= max(0, len(tokenizer.encode(working_representation_str)))

    logger.info(
        "Constructed working representation:\n%s\n",
        working_representation_str,
    )

    # 2. Recent conversation history --------------------------------------------
    # If query is session-scoped, get recent conversation history from that session
    async with tracked_db("chat.get_context") as db:
        if session_name:
            recent_history = await summarizer.get_session_context_formatted(
                db,
                workspace_name=workspace_name,
                session_name=session_name,
                token_limit=context_window_size,
                include_summary=True,
            )
            logger.info("Retrieved recent conversation history")
        else:
            recent_history = None
            logger.info(
                "Query is not session-scoped, skipping recent conversation history"
            )

    context_window_size -= max(0, len(tokenizer.encode(recent_history or "")))

    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "tokens_used_estimate",
        settings.DIALECTIC.CONTEXT_WINDOW_SIZE - context_window_size,
        "tokens",
    )

    # 3. Peer card(s) ----------------------------------------------------------
    if settings.PEER_CARD.ENABLED:
        async with tracked_db("chat.get_peer_card") as db:
            peer_card = await crud.get_peer_card(
                db, workspace_name, observer=observer, observed=observed
            )
            if observer != observed:
                observed_peer_card = await crud.get_peer_card(
                    db, workspace_name, observer=observer, observed=observed
                )
            else:
                observed_peer_card = None

        if observed_peer_card:
            logger.info("Retrieved peer cards:\n%s\n%s", peer_card, observed_peer_card)
        else:
            logger.info("Retrieved peer card:\n%s", peer_card)
    else:
        peer_card = None
        observed_peer_card = None

    # 4. Dialectic call --------------------------------------------------------
    dialectic_call_start_time = time.perf_counter()
    if stream:
        return await dialectic_stream(
            query,
            working_representation_str,
            recent_history,
            peer_card,
            observed_peer_card,
            observer=observer,
            observed=observed,
        )

    response = await dialectic_call(
        query,
        working_representation_str,
        recent_history,
        peer_card,
        observed_peer_card,
        observer=observer,
        observed=observed,
    )
    dialectic_call_duration = (time.perf_counter() - dialectic_call_start_time) * 1000
    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "dialectic_call",
        dialectic_call_duration,
        "ms",
    )

    elapsed = (time.perf_counter() - start_time) * 1000

    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}", "total_duration", elapsed, "ms"
    )

    log_performance_metrics("dialectic_chat", dialectic_chat_uuid)
    # Convert AnthropicCallResponse to string for compatibility
    return str(response)
