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

from dotenv import load_dotenv

from src import crud, prometheus
from src.config import settings
from src.dependencies import tracked_db
from src.utils import summarizer
from src.utils.clients import HonchoLLMCallStreamChunk, honcho_llm_call
from src.utils.logging import (
    accumulate_metric,
    conditional_observe,
    log_performance_metrics,
)
from src.utils.representation import Representation
from src.utils.tokens import estimate_tokens

from .agent import DialecticAgent
from .prompts import dialectic_prompt, estimate_dialectic_prompt_tokens

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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
    # Estimate input tokens by concatenating all inputs
    prompt_tokens = estimate_dialectic_prompt_tokens()
    inputs = [
        query,
        working_representation,
        recent_conversation_history or "",
        "\n".join(peer_card) if peer_card else "",
        "\n".join(observed_peer_card) if observed_peer_card else "",
    ]
    contextual_tokens = estimate_tokens("".join(inputs))
    estimated_input_tokens = prompt_tokens + contextual_tokens

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
        llm_settings=settings.DIALECTIC,
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

    # Track tokens in prometheus
    prometheus.DIALECTIC_TOKENS_PROCESSED.labels(
        token_type="input",  # nosec B106
    ).inc(estimated_input_tokens)

    prometheus.DIALECTIC_TOKENS_PROCESSED.labels(
        token_type="output",  # nosec B106
    ).inc(response.output_tokens)

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
    # Estimate input tokens by concatenating all inputs
    prompt_tokens = estimate_dialectic_prompt_tokens()
    variable_inputs = [
        query,
        working_representation,
        recent_conversation_history or "",
        "\n".join(peer_card) if peer_card else "",
        "\n".join(observed_peer_card) if observed_peer_card else "",
    ]
    variable_tokens = estimate_tokens("".join(variable_inputs))
    estimated_input_tokens = prompt_tokens + variable_tokens

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
        llm_settings=settings.DIALECTIC,
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

    # Track input tokens in prometheus
    # Note: Output tokens are available in the final chunk of the stream (is_done=True)
    prometheus.DIALECTIC_TOKENS_PROCESSED.labels(
        token_type="input",  # nosec B106
    ).inc(estimated_input_tokens)

    # Wrap the response to log output tokens from final chunk
    async def log_streaming_response():
        async for chunk in response:
            if chunk.is_done and chunk.output_tokens is not None:
                # TODO: Currently not tracking output tokens for groq models
                prometheus.DIALECTIC_TOKENS_PROCESSED.labels(
                    token_type="output",  # nosec B106
                ).inc(chunk.output_tokens)
            yield chunk

    return log_streaming_response()


@conditional_observe(name="Dialectic")
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

    context_window_size = (
        settings.DIALECTIC.CONTEXT_WINDOW_SIZE - 750
    )  # this is a hardcoded (accurate, slightly conservative) estimate of system prompt

    context_window_size -= estimate_tokens(query)

    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "query",
        f"{query}\n\nobserver: {observer}\nobserved: {observed}\n{f'session: {session_name}' if session_name else ''}",
        "blob",
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
    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "working_rep_explicit",
        len(working_representation.explicit),
        "count",
    )
    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "working_rep_deductive",
        len(working_representation.deductive),
        "count",
    )

    working_representation_str = str(working_representation)

    context_window_size -= max(0, estimate_tokens(working_representation_str))

    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "working_rep",
        working_representation_str,
        "blob",
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
        else:
            recent_history = None

    recent_history_tokens = estimate_tokens(recent_history or "")
    context_window_size -= max(0, recent_history_tokens)

    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "recent_history_tokens",
        recent_history_tokens,
        "tokens",
    )

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
            accumulate_metric(
                f"dialectic_chat_{dialectic_chat_uuid}",
                "peer_card",
                "\n".join(peer_card) if peer_card else "",
                "blob",
            )
            accumulate_metric(
                f"dialectic_chat_{dialectic_chat_uuid}",
                "observed_peer_card",
                "\n".join(observed_peer_card),
                "blob",
            )
        else:
            accumulate_metric(
                f"dialectic_chat_{dialectic_chat_uuid}",
                "peer_card",
                "\n".join(peer_card) if peer_card else "",
                "blob",
            )
    else:
        peer_card = None
        observed_peer_card = None

    # 4. Dialectic call --------------------------------------------------------
    dialectic_call_start_time = time.perf_counter()
    if stream:
        elapsed = (time.perf_counter() - start_time) * 1000
        accumulate_metric(
            f"dialectic_chat_{dialectic_chat_uuid}",
            "response",
            "(no logged response, streaming=true)",
            "blob",
        )
        accumulate_metric(
            f"dialectic_chat_{dialectic_chat_uuid}",
            "duration_to_streaming",
            elapsed,
            "ms",
        )
        log_performance_metrics("dialectic_chat", dialectic_chat_uuid)
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
        "response",
        response,
        "blob",
    )
    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "dialectic_call",
        dialectic_call_duration,
        "ms",
    )

    elapsed = (time.perf_counter() - start_time) * 1000

    log_performance_metrics("dialectic_chat", dialectic_chat_uuid)
    return response


@conditional_observe(name="Dialectic Agent")
async def agentic_chat(
    workspace_name: str,
    session_name: str | None,
    query: str,
    *,
    observer: str,
    observed: str,
) -> str:
    """
    Chat with the Agentic Dialectic API that dynamically gathers context.

    Unlike the standard dialectic which pre-gathers all context, the agentic
    dialectic uses tools to strategically gather only the context needed
    to answer the specific query.

    Args:
        workspace_name: Name of the workspace
        session_name: Optional session name for scoping
        query: Input Dialectic Query
        observer: Name of the peer making the query
        observed: Name of the peer being queried about

    Returns:
        Dialectic response string (no streaming support)
    """
    agentic_chat_uuid = str(uuid.uuid4())
    start_time = time.perf_counter()

    # Get peer cards upfront - these provide useful identity context for the agent
    observer_peer_card: list[str] | None = None
    observed_peer_card: list[str] | None = None

    if settings.PEER_CARD.ENABLED:
        async with tracked_db("agentic_chat.get_peer_card") as db:
            observer_peer_card = await crud.get_peer_card(
                db, workspace_name, observer=observer, observed=observer
            )
            if observer != observed:
                observed_peer_card = await crud.get_peer_card(
                    db, workspace_name, observer=observer, observed=observed
                )

    # Create and run the agentic dialectic
    metric_key = f"dialectic_chat_{agentic_chat_uuid}"
    async with tracked_db("agentic_chat.agent") as db:
        agent = DialecticAgent(
            db=db,
            workspace_name=workspace_name,
            session_name=session_name,
            observer=observer,
            observed=observed,
            observer_peer_card=observer_peer_card,
            observed_peer_card=observed_peer_card,
            metric_key=metric_key,
        )

        response = await agent.answer(query)

    elapsed = (time.perf_counter() - start_time) * 1000

    # Update total_duration to include peer card fetch time (overwrites agent's duration)
    accumulate_metric(metric_key, "total_duration", elapsed, "ms")

    log_performance_metrics("dialectic_chat", agentic_chat_uuid)
    return response
