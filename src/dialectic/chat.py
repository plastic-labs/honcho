"""
Main dialectic system for AI-powered context synthesis and user representation.

The Dialectic class provides a natural language API for AI applications to query
and understand users through context synthesis of working representations and
historical observations.
"""

import asyncio
import logging
import uuid

import tiktoken
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context
from mirascope.llm import Stream

from src import crud
from src.config import settings
from src.dependencies import tracked_db
from src.routers.sessions import get_session_context
from src.utils.clients import honcho_llm_call
from src.utils.embedding_store import EmbeddingStore
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
)

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
    working_representation: str | None,
    recent_conversation_history: str | None,
    additional_context: str | None,
    peer_name: str,
    peer_card: str | None,
    target_name: str | None = None,
    target_peer_card: str | None = None,
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
        query,
        working_representation,
        recent_conversation_history,
        additional_context,
        peer_name,
        peer_card,
        target_name,
        target_peer_card,
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
    working_representation: str | None,
    recent_conversation_history: str | None,
    additional_context: str | None,
    peer_name: str,
    peer_card: str | None,
    target_name: str | None = None,
    target_peer_card: str | None = None,
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
        query,
        working_representation,
        recent_conversation_history,
        additional_context,
        peer_name,
        peer_card,
        target_name,
        target_peer_card,
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

    dialectic_chat_uuid = str(uuid.uuid4())

    tokenizer = tiktoken.get_encoding("cl100k_base")

    context_window_size = (
        settings.DIALECTIC.CONTEXT_WINDOW_SIZE - 750
    )  # this is a hardcoded (accurate, slightly conservative) estimate of system prompt

    context_window_size -= len(tokenizer.encode(query))

    if settings.LANGFUSE_PUBLIC_KEY:
        langfuse_context.update_current_trace(
            metadata={
                "query_generation_model": settings.DIALECTIC.QUERY_GENERATION_MODEL,
                "query_generation_provider": settings.DIALECTIC.QUERY_GENERATION_PROVIDER,
                "dialectic_model": settings.DIALECTIC.MODEL,
            }
        )
    logger.info(
        "Received query:\n'%s'\nobserver: %s%s%s\n",
        query,
        peer_name,
        f", target: {target_name}" if target_name else "",
        f", session: {session_name}" if session_name else "",
    )
    start_time = asyncio.get_event_loop().time()

    # 1. Working representation (short-term) -----------------------------------
    # Only useful for session-scoped queries, not global queries
    if session_name:
        working_rep_start_time = asyncio.get_event_loop().time()
        async with tracked_db("chat.get_working_representation") as db:
            # If no target specified, get global representation (peer observing themselves)
            target_peer = target_name if target_name is not None else peer_name
            working_representation = await crud.get_working_representation(
                db, workspace_name, peer_name, target_peer, session_name
            )
        working_rep_duration = asyncio.get_event_loop().time() - working_rep_start_time
        accumulate_metric(
            f"dialectic_chat_{dialectic_chat_uuid}",
            "retrieve_working_rep",
            working_rep_duration,
            "s",
        )
        logger.info("Retrieved working representation:\n%s\n", working_representation)
        context_window_size -= len(tokenizer.encode(working_representation))
    else:
        # For global queries, working representation isn't useful - use historical context instead
        working_representation = None
        logger.info("Query is not session-scoped, skipping working representation")

    # 2. Additional context (long-term semantic search) ------------------------
    # If the query is not targeted, get global_representation facts from other sessions
    # If the query is targeted, get facts from other sessions for our target
    additional_context_start_time = asyncio.get_event_loop().time()
    embedding_store = EmbeddingStore(
        workspace_name=workspace_name,
        peer_name=target_name if target_name else peer_name,
        collection_name="global_representation"
        if not target_name
        else crud.construct_collection_name(observer=peer_name, observed=target_name),
    )
    additional_context: str = await get_observations(
        query,
        target_name if target_name else peer_name,
        embedding_store,
        include_premises=True,
    )
    additional_context_duration = (
        asyncio.get_event_loop().time() - additional_context_start_time
    )
    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "retrieve_additional_context",
        additional_context_duration,
        "s",
    )

    logger.info("Retrieved additional context:\n%s", additional_context)
    context_window_size -= len(tokenizer.encode(additional_context))

    # 3. Recent conversation history --------------------------------------------
    # If query is session-scoped, get recent conversation history from that session
    if session_name:
        async with tracked_db("chat.get_session_context") as db:
            session_context = await get_session_context(
                workspace_id=workspace_name,
                session_id=session_name,
                tokens=context_window_size,
                summary=True,
                db=db,
            )
        logger.info(
            "Retrieved recent conversation history with %s messages",
            len(session_context.messages),
        )
        recent_conversation_history = f"""
        <summary>
        {session_context.summary}
        </summary>
        <recent_messages>
        {session_context.messages}
        </recent_messages>
        """
    else:
        recent_conversation_history = None
        logger.info("Query is not session-scoped, skipping recent conversation history")

    context_window_size -= len(tokenizer.encode(recent_conversation_history or ""))

    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "tokens_used_estimate",
        settings.DIALECTIC.CONTEXT_WINDOW_SIZE - context_window_size,
        "tokens",
    )

    # 4. Peer card(s) ----------------------------------------------------------
    async with tracked_db("chat.get_peer_card") as db:
        peer_card = await crud.get_peer_card(db, workspace_name, peer_name)
        if target_name:
            target_peer_card = await crud.get_peer_card(db, workspace_name, target_name)
        else:
            target_peer_card = None

    logger.info(
        "Retrieved peer cards:\n%s\n%s",
        peer_card,
        target_peer_card if target_peer_card else "",
    )

    # 5. Dialectic call --------------------------------------------------------
    dialectic_call_start_time = asyncio.get_event_loop().time()
    if stream:
        return await dialectic_stream(
            query,
            working_representation,
            recent_conversation_history,
            additional_context,
            peer_name,
            peer_card,
            target_name,
            target_peer_card,
        )

    response = await dialectic_call(
        query,
        working_representation,
        recent_conversation_history,
        additional_context,
        peer_name,
        peer_card,
        target_name,
        target_peer_card,
    )
    dialectic_call_duration = (
        asyncio.get_event_loop().time() - dialectic_call_start_time
    )
    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}",
        "dialectic_call",
        dialectic_call_duration,
        "s",
    )

    elapsed = asyncio.get_event_loop().time() - start_time

    accumulate_metric(
        f"dialectic_chat_{dialectic_chat_uuid}", "total_duration", elapsed, "s"
    )

    log_performance_metrics(f"dialectic_chat_{dialectic_chat_uuid}")
    # Convert AnthropicCallResponse to string for compatibility
    return str(response)
