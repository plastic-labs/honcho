"""
Core Dialectic Agent implementation.

This agent uses tools to gather context from the memory system
and synthesize responses to queries about a peer.
"""

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.config import settings
from src.dialectic import prompts
from src.utils.agent_tools import DIALECTIC_TOOLS, create_tool_executor, search_memory
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
    log_token_usage_metrics,
)

logger = logging.getLogger(__name__)


class DialecticAgent:
    """
    An agentic dialectic that iteratively gathers context to answer queries.

    Unlike the standard dialectic which pre-gathers all context before a single
    LLM call, this agent uses tools to strategically gather only the context
    needed to answer the specific query.
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_name: str,
        session_name: str | None,
        observer: str,
        observed: str,
        observer_peer_card: list[str] | None = None,
        observed_peer_card: list[str] | None = None,
        metric_key: str | None = None,
    ):
        """
        Initialize the dialectic agent.

        Args:
            db: Database session
            workspace_name: Workspace identifier
            session_name: Session identifier (may be None for global queries)
            observer: The peer making the query
            observed: The peer being queried about
            observer_peer_card: Biographical information about the observer
            observed_peer_card: Biographical information about the observed peer
            metric_key: Optional key for logging metrics (if provided, agent won't log separately)
        """
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.session_name: str | None = session_name
        self.observer: str = observer
        self.observed: str = observed
        self.observer_peer_card: list[str] | None = observer_peer_card
        self.observed_peer_card: list[str] | None = observed_peer_card
        self.metric_key: str | None = metric_key

        # Initialize conversation history with system prompt
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": prompts.agent_system_prompt(
                    observer, observed, observer_peer_card, observed_peer_card
                ),
            }
        ]
        self._session_history_initialized: bool = False

    async def _initialize_session_history(self) -> None:
        """Fetch and inject session history into the system prompt if configured."""
        if self._session_history_initialized:
            return
        self._session_history_initialized = True

        max_tokens = settings.DIALECTIC.SESSION_HISTORY_MAX_TOKENS
        if max_tokens == 0 or not self.session_name:
            return

        # Fetch recent messages up to the token limit
        stmt = await crud.get_messages(
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            token_limit=max_tokens,
            reverse=False,  # chronological order
        )
        result = await self.db.execute(stmt)
        messages = result.scalars().all()

        if not messages:
            return

        # Format messages for injection
        formatted_messages: list[str] = []
        for msg in messages:
            formatted = format_new_turn_with_timestamp(
                msg.content, msg.created_at, msg.peer_name
            )
            formatted_messages.append(formatted)

        session_history_section = (
            "\n\n## SESSION HISTORY\n\n"
            "The following is the recent conversation history from this session. "
            "Use this as immediate context when answering the query.\n\n"
            "<session_history>\n"
            f"{chr(10).join(formatted_messages)}\n"
            "</session_history>"
        )

        # Append session history to the system prompt
        self.messages[0]["content"] += session_history_section

    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        """
        Prefetch semantically relevant observations for the query.

        This provides immediate context to the agent without requiring
        tool calls, improving response quality and speed.

        Performs two separate searches to prevent retrieval dilution:
        - 25 explicit observations (direct facts from messages)
        - 25 derived observations (deductive, inductive, contradiction, vignette)

        Args:
            query: The user's query

        Returns:
            Formatted observations string or None if no observations found
        """
        try:
            # Search explicit observations separately
            explicit_repr = await search_memory(
                db=self.db,
                workspace_name=self.workspace_name,
                observer=self.observer,
                observed=self.observed,
                query=query,
                limit=25,
                levels=["explicit"],
            )

            # Search derived observations separately
            derived_repr = await search_memory(
                db=self.db,
                workspace_name=self.workspace_name,
                observer=self.observer,
                observed=self.observed,
                query=query,
                limit=25,
                levels=["deductive", "inductive", "contradiction", "vignette"],
            )

            if explicit_repr.is_empty() and derived_repr.is_empty():
                return None

            # Format as two separate sections
            parts: list[str] = []

            if not explicit_repr.is_empty():
                parts.append(explicit_repr.format_as_markdown(include_ids=False))

            if not derived_repr.is_empty():
                # Include IDs for derived so agent can use get_reasoning_chain
                parts.append(derived_repr.format_as_markdown(include_ids=True))

            return "\n".join(parts)

        except Exception as e:
            logger.warning(f"Failed to prefetch observations: {e}")
            return None

    async def answer(self, query: str) -> str:
        """
        Answer a query about the peer using agentic tool calling.

        The agent will:
        1. Receive the query
        2. Use tools to gather relevant context
        3. Synthesize a response grounded in the gathered context

        Args:
            query: The question to answer about the peer

        Returns:
            The synthesized answer string
        """
        # Initialize session history if configured
        await self._initialize_session_history()

        # Use provided metric_key or generate our own
        run_id: str | None = None
        if self.metric_key:
            task_name = self.metric_key
        else:
            run_id = str(uuid.uuid4())[:8]
            task_name = f"dialectic_chat_{run_id}"
        start_time = time.perf_counter()

        # Log input context
        accumulate_metric(
            task_name,
            "context",
            (
                f"workspace: {self.workspace_name}\n"
                f"session: {self.session_name or '(global)'}\n"
                f"observer: {self.observer}\n"
                f"observed: {self.observed}"
            ),
            "blob",
        )
        accumulate_metric(task_name, "query", query, "blob")

        # Prefetch relevant observations for the query
        prefetched_observations = await self._prefetch_relevant_observations(query)

        # Build the user message with prefetched context
        if prefetched_observations:
            user_content = (
                f"Query: {query}\n\n"
                f"## Relevant Observations (prefetched)\n"
                f"The following observations were found to be semantically relevant to your query. "
                f"Use these as primary context. You may still use tools to find additional information if needed.\n\n"
                f"{prefetched_observations}"
            )
            accumulate_metric(
                task_name, "prefetched_observations", prefetched_observations, "blob"
            )
        else:
            user_content = f"Query: {query}"

        # Add the query to conversation history
        self.messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        # Create tool executor with context
        tool_executor: Callable[[str, dict[str, Any]], Any] = create_tool_executor(
            db=self.db,
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            observer=self.observer,
            observed=self.observed,
            history_token_limit=settings.DIALECTIC.HISTORY_TOKEN_LIMIT,
        )

        # Run the agent loop
        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=settings.DIALECTIC,
            prompt="",  # Ignored since we pass messages
            max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
            tools=DIALECTIC_TOOLS,
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=settings.DIALECTIC.MAX_TOOL_ITERATIONS,
            messages=self.messages,
            track_name="Dialectic Agent",
            thinking_budget_tokens=settings.DIALECTIC.THINKING_BUDGET_TOKENS,
            max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
            trace_name="dialectic_chat",
        )

        # Log tool calls made with inputs and outputs
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )

        # Log thinking trace if present
        if response.thinking_content:
            accumulate_metric(task_name, "thinking", response.thinking_content, "blob")

        # Log token usage with cache awareness
        log_token_usage_metrics(
            task_name,
            response.input_tokens,
            response.output_tokens,
            response.cache_read_input_tokens,
            response.cache_creation_input_tokens,
        )
        accumulate_metric(task_name, "response", response.content, "blob")

        # Log timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", elapsed_ms, "ms")

        # Only log metrics here if we're not using a caller-provided metric_key
        if not self.metric_key and run_id is not None:
            log_performance_metrics("dialectic_chat", run_id)

        return response.content
