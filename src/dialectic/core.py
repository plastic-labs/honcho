"""
Core Dialectic Agent implementation.

This agent uses tools to gather context from the memory system
and synthesize responses to queries about a peer.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.config import ReasoningLevel, settings
from src.dialectic import prompts
from src.telemetry import prometheus_metrics
from src.telemetry.events import DialecticCompletedEvent, emit
from src.telemetry.logging import (
    accumulate_metric,
    log_performance_metrics,
    log_token_usage_metrics,
)
from src.telemetry.prometheus.metrics import DialecticComponents, TokenTypes
from src.utils.agent_tools import (
    DIALECTIC_TOOLS,
    DIALECTIC_TOOLS_MINIMAL,
    create_tool_executor,
    search_memory,
)
from src.utils.clients import (
    HonchoLLMCallResponse,
    StreamingResponseWithMetadata,
    honcho_llm_call,
)
from src.utils.formatting import format_new_turn_with_timestamp

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
        reasoning_level: ReasoningLevel = "low",
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
            reasoning_level: Level of reasoning to apply
        """
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.session_name: str | None = session_name
        self.observer: str = observer
        self.observed: str = observed
        self.observer_peer_card: list[str] | None = observer_peer_card
        self.observed_peer_card: list[str] | None = observed_peer_card
        self.metric_key: str | None = metric_key
        self.reasoning_level: ReasoningLevel = reasoning_level

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
        self._prefetched_conclusion_count: int = 0
        self._run_id: str = str(uuid.uuid4())[
            :8
        ]  # Always generate for event correlation

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
        - Explicit observations (produced by deriver)
        - Higher-level observations (produced in dreaming/background/chat)

        The number of observations fetched depends on reasoning level:
        - minimal: 10 of each type (reduced context for cost savings)
        - all others: 25 of each type

        Args:
            query: The user's query

        Returns:
            Formatted observations string or None if no observations found
        """
        # Use reduced prefetch for minimal reasoning to save tokens
        prefetch_limit = 10 if self.reasoning_level == "minimal" else 25

        try:
            # Search explicit observations separately
            explicit_repr = await search_memory(
                db=self.db,
                workspace_name=self.workspace_name,
                observer=self.observer,
                observed=self.observed,
                query=query,
                limit=prefetch_limit,
                levels=["explicit"],
            )

            # Search derived observations separately
            derived_repr = await search_memory(
                db=self.db,
                workspace_name=self.workspace_name,
                observer=self.observer,
                observed=self.observed,
                query=query,
                limit=prefetch_limit,
                levels=["deductive", "inductive", "contradiction"],
            )

            if explicit_repr.is_empty() and derived_repr.is_empty():
                return None

            # Count prefetched conclusions for telemetry
            explicit_count = len(explicit_repr.explicit) + len(explicit_repr.deductive)
            derived_count = len(derived_repr.explicit) + len(derived_repr.deductive)
            self._prefetched_conclusion_count = explicit_count + derived_count

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

    async def _prepare_query(
        self, query: str
    ) -> tuple[Callable[[str, dict[str, Any]], Any], str, str | None, float]:
        """
        Prepare common state for answering a query.

        Handles session history initialization, metrics setup, observation prefetching,
        user message construction, and tool executor creation.

        Args:
            query: The question to answer about the peer

        Returns:
            A tuple of (tool_executor, task_name, run_id, start_time)
        """
        await self._initialize_session_history()

        run_id: str | None = None
        if self.metric_key:
            task_name = self.metric_key
        else:
            run_id = str(uuid.uuid4())[:8]
            task_name = f"dialectic_chat_{run_id}"
        start_time = time.perf_counter()

        accumulate_metric(
            task_name,
            "context",
            (
                f"workspace: {self.workspace_name}\n"
                f"session: {self.session_name or '(global)'}\n"
                f"observer: {self.observer}\n"
                f"observed: {self.observed}\n"
                f"reasoning_level: {self.reasoning_level}"
            ),
            "blob",
        )
        accumulate_metric(task_name, "query", query, "blob")

        prefetched_observations = await self._prefetch_relevant_observations(query)

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

        self.messages.append({"role": "user", "content": user_content})

        tool_executor: Callable[
            [str, dict[str, Any]], Any
        ] = await create_tool_executor(
            db=self.db,
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            observer=self.observer,
            observed=self.observed,
            history_token_limit=settings.DIALECTIC.HISTORY_TOKEN_LIMIT,
            run_id=self._run_id,
            agent_type="dialectic",
            parent_category="dialectic",
        )

        return tool_executor, task_name, run_id, start_time

    def _log_response_metrics(
        self,
        task_name: str,
        run_id: str | None,
        start_time: float,
        response_content: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_input_tokens: int | None,
        cache_creation_input_tokens: int | None,
        tool_calls_count: int,
        thinking_content: str | None,
        iterations: int,
    ) -> None:
        """
        Log metrics common to both streaming and non-streaming responses.

        Args:
            task_name: Metrics task identifier
            run_id: Run identifier (None if using caller-provided metric_key)
            start_time: Start time from time.perf_counter()
            response_content: The full response text
            input_tokens: Input token count (actual from API)
            output_tokens: Output token count (actual from API)
            cache_read_input_tokens: Cache read tokens (if any)
            cache_creation_input_tokens: Cache creation tokens (if any)
            tool_calls_count: Number of tool calls made
            thinking_content: Thinking trace content (if any)
            iterations: Number of iterations in the tool execution loop
        """
        accumulate_metric(task_name, "tool_calls", tool_calls_count, "count")

        if thinking_content:
            accumulate_metric(task_name, "thinking", thinking_content, "blob")

        log_token_usage_metrics(
            task_name,
            input_tokens,
            output_tokens,
            cache_read_input_tokens or 0,
            cache_creation_input_tokens or 0,
        )
        accumulate_metric(task_name, "response", response_content, "blob")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", elapsed_ms, "ms")

        if not self.metric_key and run_id is not None:
            log_performance_metrics("dialectic_chat", run_id)

        # Prometheus metrics
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_dialectic_tokens(
                count=input_tokens,
                token_type=TokenTypes.INPUT.value,
                component=DialecticComponents.TOTAL.value,
                reasoning_level=self.reasoning_level,
            )
            prometheus_metrics.record_dialectic_tokens(
                count=output_tokens,
                token_type=TokenTypes.OUTPUT.value,
                component=DialecticComponents.TOTAL.value,
                reasoning_level=self.reasoning_level,
            )

        # Emit telemetry event
        emit(
            DialecticCompletedEvent(
                run_id=self._run_id,
                workspace_name=self.workspace_name,
                peer_name=self.observed,
                session_name=self.session_name,
                reasoning_level=self.reasoning_level,
                total_iterations=iterations,
                prefetched_conclusion_count=self._prefetched_conclusion_count,
                tool_calls_count=tool_calls_count,
                total_duration_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_input_tokens or 0,
                cache_creation_tokens=cache_creation_input_tokens or 0,
            )
        )

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
        tool_executor, task_name, run_id, start_time = await self._prepare_query(query)

        # Get level-specific settings
        level_settings = settings.DIALECTIC.LEVELS[self.reasoning_level]

        # Use minimal tools for minimal reasoning to reduce cost
        tools = (
            DIALECTIC_TOOLS_MINIMAL
            if self.reasoning_level == "minimal"
            else DIALECTIC_TOOLS
        )
        # Use level-specific max_output_tokens if set, otherwise global default
        max_tokens = (
            level_settings.MAX_OUTPUT_TOKENS
            if level_settings.MAX_OUTPUT_TOKENS is not None
            else settings.DIALECTIC.MAX_OUTPUT_TOKENS
        )

        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=level_settings,
            prompt="",  # Ignored since we pass messages
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=level_settings.TOOL_CHOICE,
            tool_executor=tool_executor,
            max_tool_iterations=level_settings.MAX_TOOL_ITERATIONS,
            messages=self.messages,
            track_name="Dialectic Agent",
            thinking_budget_tokens=level_settings.THINKING_BUDGET_TOKENS,
            max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
            trace_name="dialectic_chat",
        )

        self._log_response_metrics(
            task_name=task_name,
            run_id=run_id,
            start_time=start_time,
            response_content=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_input_tokens=response.cache_read_input_tokens,
            cache_creation_input_tokens=response.cache_creation_input_tokens,
            tool_calls_count=len(response.tool_calls_made),
            thinking_content=response.thinking_content,
            iterations=response.iterations,
        )

        return response.content

    async def answer_stream(self, query: str) -> AsyncIterator[str]:
        """
        Answer a query about the peer using agentic tool calling, streaming the response.

        The agent will:
        1. Receive the query
        2. Use tools to gather relevant context (non-streaming)
        3. Stream the synthesized response

        Args:
            query: The question to answer about the peer

        Yields:
            Chunks of the response text as they are generated
        """
        tool_executor, task_name, run_id, start_time = await self._prepare_query(query)

        # Get level-specific settings
        level_settings = settings.DIALECTIC.LEVELS[self.reasoning_level]

        # Use minimal tools for minimal reasoning to reduce cost
        tools = (
            DIALECTIC_TOOLS_MINIMAL
            if self.reasoning_level == "minimal"
            else DIALECTIC_TOOLS
        )
        # Use level-specific max_output_tokens if set, otherwise global default
        max_tokens = (
            level_settings.MAX_OUTPUT_TOKENS
            if level_settings.MAX_OUTPUT_TOKENS is not None
            else settings.DIALECTIC.MAX_OUTPUT_TOKENS
        )

        response = cast(
            StreamingResponseWithMetadata,
            await honcho_llm_call(
                llm_settings=level_settings,
                prompt="",  # Ignored since we pass messages
                max_tokens=max_tokens,
                stream=True,
                stream_final_only=True,
                tools=tools,
                tool_choice=level_settings.TOOL_CHOICE,
                tool_executor=tool_executor,
                max_tool_iterations=level_settings.MAX_TOOL_ITERATIONS,
                messages=self.messages,
                track_name="Dialectic Agent Stream",
                thinking_budget_tokens=level_settings.THINKING_BUDGET_TOKENS,
                max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
                trace_name="dialectic_chat",
            ),
        )

        accumulated_content: list[str] = []
        async for chunk in response:
            if chunk.content:
                accumulated_content.append(chunk.content)
                yield chunk.content

        self._log_response_metrics(
            task_name=task_name,
            run_id=run_id,
            start_time=start_time,
            response_content="".join(accumulated_content),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_input_tokens=response.cache_read_input_tokens,
            cache_creation_input_tokens=response.cache_creation_input_tokens,
            tool_calls_count=len(response.tool_calls_made),
            thinking_content=response.thinking_content,
            iterations=response.iterations,
        )
