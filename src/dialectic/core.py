"""
Core Dialectic Agent implementation.

This agent uses tools to gather context from the memory system
and synthesize responses to queries about a peer.
"""

import asyncio
import json
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
from src.telemetry.events import DialecticCompletedEvent, DialecticPhaseMetrics, emit
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

    def _stringify_tool_result_content(self, content: Any) -> str:
        """
        Convert a tool result payload into a stable, human-readable string.

        Tool results can vary by provider and may include nested content blocks
        (e.g., Anthropic-style lists with text/image/attachment blocks). The
        synthesis prompt needs a consistent text representation that preserves
        all information without assuming a single schema.

        Args:
            content: Tool result payload (string, dict, list, or arbitrary object).

        Returns:
            A best-effort string representation suitable for inclusion in the
            synthesis context.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        try:
            if isinstance(content, list):
                rendered_parts: list[str] = []
                for raw_item in cast(list[Any], content):
                    item: Any = raw_item
                    if isinstance(item, dict):
                        item_dict = cast(dict[str, Any], item)
                        if item_dict.get("type") == "text":
                            text = item_dict.get("text", "")
                            if isinstance(text, str) and text:
                                rendered_parts.append(text)
                                continue

                    rendered_parts.append(
                        json.dumps(item, ensure_ascii=False, default=str)
                    )
                return "\n".join(part for part in rendered_parts if part)

            if isinstance(content, dict):
                return json.dumps(
                    cast(dict[str, Any], content), ensure_ascii=False, default=str
                )

            return str(cast(object, content))
        except Exception:
            return str(cast(object, content))

    def _build_synthesis_messages(
        self,
        search_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Build messages for synthesis model in two-phase dialectic.

        Takes the full search conversation (including tool calls and results)
        and serializes them into a text-based context that any provider can handle.

        Args:
            search_messages: Full conversation history from search phase

        Returns:
            Messages list for synthesis model
        """
        # Extract system message and serialize the rest into text
        system_content = ""
        search_context_parts: list[str] = []

        for msg in search_messages:
            role = msg.get("role", "")

            if role == "system":
                # Preserve system message content
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_content: str = content
                elif isinstance(content, list):
                    # Anthropic-style content blocks
                    for block in content:  # pyright: ignore[reportUnknownVariableType]
                        if isinstance(block, dict) and block.get("type") == "text":  # pyright: ignore[reportUnknownMemberType]
                            system_content += cast(str, block.get("text", ""))  # pyright: ignore[reportUnknownMemberType]

            elif role == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    search_context_parts.append(f"[USER]: {content}")
                elif isinstance(content, list):
                    # Anthropic-style content blocks (text + tool results)
                    for block in content:  # pyright: ignore[reportUnknownVariableType]
                        if not isinstance(block, dict):
                            continue
                        block_dict = cast(dict[str, Any], block)

                        if block_dict.get("type") == "text":
                            text_any = block_dict.get("text")
                            if isinstance(text_any, str) and text_any:
                                search_context_parts.append(f"[USER]: {text_any}")
                            continue

                        if block_dict.get("type") == "tool_result":
                            tool_id: str = cast(
                                str,
                                block_dict.get("tool_use_id", "unknown"),
                            )
                            result = self._stringify_tool_result_content(
                                block_dict.get("content")
                            )
                            if result:
                                search_context_parts.append(
                                    f"[TOOL RESULT ({tool_id})]: {result}"
                                )

            elif role == "assistant":
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                # Handle text content
                if isinstance(content, str) and content:
                    search_context_parts.append(f"[ASSISTANT]: {content}")
                elif isinstance(content, list):
                    # Anthropic-style content blocks
                    for block in content:  # pyright: ignore[reportUnknownVariableType]
                        if isinstance(block, dict):
                            if block.get("type") == "text":  # pyright: ignore[reportUnknownMemberType]
                                text: str = cast(str, block.get("text", ""))  # pyright: ignore[reportUnknownMemberType]
                                if text:
                                    search_context_parts.append(f"[ASSISTANT]: {text}")
                            elif block.get("type") == "tool_use":  # pyright: ignore[reportUnknownMemberType]
                                name: str = cast(str, block.get("name", "unknown"))  # pyright: ignore[reportUnknownMemberType]
                                tool_input: Any = block.get("input", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                                search_context_parts.append(
                                    f"[TOOL CALL: {name}]: {json.dumps(tool_input)}"
                                )

                # Handle OpenAI-style tool_calls
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    search_context_parts.append(f"[TOOL CALL: {name}]: {args}")

            elif role == "tool":
                # OpenAI-style tool result
                tool_id = msg.get("tool_call_id", "unknown")
                content = msg.get("content", "")
                search_context_parts.append(f"[TOOL RESULT ({tool_id})]: {content}")

        # Build the synthesis messages
        messages_out: list[dict[str, Any]] = []

        # Include system prompt if present
        if system_content:
            messages_out.append({"role": "system", "content": system_content})

        # Add the search context as a user message
        search_context = "\n\n".join(search_context_parts)
        synthesis_prompt = (
            f"The following is the search process used to gather information:\n\n"
            f"---\n{search_context}\n---\n\n"
            f"Based on the information gathered above, provide your "
            f"final response to the original query. Be direct and helpful."
        )

        messages_out.append({"role": "user", "content": synthesis_prompt})

        return messages_out

    def _log_two_phase_metrics(
        self,
        task_name: str,
        run_id: str | None,
        start_time: float,
        response_content: str,
        search_input_tokens: int,
        search_output_tokens: int,
        search_cache_read_tokens: int,
        search_cache_creation_tokens: int,
        search_tool_calls_count: int,
        search_iterations: int,
        synthesis_input_tokens: int,
        synthesis_output_tokens: int,
        synthesis_cache_read_tokens: int,
        synthesis_cache_creation_tokens: int,
        synthesis_thinking_content: str | None,
    ) -> None:
        """
        Log metrics for two-phase dialectic (search + synthesis).

        Args:
            task_name: Metrics task identifier
            run_id: Run identifier (None if using caller-provided metric_key)
            start_time: Start time from time.perf_counter()
            response_content: The full response text
            search_*: Metrics from search phase
            synthesis_*: Metrics from synthesis phase
        """
        # Total metrics
        total_input_tokens = search_input_tokens + synthesis_input_tokens
        total_output_tokens = search_output_tokens + synthesis_output_tokens
        total_cache_read_tokens = search_cache_read_tokens + synthesis_cache_read_tokens
        total_cache_creation_tokens = (
            search_cache_creation_tokens + synthesis_cache_creation_tokens
        )

        accumulate_metric(task_name, "tool_calls", search_tool_calls_count, "count")
        accumulate_metric(task_name, "search_iterations", search_iterations, "count")

        if synthesis_thinking_content:
            accumulate_metric(
                task_name, "synthesis_thinking", synthesis_thinking_content, "blob"
            )

        log_token_usage_metrics(
            task_name,
            total_input_tokens,
            total_output_tokens,
            total_cache_read_tokens,
            total_cache_creation_tokens,
        )

        # Log phase-specific token metrics for cost analysis
        accumulate_metric(
            task_name, "search_input_tokens", search_input_tokens, "tokens"
        )
        accumulate_metric(
            task_name, "search_output_tokens", search_output_tokens, "tokens"
        )
        accumulate_metric(
            task_name, "synthesis_input_tokens", synthesis_input_tokens, "tokens"
        )
        accumulate_metric(
            task_name, "synthesis_output_tokens", synthesis_output_tokens, "tokens"
        )

        accumulate_metric(task_name, "response", response_content, "blob")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", elapsed_ms, "ms")

        if not self.metric_key and run_id is not None:
            log_performance_metrics("dialectic_chat", run_id)

        # Get model/provider info for phase metrics
        level_settings = settings.DIALECTIC.LEVELS[self.reasoning_level]
        synthesis_settings = level_settings.SYNTHESIS

        # Build phase metrics
        phases = [
            DialecticPhaseMetrics(
                phase_name="search",
                provider=level_settings.PROVIDER,
                model=level_settings.MODEL,
                input_tokens=search_input_tokens,
                output_tokens=search_output_tokens,
                cache_read_tokens=search_cache_read_tokens,
                cache_creation_tokens=search_cache_creation_tokens,
                iterations=search_iterations,
                tool_calls_count=search_tool_calls_count,
            ),
            DialecticPhaseMetrics(
                phase_name="synthesis",
                provider=synthesis_settings.PROVIDER if synthesis_settings else None,
                model=synthesis_settings.MODEL if synthesis_settings else None,
                input_tokens=synthesis_input_tokens,
                output_tokens=synthesis_output_tokens,
                cache_read_tokens=synthesis_cache_read_tokens,
                cache_creation_tokens=synthesis_cache_creation_tokens,
                iterations=1,  # Synthesis is always 1 iteration
                tool_calls_count=0,  # No tools in synthesis
            ),
        ]

        # Emit telemetry event with total iterations (search + 1 for synthesis)
        emit(
            DialecticCompletedEvent(
                run_id=self._run_id,
                workspace_name=self.workspace_name,
                peer_name=self.observed,
                session_name=self.session_name,
                reasoning_level=self.reasoning_level,
                two_phase_mode=True,
                total_iterations=search_iterations + 1,
                prefetched_conclusion_count=self._prefetched_conclusion_count,
                tool_calls_count=search_tool_calls_count,
                total_duration_ms=elapsed_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_read_tokens=total_cache_read_tokens,
                cache_creation_tokens=total_cache_creation_tokens,
                phases=phases,
            )
        )

    async def answer(self, query: str) -> str:
        """
        Answer a query about the peer using agentic tool calling.

        Supports two modes:
        1. Single-model mode: One model handles both tool calling and synthesis
        2. Two-model mode: Search model handles tool calling, synthesis model generates response

        The agent will:
        1. Receive the query
        2. Use tools to gather relevant context (search phase)
        3. Synthesize a response grounded in the gathered context (synthesis phase)

        Args:
            query: The question to answer about the peer

        Returns:
            The synthesized answer string
        """
        tool_executor, task_name, run_id, start_time = await self._prepare_query(query)

        # Get level-specific settings
        level_settings = settings.DIALECTIC.LEVELS[self.reasoning_level]
        synthesis_settings = level_settings.SYNTHESIS

        # Use minimal tools for minimal reasoning to reduce cost
        tools = (
            DIALECTIC_TOOLS_MINIMAL
            if self.reasoning_level == "minimal"
            else DIALECTIC_TOOLS
        )

        # Check if two-phase mode is enabled (synthesis config exists and not minimal)
        use_two_phase = (
            synthesis_settings is not None and self.reasoning_level != "minimal"
        )

        if not use_two_phase:
            # Single-model path (original behavior)
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

        # Two-phase mode: Search then Synthesis
        # Type narrowing: synthesis_settings is guaranteed non-None in two-phase mode
        assert synthesis_settings is not None  # nosec B101

        # Phase 1: Search (non-streaming, with tools)
        search_max_tokens = (
            level_settings.MAX_OUTPUT_TOKENS
            if level_settings.MAX_OUTPUT_TOKENS is not None
            else 1024  # Lower default for search - just tool call JSON
        )

        try:
            search_response: HonchoLLMCallResponse[str] = await honcho_llm_call(
                llm_settings=level_settings,
                prompt="",
                max_tokens=search_max_tokens,
                tools=tools,
                tool_choice=level_settings.TOOL_CHOICE,
                tool_executor=tool_executor,
                max_tool_iterations=level_settings.MAX_TOOL_ITERATIONS,
                messages=self.messages,
                track_name="Dialectic Search",
                thinking_budget_tokens=level_settings.THINKING_BUDGET_TOKENS,
                max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
                trace_name="dialectic_search",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Fallback to single-model on search failure
            logger.warning(f"Search phase failed: {e}, falling back to single-model")
            max_tokens = (
                synthesis_settings.MAX_OUTPUT_TOKENS
                if synthesis_settings.MAX_OUTPUT_TOKENS is not None
                else settings.DIALECTIC.MAX_OUTPUT_TOKENS
            )
            response = await honcho_llm_call(
                llm_settings=synthesis_settings,
                prompt="",
                max_tokens=max_tokens,
                tools=None,  # No tools for fallback
                tool_executor=None,
                max_tool_iterations=1,
                messages=self.messages,
                track_name="Dialectic Fallback",
                thinking_budget_tokens=synthesis_settings.THINKING_BUDGET_TOKENS,
                max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
                trace_name="dialectic_fallback",
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
                tool_calls_count=0,
                thinking_content=response.thinking_content,
                iterations=1,
            )
            return response.content

        # Phase 2: Synthesis (non-streaming, no tools)
        synthesis_messages = self._build_synthesis_messages(search_response.messages)
        synthesis_max_tokens = (
            synthesis_settings.MAX_OUTPUT_TOKENS
            if synthesis_settings.MAX_OUTPUT_TOKENS is not None
            else settings.DIALECTIC.MAX_OUTPUT_TOKENS
        )

        synthesis_response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=synthesis_settings,
            prompt="",
            max_tokens=synthesis_max_tokens,
            tools=None,  # No tools for synthesis
            tool_executor=None,
            max_tool_iterations=1,
            messages=synthesis_messages,
            track_name="Dialectic Synthesis",
            thinking_budget_tokens=synthesis_settings.THINKING_BUDGET_TOKENS,
            max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
            trace_name="dialectic_synthesis",
        )

        # Log combined metrics for two-phase
        self._log_two_phase_metrics(
            task_name=task_name,
            run_id=run_id,
            start_time=start_time,
            response_content=synthesis_response.content,
            search_input_tokens=search_response.input_tokens,
            search_output_tokens=search_response.output_tokens,
            search_cache_read_tokens=search_response.cache_read_input_tokens,
            search_cache_creation_tokens=search_response.cache_creation_input_tokens,
            search_tool_calls_count=len(search_response.tool_calls_made),
            search_iterations=search_response.iterations,
            synthesis_input_tokens=synthesis_response.input_tokens,
            synthesis_output_tokens=synthesis_response.output_tokens,
            synthesis_cache_read_tokens=synthesis_response.cache_read_input_tokens,
            synthesis_cache_creation_tokens=synthesis_response.cache_creation_input_tokens,
            synthesis_thinking_content=synthesis_response.thinking_content,
        )

        return synthesis_response.content

    async def answer_stream(self, query: str) -> AsyncIterator[str]:
        """
        Answer a query about the peer using agentic tool calling, streaming the response.

        Supports two modes:
        1. Single-model mode: One model handles both tool calling and synthesis (streams final)
        2. Two-model mode: Search model handles tool calling, synthesis model streams response

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
        synthesis_settings = level_settings.SYNTHESIS

        # Use minimal tools for minimal reasoning to reduce cost
        tools = (
            DIALECTIC_TOOLS_MINIMAL
            if self.reasoning_level == "minimal"
            else DIALECTIC_TOOLS
        )

        # Check if two-phase mode is enabled (synthesis config exists and not minimal)
        use_two_phase = (
            synthesis_settings is not None and self.reasoning_level != "minimal"
        )

        if not use_two_phase:
            # Single-model path (original behavior - stream final response)
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
            return

        # Two-phase mode: Search (non-streaming) then Synthesis (streaming)
        # Type narrowing: synthesis_settings is guaranteed non-None in two-phase mode
        assert synthesis_settings is not None  # nosec B101

        # Phase 1: Search (non-streaming, with tools)
        search_max_tokens = (
            level_settings.MAX_OUTPUT_TOKENS
            if level_settings.MAX_OUTPUT_TOKENS is not None
            else 1024  # Lower default for search
        )

        try:
            search_response: HonchoLLMCallResponse[str] = await honcho_llm_call(
                llm_settings=level_settings,
                prompt="",
                max_tokens=search_max_tokens,
                tools=tools,
                tool_choice=level_settings.TOOL_CHOICE,
                tool_executor=tool_executor,
                max_tool_iterations=level_settings.MAX_TOOL_ITERATIONS,
                messages=self.messages,
                track_name="Dialectic Search Stream",
                thinking_budget_tokens=level_settings.THINKING_BUDGET_TOKENS,
                max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
                trace_name="dialectic_search",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Fallback to single-model streaming on search failure
            logger.warning(
                f"Search phase failed: {e}, falling back to single-model streaming"
            )
            max_tokens = (
                synthesis_settings.MAX_OUTPUT_TOKENS
                if synthesis_settings.MAX_OUTPUT_TOKENS is not None
                else settings.DIALECTIC.MAX_OUTPUT_TOKENS
            )
            fallback_response = cast(
                StreamingResponseWithMetadata,
                await honcho_llm_call(
                    llm_settings=synthesis_settings,
                    prompt="",
                    max_tokens=max_tokens,
                    stream=True,
                    tools=None,
                    tool_executor=None,
                    max_tool_iterations=1,
                    messages=self.messages,
                    track_name="Dialectic Fallback Stream",
                    thinking_budget_tokens=synthesis_settings.THINKING_BUDGET_TOKENS,
                    max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
                    trace_name="dialectic_fallback",
                ),
            )
            accumulated_content = []
            async for chunk in fallback_response:
                if chunk.content:
                    accumulated_content.append(chunk.content)
                    yield chunk.content
            self._log_response_metrics(
                task_name=task_name,
                run_id=run_id,
                start_time=start_time,
                response_content="".join(accumulated_content),
                input_tokens=fallback_response.input_tokens,
                output_tokens=fallback_response.output_tokens,
                cache_read_input_tokens=fallback_response.cache_read_input_tokens,
                cache_creation_input_tokens=fallback_response.cache_creation_input_tokens,
                tool_calls_count=0,
                thinking_content=fallback_response.thinking_content,
                iterations=1,
            )
            return

        # Phase 2: Synthesis (streaming, no tools)
        synthesis_messages = self._build_synthesis_messages(search_response.messages)
        synthesis_max_tokens = (
            synthesis_settings.MAX_OUTPUT_TOKENS
            if synthesis_settings.MAX_OUTPUT_TOKENS is not None
            else settings.DIALECTIC.MAX_OUTPUT_TOKENS
        )

        synthesis_stream = cast(
            StreamingResponseWithMetadata,
            await honcho_llm_call(
                llm_settings=synthesis_settings,
                prompt="",
                max_tokens=synthesis_max_tokens,
                stream=True,
                tools=None,  # No tools for synthesis
                tool_executor=None,
                max_tool_iterations=1,
                messages=synthesis_messages,
                track_name="Dialectic Synthesis Stream",
                thinking_budget_tokens=synthesis_settings.THINKING_BUDGET_TOKENS,
                max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
                trace_name="dialectic_synthesis",
            ),
        )

        accumulated_content = []
        async for chunk in synthesis_stream:
            if chunk.content:
                accumulated_content.append(chunk.content)
                yield chunk.content

        # Log combined metrics for two-phase
        self._log_two_phase_metrics(
            task_name=task_name,
            run_id=run_id,
            start_time=start_time,
            response_content="".join(accumulated_content),
            search_input_tokens=search_response.input_tokens,
            search_output_tokens=search_response.output_tokens,
            search_cache_read_tokens=search_response.cache_read_input_tokens,
            search_cache_creation_tokens=search_response.cache_creation_input_tokens,
            search_tool_calls_count=len(search_response.tool_calls_made),
            search_iterations=search_response.iterations,
            synthesis_input_tokens=synthesis_stream.input_tokens,
            synthesis_output_tokens=synthesis_stream.output_tokens,
            synthesis_cache_read_tokens=synthesis_stream.cache_read_input_tokens,
            synthesis_cache_creation_tokens=synthesis_stream.cache_creation_input_tokens,
            synthesis_thinking_content=synthesis_stream.thinking_content,
        )
