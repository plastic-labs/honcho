"""
Base class for dialectic agents.

Extracts shared logic from DialecticAgent and WorkspaceDialecticAgent
using the template method pattern. Subclasses override abstract methods
and properties to customize behavior.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import ReasoningLevel, settings
from src.telemetry import prometheus_metrics
from src.telemetry.events import DialecticCompletedEvent, emit
from src.telemetry.logging import (
    accumulate_metric,
    log_performance_metrics,
    log_token_usage_metrics,
)
from src.telemetry.prometheus.metrics import DialecticComponents, TokenTypes
from src.utils.clients import (
    HonchoLLMCallResponse,
    StreamingResponseWithMetadata,
    honcho_llm_call,
)

logger = logging.getLogger(__name__)


class BaseDialecticAgent(ABC):
    """
    Base class for dialectic agents that gather context and answer queries.

    Subclasses must implement abstract methods/properties to customize
    tool selection, observation prefetching, and telemetry labels.
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_name: str,
        session_name: str | None,
        reasoning_level: ReasoningLevel,
        system_prompt: str,
    ):
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.session_name: str | None = session_name
        self.reasoning_level: ReasoningLevel = reasoning_level
        self._prefetched_conclusion_count: int = 0
        self._run_id: str = str(uuid.uuid4())[:8]

        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    # ------------------------------------------------------------------
    # Abstract methods — subclass MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        """Prefetch semantically relevant observations for the query."""
        ...

    @abstractmethod
    async def _create_tool_executor(self) -> Callable[[str, dict[str, Any]], Any]:
        """Create the tool executor for this agent type."""
        ...

    @abstractmethod
    def _get_tools(self) -> list[dict[str, Any]]:
        """Return the tool definitions for this agent type."""
        ...

    # ------------------------------------------------------------------
    # Abstract properties — subclass MUST implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _trace_name(self) -> str:
        """Langfuse trace name (e.g. 'dialectic_chat', 'workspace_chat')."""
        ...

    @property
    @abstractmethod
    def _track_name(self) -> str:
        """Langfuse track/generation name (e.g. 'Dialectic Agent')."""
        ...

    @property
    @abstractmethod
    def _telemetry_peer_name(self) -> str:
        """Peer name used in the DialecticCompletedEvent."""
        ...

    @property
    @abstractmethod
    def _prefetch_header(self) -> str:
        """Header text prepended to prefetched observations in the user message."""
        ...

    # ------------------------------------------------------------------
    # Overridable hooks — base provides defaults, subclass may override
    # ------------------------------------------------------------------

    async def _pre_prepare_query(self) -> None:  # noqa: B027
        """Hook called at the start of _prepare_query. Default: no-op."""

    def _get_task_name(self) -> tuple[str, str | None]:
        """Return (task_name, run_id). Default generates a fresh run_id."""
        run_id = str(uuid.uuid4())[:8]
        task_name = f"{self._trace_name}_{run_id}"
        return task_name, run_id

    def _get_context_string(self) -> str:
        """Return the context blob logged via accumulate_metric."""
        return (
            f"workspace: {self.workspace_name}\n"
            f"session: {self.session_name or '(global)'}\n"
            f"reasoning_level: {self.reasoning_level}"
        )

    def _should_log_performance_metrics(self, run_id: str | None) -> bool:
        """Whether to call log_performance_metrics at the end. Default: when run_id is set."""
        return run_id is not None

    # ------------------------------------------------------------------
    # Concrete methods — shared logic
    # ------------------------------------------------------------------

    async def _prepare_query(
        self, query: str
    ) -> tuple[Callable[[str, dict[str, Any]], Any], str, str | None, float]:
        await self._pre_prepare_query()

        task_name, run_id = self._get_task_name()
        start_time = time.perf_counter()

        accumulate_metric(task_name, "context", self._get_context_string(), "blob")
        accumulate_metric(task_name, "query", query, "blob")

        prefetched_observations = await self._prefetch_relevant_observations(query)

        if prefetched_observations:
            user_content = (
                f"Query: {query}\n\n"
                f"{self._prefetch_header}\n\n"
                f"{prefetched_observations}"
            )
            accumulate_metric(
                task_name, "prefetched_observations", prefetched_observations, "blob"
            )
        else:
            user_content = f"Query: {query}"

        self.messages.append({"role": "user", "content": user_content})

        tool_executor = await self._create_tool_executor()

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

        if self._should_log_performance_metrics(run_id) and run_id is not None:
            log_performance_metrics(self._trace_name, run_id)

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
                peer_name=self._telemetry_peer_name,
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
        """Answer a query using agentic tool calling."""
        tool_executor, task_name, run_id, start_time = await self._prepare_query(query)

        level_settings = settings.DIALECTIC.LEVELS[self.reasoning_level]
        tools = self._get_tools()

        max_tokens = (
            level_settings.MAX_OUTPUT_TOKENS
            if level_settings.MAX_OUTPUT_TOKENS is not None
            else settings.DIALECTIC.MAX_OUTPUT_TOKENS
        )

        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=level_settings,
            prompt="",
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=level_settings.TOOL_CHOICE,
            tool_executor=tool_executor,
            max_tool_iterations=level_settings.MAX_TOOL_ITERATIONS,
            messages=self.messages,
            track_name=self._track_name,
            thinking_budget_tokens=level_settings.THINKING_BUDGET_TOKENS,
            max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
            trace_name=self._trace_name,
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
        """Answer a query using agentic tool calling, streaming the response."""
        tool_executor, task_name, run_id, start_time = await self._prepare_query(query)

        level_settings = settings.DIALECTIC.LEVELS[self.reasoning_level]
        tools = self._get_tools()

        max_tokens = (
            level_settings.MAX_OUTPUT_TOKENS
            if level_settings.MAX_OUTPUT_TOKENS is not None
            else settings.DIALECTIC.MAX_OUTPUT_TOKENS
        )

        response = cast(
            StreamingResponseWithMetadata,
            await honcho_llm_call(
                llm_settings=level_settings,
                prompt="",
                max_tokens=max_tokens,
                stream=True,
                stream_final_only=True,
                tools=tools,
                tool_choice=level_settings.TOOL_CHOICE,
                tool_executor=tool_executor,
                max_tool_iterations=level_settings.MAX_TOOL_ITERATIONS,
                messages=self.messages,
                track_name=f"{self._track_name} Stream",
                thinking_budget_tokens=level_settings.THINKING_BUDGET_TOKENS,
                max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
                trace_name=self._trace_name,
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
