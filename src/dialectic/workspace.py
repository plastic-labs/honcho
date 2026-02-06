"""
Workspace-level Dialectic Agent implementation.

This agent answers queries across ALL peers in a workspace,
providing an omniscient view that synthesizes across the entire
workspace's data.
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
    WORKSPACE_DIALECTIC_TOOLS,
    create_workspace_tool_executor,
)
from src.utils.clients import (
    HonchoLLMCallResponse,
    StreamingResponseWithMetadata,
    honcho_llm_call,
)
from src.utils.representation import format_documents_with_attribution

logger = logging.getLogger(__name__)


class WorkspaceDialecticAgent:
    """
    A workspace-level dialectic agent that answers queries across all peers.

    Unlike DialecticAgent which focuses on a single observer/observed pair,
    this agent has an omniscient view of all observations and conversations
    in the workspace.
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_name: str,
        session_name: str | None = None,
        reasoning_level: ReasoningLevel = "low",
    ):
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.session_name: str | None = session_name
        self.reasoning_level: ReasoningLevel = reasoning_level
        self._prefetched_conclusion_count: int = 0
        self._run_id: str = str(uuid.uuid4())[:8]

        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": prompts.workspace_agent_system_prompt(),
            }
        ]

    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        """
        Prefetch observations from across the workspace for the query.

        Args:
            query: The user's query

        Returns:
            Formatted observations string with attribution, or None
        """
        prefetch_limit = 10 if self.reasoning_level == "minimal" else 25

        try:
            documents = await crud.query_documents_workspace(
                db=self.db,
                workspace_name=self.workspace_name,
                query=query,
                top_k=prefetch_limit,
            )

            if not documents:
                return None

            self._prefetched_conclusion_count = len(documents)
            return format_documents_with_attribution(documents, include_ids=True)

        except Exception as e:
            logger.warning(f"Failed to prefetch workspace observations: {e}")
            return None

    async def _prepare_query(
        self, query: str
    ) -> tuple[Callable[[str, dict[str, Any]], Any], str, str | None, float]:
        """
        Prepare common state for answering a query.

        Returns:
            A tuple of (tool_executor, task_name, run_id, start_time)
        """
        run_id = str(uuid.uuid4())[:8]
        task_name = f"workspace_chat_{run_id}"
        start_time = time.perf_counter()

        accumulate_metric(
            task_name,
            "context",
            (
                f"workspace: {self.workspace_name}\n"
                f"session: {self.session_name or '(global)'}\n"
                f"reasoning_level: {self.reasoning_level}"
            ),
            "blob",
        )
        accumulate_metric(task_name, "query", query, "blob")

        prefetched_observations = await self._prefetch_relevant_observations(query)

        if prefetched_observations:
            user_content = (
                f"Query: {query}\n\n"
                f"## Relevant Observations (prefetched from across workspace)\n"
                f"The following observations were found to be semantically relevant to your query. "
                f"They span multiple peers. Use these as primary context. You may still use tools to find additional information.\n\n"
                f"{prefetched_observations}"
            )
            accumulate_metric(
                task_name, "prefetched_observations", prefetched_observations, "blob"
            )
        else:
            user_content = f"Query: {query}"

        self.messages.append({"role": "user", "content": user_content})

        tool_executor = await create_workspace_tool_executor(
            db=self.db,
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            history_token_limit=settings.DIALECTIC.HISTORY_TOKEN_LIMIT,
            run_id=self._run_id,
            agent_type="workspace_dialectic",
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

        if run_id is not None:
            log_performance_metrics("workspace_chat", run_id)

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

        emit(
            DialecticCompletedEvent(
                run_id=self._run_id,
                workspace_name=self.workspace_name,
                peer_name="(workspace)",
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
        Answer a workspace-level query using agentic tool calling.

        Args:
            query: The question to answer about the workspace

        Returns:
            The synthesized answer string
        """
        tool_executor, task_name, run_id, start_time = await self._prepare_query(query)

        level_settings = settings.DIALECTIC.LEVELS[self.reasoning_level]

        max_tokens = (
            level_settings.MAX_OUTPUT_TOKENS
            if level_settings.MAX_OUTPUT_TOKENS is not None
            else settings.DIALECTIC.MAX_OUTPUT_TOKENS
        )

        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=level_settings,
            prompt="",
            max_tokens=max_tokens,
            tools=WORKSPACE_DIALECTIC_TOOLS,
            tool_choice=level_settings.TOOL_CHOICE,
            tool_executor=tool_executor,
            max_tool_iterations=level_settings.MAX_TOOL_ITERATIONS,
            messages=self.messages,
            track_name="Workspace Dialectic Agent",
            thinking_budget_tokens=level_settings.THINKING_BUDGET_TOKENS,
            max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
            trace_name="workspace_chat",
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
        Stream an answer to a workspace-level query.

        Args:
            query: The question to answer about the workspace

        Yields:
            Chunks of the response text as they are generated
        """
        tool_executor, task_name, run_id, start_time = await self._prepare_query(query)

        level_settings = settings.DIALECTIC.LEVELS[self.reasoning_level]

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
                tools=WORKSPACE_DIALECTIC_TOOLS,
                tool_choice=level_settings.TOOL_CHOICE,
                tool_executor=tool_executor,
                max_tool_iterations=level_settings.MAX_TOOL_ITERATIONS,
                messages=self.messages,
                track_name="Workspace Dialectic Agent Stream",
                thinking_budget_tokens=level_settings.THINKING_BUDGET_TOKENS,
                max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
                trace_name="workspace_chat",
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
