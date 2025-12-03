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

from src.config import settings
from src.dialectic import prompts
from src.utils.agent_tools import DIALECTIC_TOOLS, create_tool_executor
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.logging import accumulate_metric, log_performance_metrics

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

        # Add the query to conversation history
        self.messages.append(
            {
                "role": "user",
                "content": f"Query: {query}",
            }
        )

        # Create tool executor with context
        tool_executor: Callable[[str, dict[str, Any]], Any] = create_tool_executor(
            db=self.db,
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            observer=self.observer,
            observed=self.observed,
        )

        # Run the agent loop
        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=settings.DIALECTIC,
            prompt="",  # Ignored since we pass messages
            max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
            tools=DIALECTIC_TOOLS,
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=20,
            messages=self.messages,
            track_name="Dialectic Agent",
            thinking_budget_tokens=settings.DIALECTIC.THINKING_BUDGET_TOKENS,
            max_input_tokens=settings.DIALECTIC.MAX_INPUT_TOKENS,
        )

        # Log tool calls made with inputs and outputs
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )
        # for i, tc in enumerate(response.tool_calls_made, 1):
        #     tool_name = tc.get("tool_name", "unknown")
        #     tool_input = tc.get("tool_input", {})
        #     tool_result = tc.get("tool_result", "")
        #     accumulate_metric(
        #         task_name,
        #         f"tool_{i}_{tool_name}",
        #         f"INPUT: {tool_input}\nOUTPUT: {tool_result}",
        #         "blob",
        #     )

        # Log thinking trace if present
        if response.thinking_content:
            accumulate_metric(task_name, "thinking", response.thinking_content, "blob")

        # Log output
        accumulate_metric(task_name, "input_tokens", response.input_tokens, "tokens")
        accumulate_metric(task_name, "output_tokens", response.output_tokens, "tokens")
        # Total tokens used for efficiency tracking
        tokens_used_estimate = response.input_tokens + response.output_tokens
        accumulate_metric(
            task_name, "tokens_used_estimate", tokens_used_estimate, "tokens"
        )
        accumulate_metric(task_name, "response", response.content, "blob")

        # Log timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", elapsed_ms, "ms")

        # Only log metrics here if we're not using a caller-provided metric_key
        if not self.metric_key and run_id is not None:
            log_performance_metrics("dialectic_chat", run_id)

        return response.content
