import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.deriver.agent import prompts
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.utils.agent_tools import DERIVER_TOOLS, create_tool_executor
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
    log_token_usage_metrics,
)

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        db: AsyncSession,
        workspace_name: str,
        session_name: str,
        configuration: schemas.ResolvedConfiguration,
        observer: str,
        observed: str,
        observed_peer_card: list[str] | None = None,
    ):
        self.db: AsyncSession = db
        self.workspace_name: str = workspace_name
        self.session_name: str = session_name
        self.configuration: ResolvedConfiguration = configuration
        self.observer: str = observer
        self.observed: str = observed
        self.observed_peer_card: list[str] | None = observed_peer_card
        self._current_messages: list[Message] = []

        # Only include peer card in prompt if use is enabled
        prompt_peer_card: list[str] | None = (
            observed_peer_card if configuration.peer_card.use is not False else None
        )

        # Build tools list based on configuration
        self._tools: list[dict[str, Any]] = [
            tool
            for tool in DERIVER_TOOLS
            if not (
                tool["name"] == "update_peer_card"
                and configuration.peer_card.create is False
            )
        ]

        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": prompts.agent_system_prompt(
                    observer, observed, prompt_peer_card
                ),
            }
        ]

    async def run_loop(self, messages: list[models.Message]) -> None:
        """
        Run the agent loop for a batch of messages using native tool calling.

        Args:
            messages: List of messages to process in this agent run
        """
        if not messages:
            logger.warning("run_loop called with empty message list")
            return

        # Generate unique ID for this run
        run_id = str(uuid.uuid4())[:8]
        task_name = f"deriver_agent_{run_id}"
        start_time = time.perf_counter()

        # Log input context
        accumulate_metric(
            task_name,
            "context",
            (
                f"workspace: {self.workspace_name}\n"
                f"session: {self.session_name}\n"
                f"observer: {self.observer}\n"
                f"observed: {self.observed}"
            ),
            "blob",
        )
        accumulate_metric(task_name, "message_count", len(messages), "count")

        # Add all new messages to context at once
        messages_summary: list[Any] = []
        for msg in messages:
            messages_summary.append(
                f"[{msg.created_at}] {msg.peer_name}: {msg.content}"
            )

        messages_input = "\n".join(messages_summary)

        self.messages.append(
            {
                "role": "user",
                "content": "New messages to process:\n" + messages_input,
            }
        )

        # Store messages for tool execution
        self._current_messages = messages

        # Create tool executor with context
        tool_executor: Callable[[str, dict[str, Any]], Any] = create_tool_executor(
            db=self.db,
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            observer=self.observer,
            observed=self.observed,
            current_messages=messages,
            history_token_limit=settings.DERIVER.HISTORY_TOKEN_LIMIT,
        )

        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=settings.DERIVER,
            prompt="",  # Ignored since we pass messages
            max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS,
            tools=self._tools,
            tool_choice="required",
            tool_executor=tool_executor,
            max_tool_iterations=settings.DERIVER.MAX_TOOL_ITERATIONS,
            messages=self.messages,
            track_name="Deriver Agent",
            max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
        )

        # Log tool calls made with inputs and outputs
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )
        for i, tc in enumerate(response.tool_calls_made, 1):
            tool_name = tc.get("tool_name", "unknown")
            tool_input = tc.get("tool_input", {})
            accumulate_metric(
                task_name,
                f"tool_{i}_{tool_name}",
                f"INPUT: {tool_input}",  # \nOUTPUT: {tool_result}",
                "blob",
            )

        # Log token usage with cache awareness
        log_token_usage_metrics(
            task_name,
            response.input_tokens,
            response.output_tokens,
            response.cache_read_input_tokens,
            response.cache_creation_input_tokens,
        )
        if response.content:
            accumulate_metric(task_name, "response", response.content, "blob")

        # Log timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", elapsed_ms, "ms")

        log_performance_metrics("deriver_agent", run_id)
