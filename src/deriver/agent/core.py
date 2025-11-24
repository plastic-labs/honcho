import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.deriver.agent import prompts
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.utils.agent_tools import AGENT_TOOLS, create_agent_tool_executor
from src.utils.clients import honcho_llm_call

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
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": prompts.agent_system_prompt(
                    observer, observed, observed_peer_card
                ),
            }
        ]
        self._current_messages: list[Message] = []

    async def run_loop(self, messages: list[models.Message]) -> None:
        """
        Run the agent loop for a batch of messages using native tool calling.

        Args:
            messages: List of messages to process in this agent run
        """
        if not messages:
            logger.warning("run_loop called with empty message list")
            return

        # Add all new messages to context at once
        messages_summary: list[Any] = []
        for msg in messages:
            messages_summary.append(
                f"[{msg.created_at}] {msg.peer_name}: {msg.content}"
            )

        self.messages.append(
            {
                "role": "user",
                "content": "New messages to process:\n" + "\n".join(messages_summary),
            }
        )

        # Store messages for tool execution
        self._current_messages = messages

        # Create tool executor with context
        tool_executor: Callable[[str, dict[str, Any]], Any] = (
            create_agent_tool_executor(
                db=self.db,
                workspace_name=self.workspace_name,
                session_name=self.session_name,
                observer=self.observer,
                observed=self.observed,
                current_messages=messages,
            )
        )

        await honcho_llm_call(
            llm_settings=settings.DIALECTIC,
            prompt="",  # Ignored since we pass messages
            max_tokens=32_768,  # TODO config
            tools=AGENT_TOOLS,
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=10,
            messages=self.messages,
        )
