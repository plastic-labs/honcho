"""
Core Dialectic Agent implementation.

This agent uses tools to gather context from the memory system
and synthesize responses to queries about a peer.
"""

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.config import ReasoningLevel, settings
from src.dialectic import prompts
from src.dialectic.base import BaseDialecticAgent
from src.exceptions import ValidationException
from src.utils.agent_tools import (
    DIALECTIC_TOOLS,
    DIALECTIC_TOOLS_MINIMAL,
    create_tool_executor,
    search_memory,
)
from src.utils.formatting import format_new_turn_with_timestamp

logger = logging.getLogger(__name__)


class DialecticAgent(BaseDialecticAgent):
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
        self.observer: str = observer
        self.observed: str = observed
        self.observer_peer_card: list[str] | None = observer_peer_card
        self.observed_peer_card: list[str] | None = observed_peer_card
        self.metric_key: str | None = metric_key
        self._session_history_initialized: bool = False

        super().__init__(
            db=db,
            workspace_name=workspace_name,
            session_name=session_name,
            reasoning_level=reasoning_level,
            system_prompt=prompts.agent_system_prompt(
                observer, observed, observer_peer_card, observed_peer_card
            ),
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        prefetch_limit = 10 if self.reasoning_level == "minimal" else 25

        try:
            explicit_repr = await search_memory(
                db=self.db,
                workspace_name=self.workspace_name,
                observer=self.observer,
                observed=self.observed,
                query=query,
                limit=prefetch_limit,
                levels=["explicit"],
            )

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

            explicit_count: int = len(explicit_repr.explicit) + len(
                explicit_repr.deductive
            )
            derived_count: int = len(derived_repr.explicit) + len(
                derived_repr.deductive
            )
            self._prefetched_conclusion_count: int = explicit_count + derived_count

            parts: list[str] = []
            if not explicit_repr.is_empty():
                parts.append(explicit_repr.format_as_markdown(include_ids=False))
            if not derived_repr.is_empty():
                parts.append(derived_repr.format_as_markdown(include_ids=True))

            return "\n".join(parts)

        except ValidationException:
            raise
        except Exception as e:
            logger.warning(f"Failed to prefetch observations: {e}")
            return None

    async def _create_tool_executor(self) -> Callable[[str, dict[str, Any]], Any]:
        return await create_tool_executor(
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

    def _get_tools(self) -> list[dict[str, Any]]:
        if self.reasoning_level == "minimal":
            return DIALECTIC_TOOLS_MINIMAL
        return DIALECTIC_TOOLS

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def _trace_name(self) -> str:
        return "dialectic_chat"

    @property
    def _track_name(self) -> str:
        return "Dialectic Agent"

    @property
    def _telemetry_peer_name(self) -> str:
        return self.observed

    @property
    def _prefetch_header(self) -> str:
        return (
            "## Relevant Observations (prefetched)\n"
            "The following observations were found to be semantically relevant to your query. "
            "Use these as primary context. You may still use tools to find additional information if needed."
        )

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    async def _prepare_query(
        self, query: str
    ) -> tuple[Callable[[str, dict[str, Any]], Any], str, str | None, float]:
        await self._initialize_session_history()
        return await super()._prepare_query(query)

    def _get_task_name(self) -> tuple[str, str | None]:
        if self.metric_key:
            return self.metric_key, None
        return f"dialectic_chat_{self._run_id}", self._run_id

    def _get_context_string(self) -> str:
        return (
            f"workspace: {self.workspace_name}\n"
            f"session: {self.session_name or '(global)'}\n"
            f"observer: {self.observer}\n"
            f"observed: {self.observed}\n"
            f"reasoning_level: {self.reasoning_level}"
        )

    def _should_log_performance_metrics(self, run_id: str | None) -> bool:
        return not self.metric_key and run_id is not None

    # ------------------------------------------------------------------
    # Peer-only methods
    # ------------------------------------------------------------------

    async def _initialize_session_history(self) -> None:
        """Fetch and inject session history into the system prompt if configured."""
        if self._session_history_initialized:
            return
        self._session_history_initialized = True

        max_tokens = settings.DIALECTIC.SESSION_HISTORY_MAX_TOKENS
        if max_tokens == 0 or not self.session_name:
            return

        stmt = await crud.get_messages(
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            token_limit=max_tokens,
            reverse=False,
            peer_perspective=self.observer,
        )
        result = await self.db.execute(stmt)
        messages = result.scalars().all()

        if not messages:
            return

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

        self.messages[0]["content"] += session_history_section
