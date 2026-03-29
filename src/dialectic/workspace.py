"""
Workspace-level Dialectic Agent implementation.

This agent answers queries across all peers in a workspace using an
analytics-first approach: stats -> message search -> targeted observations.
"""

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

import src.crud as crud
from src.config import ReasoningLevel, settings
from src.dialectic import prompts
from src.dialectic.base import BaseDialecticAgent
from src.utils.agent_tools import (
    WORKSPACE_DIALECTIC_TOOLS,
    create_workspace_tool_executor,
)

logger = logging.getLogger(__name__)


class WorkspaceDialecticAgent(BaseDialecticAgent):
    """
    A workspace-level dialectic agent that answers queries across all peers.

    Unlike DialecticAgent which focuses on a single observer/observed pair,
    this agent discovers relevant peers through workspace stats, message
    search, and then drills into specific observation pairs.
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_name: str,
        session_name: str | None = None,
        reasoning_level: ReasoningLevel = "low",
    ):
        super().__init__(
            db=db,
            workspace_name=workspace_name,
            session_name=session_name,
            reasoning_level=reasoning_level,
            system_prompt=prompts.workspace_agent_system_prompt(),
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        """Prefetch workspace stats so the agent can orient itself."""
        _ = query
        stats = await crud.get_workspace_stats(self.db, self.workspace_name)
        if stats.peer_count == 0:
            return None
        lines = [
            f"Peers: {stats.peer_count}",
            f"Sessions: {stats.session_count}",
            f"Messages: {stats.message_count}",
        ]
        if stats.oldest_message_at and stats.newest_message_at:
            lines.append(
                f"Date range: {stats.oldest_message_at:%Y-%m-%d} to {stats.newest_message_at:%Y-%m-%d}"
            )
        return "\n".join(lines)

    async def _create_tool_executor(self) -> Callable[[str, dict[str, Any]], Any]:
        return await create_workspace_tool_executor(
            db=self.db,
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            history_token_limit=settings.DIALECTIC.HISTORY_TOKEN_LIMIT,
            run_id=self._run_id,
            agent_type="workspace_dialectic",
            parent_category="dialectic",
        )

    def _get_tools(self) -> list[dict[str, Any]]:
        return WORKSPACE_DIALECTIC_TOOLS

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def _trace_name(self) -> str:
        return "workspace_chat"

    @property
    def _track_name(self) -> str:
        return "Workspace Dialectic Agent"

    @property
    def _telemetry_peer_name(self) -> str:
        return "(workspace)"

    @property
    def _prefetch_header(self) -> str:
        return "Workspace overview (use get_active_peers or search_messages to discover relevant peers):"
