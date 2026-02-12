"""
Workspace-level Dialectic Agent implementation.

This agent answers queries across all peers in a workspace,
synthesizing information from multiple peer representations.
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
    this agent can query any peer representation in the workspace. It must
    discover peers via list_peers and search within specific observer/observed
    pairs using search_memory.
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
        """Prefetch the list of peers so the agent can skip the list_peers tool call."""
        _ = query
        stmt = await crud.get_peers(workspace_name=self.workspace_name)
        result = await self.db.execute(stmt)
        peers = list(result.scalars().all())
        if not peers:
            return None
        peer_list = "\n".join(f"- {p.name}" for p in peers)
        return f"Found {len(peers)} peers in workspace:\n{peer_list}"

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
        return "Peers in this workspace (already fetched — do NOT call list_peers):"
