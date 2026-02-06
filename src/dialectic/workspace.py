"""
Workspace-level Dialectic Agent implementation.

This agent answers queries across ALL peers in a workspace,
providing an omniscient view that synthesizes across the entire
workspace's data.
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
    WORKSPACE_DIALECTIC_TOOLS,
    create_workspace_tool_executor,
)
from src.utils.representation import format_documents_with_attribution

logger = logging.getLogger(__name__)


class WorkspaceDialecticAgent(BaseDialecticAgent):
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

            self._prefetched_conclusion_count: int = len(documents)
            return format_documents_with_attribution(documents, include_ids=True)

        except ValidationException:
            raise
        except Exception as e:
            logger.error(
                "Failed to prefetch workspace observations for workspace=%s: %s",
                self.workspace_name,
                e,
                exc_info=True,
            )
            return None

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
        return (
            "## Relevant Observations (prefetched from across workspace)\n"
            "The following observations were found to be semantically relevant to your query. "
            "They span multiple peers. Use these as primary context. You may still use tools to find additional information."
        )
