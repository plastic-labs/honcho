"""Workspace-level dialectic agent.

Answers queries across ALL peers in a workspace. Where DialecticAgent is
bound to a single (observer, observed) pair, this agent routes first —
workspace stats, active peers, and peer cards are prefetched for
orientation; message search is workspace-flat and reveals which peers
discussed a topic — and then recalls through the same pair-scoped
observation machinery, supplying the pair as tool arguments.

Observation search deliberately stays pair-scoped: it matches both the
(observer, observed) collection ownership and the per-pair vector-store
namespaces, and avoids retrieval dilution from a workspace-flat top-k.

Design carried over from plastic-labs/honcho#373 (Dan), re-grown on the
current DialecticAgent seams instead of a base-class extraction.
"""

import logging
from collections.abc import Callable
from typing import Any

from src import crud
from src.config import ReasoningLevel, settings
from src.dependencies import tracked_db
from src.dialectic import prompts
from src.dialectic.core import DialecticAgent
from src.llm.types import LLMTelemetryContext
from src.utils.agent_tools import (
    WORKSPACE_DIALECTIC_TOOLS,
    WORKSPACE_TOOLS_MINIMAL,
    create_workspace_tool_executor,
    format_workspace_stats,
)

logger = logging.getLogger(__name__)

# How many active peers (with their self peer cards) to inject at prefetch.
# Routing-obvious queries should resolve without a discovery tool round —
# each avoided tool round is a full model turn (~1.3s measured).
_PREFETCH_ACTIVE_PEERS = 5


class WorkspaceDialecticAgent(DialecticAgent):
    """Dialectic agent scoped to a whole workspace instead of a peer pair."""

    def __init__(
        self,
        workspace_name: str,
        session_name: str | None = None,
        metric_key: str | None = None,
        reasoning_level: ReasoningLevel = "low",
        session_id: str | None = None,
        session_allowlist: list[str] | None = None,
    ) -> None:
        super().__init__(
            workspace_name=workspace_name,
            session_name=session_name,
            observer="",
            observed="",
            metric_key=metric_key,
            reasoning_level=reasoning_level,
            session_id=session_id,
            session_allowlist=session_allowlist,
        )
        # Replace the pair-oriented system prompt with the workspace one.
        self.messages[0] = {
            "role": "system",
            "content": prompts.workspace_agent_system_prompt(),
        }

    # ------------------------------------------------------------------
    # DialecticAgent seams
    # ------------------------------------------------------------------

    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        """Orientation + routing prefetch: stats, active peers, peer cards.

        No semantic retrieval here — a workspace-flat observation top-k
        would be dominated by the most verbose peers. Instead give the
        agent what it needs to ROUTE: who is here, who is active, and what
        is known about them at a glance.
        """
        _ = query
        # Like the base agent, prefetch failure degrades to no prefetched
        # block rather than failing the whole request (the caller in
        # _prepare_query does not guard this).
        try:
            async with tracked_db("dialectic.workspace_prefetch", read_only=True) as db:
                stats = await crud.get_workspace_stats(
                    db,
                    self.workspace_name,
                    session_names=self.session_allowlist,
                )
                if stats.peer_count == 0:
                    return None
                peers = await crud.get_active_peers(
                    db,
                    self.workspace_name,
                    limit=_PREFETCH_ACTIVE_PEERS,
                    session_names=self.session_allowlist,
                )
                # `peers` is already allowlist-filtered, but a peer card is a
                # single cross-session aggregate: an in-scope peer's card can
                # still carry facts derived from sessions outside the scope.
                # Drop cards entirely under an allowlist — same rule the
                # get_peer_card tool enforces — and route on stats alone.
                cards: dict[str, list[str]] = {}
                if self.session_allowlist is None:
                    for peer in peers:
                        card = await crud.get_peer_card(
                            db,
                            workspace_name=self.workspace_name,
                            observer=peer.name,
                            observed=peer.name,
                        )
                        if card:
                            cards[peer.name] = card
        except Exception:
            logger.warning(
                "Failed to prefetch workspace overview for workspace=%s",
                self.workspace_name,
                exc_info=True,
            )
            return None

        return format_workspace_stats(stats, peers, cards)

    def _prefetch_intro(self) -> str:
        return (
            "Workspace overview and most-active peers with any known "
            "biographical facts. Use this to route: query a specific peer's "
            "memory with search_memory (observer and observed set to that "
            "peer's name), or use search_messages / get_workspace_stats to "
            "discover peers this overview does not cover."
        )

    def _select_tools(self) -> list[dict[str, Any]]:
        tools = (
            WORKSPACE_TOOLS_MINIMAL
            if self.reasoning_level == "minimal"
            else WORKSPACE_DIALECTIC_TOOLS
        )
        # Mirror the base agent's allowlist rule, for both tools that cannot
        # honor an allowlist: reasoning chains traverse provenance across
        # sessions, and a peer card is one cross-session aggregate with no
        # per-session attribution. Both fail closed in their handlers too;
        # dropping them here avoids paying the schema tokens and a wasted
        # turn on a tool that can only refuse.
        if self.session_allowlist is not None:
            unscopable = {"get_reasoning_chain", "get_peer_card"}
            tools = [t for t in tools if t.get("name") not in unscopable]
        return tools

    async def _create_tool_executor(self) -> Callable[[str, dict[str, Any]], Any]:
        return await create_workspace_tool_executor(
            workspace_name=self.workspace_name,
            session_name=self.session_name,
            session_allowlist=self.session_allowlist,
            history_token_limit=settings.DIALECTIC.HISTORY_TOKEN_LIMIT,
            run_id=self._run_id,
            agent_type="workspace_dialectic",
            parent_category="dialectic",
        )

    # Workspace chat shares the base "dialectic_chat" Langfuse trace name;
    # scope is distinguished by the agent_type/track_name below.

    def _telemetry_context(self, track_name: str | None = None) -> LLMTelemetryContext:
        return LLMTelemetryContext(
            workspace_name=self.workspace_name,
            call_purpose="dialectic.answer",
            parent_category="dialectic",
            agent_type="workspace_dialectic",
            run_id=self._run_id,
            trace_id=self._run_id,
            span_id=self._run_id,
            session_id=self.session_id,
            peer_name="(workspace)",
            track_name=track_name or "Workspace Dialectic Agent",
        )
