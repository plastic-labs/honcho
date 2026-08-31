"""Contracts for dialectic system prompts vs the tool loadouts they describe."""

import re

from src.dialectic.core import DialecticAgent
from src.dialectic.prompts import (
    PAIR_PROMPT_TOOLS,
    WORKSPACE_PROMPT_TOOLS,
    agent_system_prompt,
    workspace_agent_system_prompt,
)
from src.dialectic.workspace import WorkspaceDialecticAgent
from src.utils.agent_tools import (
    DIALECTIC_TOOLS,
    DIALECTIC_TOOLS_MINIMAL,
    TOOLS,
    WORKSPACE_DIALECTIC_TOOLS,
    WORKSPACE_TOOLS_MINIMAL,
)

_ALL_TOOL_NAMES = {spec["name"] for spec in TOOLS.values()}


def _loadout_names(tools: list[dict[str, object]]) -> set[str]:
    return {name for tool in tools if isinstance((name := tool.get("name")), str)}


def _mentioned_tools(text: str) -> set[str]:
    return {
        match
        for match in re.findall(r"`([a-z_][a-z0-9_]*)`", text)
        if match in _ALL_TOOL_NAMES
    }


def _tools_catalog(prompt: str) -> str:
    start = prompt.index("## TOOLS")
    rest = prompt[start:]
    next_heading = rest.find("\n## ", 1)
    return rest if next_heading == -1 else rest[:next_heading]


class TestPromptLoadouts:
    def test_pair_docs_match_dialectic_tools(self) -> None:
        assert _loadout_names(DIALECTIC_TOOLS) == PAIR_PROMPT_TOOLS

    def test_workspace_docs_match_workspace_tools(self) -> None:
        assert _loadout_names(WORKSPACE_DIALECTIC_TOOLS) == WORKSPACE_PROMPT_TOOLS

    def test_catalog_lists_only_offered_workspace_tools(self) -> None:
        offered = _loadout_names(WORKSPACE_TOOLS_MINIMAL)
        catalog = _tools_catalog(workspace_agent_system_prompt(offered))
        assert _mentioned_tools(catalog) == offered


class TestPairAgentPrompt:
    def test_teaches_honcho_world_without_workspace_sibling(self) -> None:
        prompt = agent_system_prompt("alice", "alice", None, None).lower()
        assert "workspace dialectic" not in prompt
        assert "peer-level" not in prompt
        for term in ("honcho", "peer", "session", "message", "conclusion"):
            assert term in prompt

    def test_does_not_offer_removed_write_tools(self) -> None:
        prompt = agent_system_prompt("alice", "alice", None, None)
        assert "create_observations_deductive" not in prompt
        assert "create_observations" not in prompt

    def test_agent_lists_selected_tools(self) -> None:
        agent = DialecticAgent(
            workspace_name="w",
            session_name=None,
            observer="alice",
            observed="alice",
            reasoning_level="minimal",
        )
        offered = _loadout_names(DIALECTIC_TOOLS_MINIMAL)
        catalog = _tools_catalog(agent.messages[0]["content"])
        assert _mentioned_tools(catalog) == offered
        assert agent.messages[0]["content"] == agent_system_prompt(
            "alice", "alice", None, None, available_tools=offered
        )


class TestWorkspaceAgentPrompt:
    def test_minimal_agent_matches_filtered_prompt(self) -> None:
        agent = WorkspaceDialecticAgent(workspace_name="w", reasoning_level="minimal")
        offered = _loadout_names(WORKSPACE_TOOLS_MINIMAL)
        prompt = agent.messages[0]["content"]
        assert prompt == workspace_agent_system_prompt(offered)
        catalog = _tools_catalog(prompt)
        assert _mentioned_tools(catalog) == offered
        assert "get_peer_card" not in catalog
        assert "get_reasoning_chain" not in catalog
