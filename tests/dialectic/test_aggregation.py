"""The dialectic over a peer with members.

Recall (prefetch, `search_memory`, reasoning chains, the brief) covers every
peer in the observed set and keeps each conclusion attributed. Only the LLM
call is mocked.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.dialectic import prompts
from src.dialectic.core import DialecticAgent
from src.llm import HonchoLLMCallResponse
from src.utils.evidence import EvidenceAccumulator
from tests.conftest import (
    _content_to_embedding,  # pyright: ignore[reportPrivateUsage]
)

FACTS = {
    "team": "The team ships on Fridays",
    "alice": "Alice leads the rewrite",
    "bob": "Bob owns the billing service",
}


def _response() -> HonchoLLMCallResponse[str]:
    return HonchoLLMCallResponse(
        content="answer",
        input_tokens=10,
        output_tokens=5,
        finish_reasons=["end_turn"],
        tool_calls_made=[],
    )


@pytest.fixture
async def team_set(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, crud.ObservedSet, dict[str, models.Document]]:
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", False)
    workspace, _ = sample_data
    ws = workspace.name
    db_session.add_all(models.Peer(name=name, workspace_name=ws) for name in FACTS)
    await db_session.flush()
    documents: dict[str, models.Document] = {}
    for name, fact in FACTS.items():
        db_session.add(
            models.Collection(workspace_name=ws, observer=name, observed=name)
        )
        await db_session.flush()
        document = models.Document(
            workspace_name=ws,
            observer=name,
            observed=name,
            content=fact,
            level="explicit",
            embedding=_content_to_embedding(fact),
        )
        db_session.add(document)
        documents[name] = document
    await db_session.commit()
    observed_set = crud.ObservedSet(
        peers=["team", "alice", "bob"],
        pairs=[("team", "team"), ("alice", "alice"), ("bob", "bob")],
    )
    return ws, observed_set, documents


def _agent(ws: str, observed_set: crud.ObservedSet | None, **kwargs: Any):
    return DialecticAgent(
        workspace_name=ws,
        session_name=None,
        observer="team",
        observed="team",
        observed_set=observed_set,
        **kwargs,
    )


async def _answer(agent: DialecticAgent) -> AsyncMock:
    mock = AsyncMock(return_value=_response())
    with patch("src.dialectic.core.honcho_llm_call", new=mock):
        await agent.answer("What is the team working on?")
    return mock


@pytest.mark.asyncio
class TestRecallCoversMembers:
    async def test_prefetch_reads_every_member_under_its_heading(self, team_set: Any):
        ws, observed_set, _ = team_set
        agent = _agent(ws, observed_set)
        await _answer(agent)

        user = agent.messages[1]["content"]
        for name, fact in FACTS.items():
            assert f"# {name}" in user
            assert fact in user

    async def test_without_members_prefetch_is_the_target_alone(self, team_set: Any):
        ws, _, _ = team_set
        agent = _agent(ws, None)
        await _answer(agent)

        user = agent.messages[1]["content"]
        assert FACTS["team"] in user
        assert FACTS["alice"] not in user
        assert "# team" not in user

    async def test_search_memory_tool_attributes_results(self, team_set: Any):
        ws, observed_set, _ = team_set
        agent = _agent(ws, observed_set)
        execute = await agent._create_tool_executor()  # pyright: ignore[reportPrivateUsage]

        result = await execute("search_memory", {"query": "billing", "top_k": 10})
        content = result if isinstance(result, str) else result.content
        assert "# bob" in content
        assert FACTS["bob"] in content

    async def test_evidence_names_the_member_each_conclusion_is_about(
        self, team_set: Any
    ):
        ws, observed_set, documents = team_set
        evidence = EvidenceAccumulator()
        await _answer(_agent(ws, observed_set, evidence=evidence))

        observed_by_id = {c.id: c.observed_id for c in evidence.build().conclusions}
        assert observed_by_id[documents["alice"].id] == "alice"
        assert observed_by_id[documents["bob"].id] == "bob"

    async def test_reasoning_chain_finds_children_in_member_collections(
        self, db_session: AsyncSession, team_set: Any
    ):
        ws, observed_set, documents = team_set
        child = models.Document(
            workspace_name=ws,
            observer="alice",
            observed="alice",
            content="Alice is accountable for the rewrite's schedule",
            level="deductive",
            embedding=_content_to_embedding("schedule"),
        )
        db_session.add(child)
        await db_session.flush()
        db_session.add(
            models.DocumentSource(
                derived_id=child.id,
                source_id=documents["alice"].id,
                workspace_name=ws,
            )
        )
        await db_session.commit()

        agent = _agent(ws, observed_set)
        execute = await agent._create_tool_executor()  # pyright: ignore[reportPrivateUsage]
        result = await execute(
            "get_reasoning_chain",
            {"observation_id": documents["alice"].id, "direction": "conclusions"},
        )
        content = result if isinstance(result, str) else result.content
        assert "accountable for the rewrite's schedule" in content

    async def test_brief_counts_span_the_set(
        self, team_set: Any, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.DIALECTIC, "BRIEF_ENABLED", True)
        ws, observed_set, _ = team_set
        aggregated = await _agent(ws, observed_set)._pair_brief()  # pyright: ignore[reportPrivateUsage]
        alone = await _agent(ws, None)._pair_brief()  # pyright: ignore[reportPrivateUsage]
        assert aggregated is not None and alone is not None
        assert "(3 conclusions" in aggregated
        assert "(1 conclusions" in alone


class TestPrompt:
    def test_members_section_names_members_and_cards(self):
        prompt = prompts.agent_system_prompt(
            "team",
            "team",
            None,
            None,
            members=["alice", "bob"],
            member_peer_cards={"alice": ["Name: Alice"]},
        )
        assert "## MEMBERS" in prompt
        assert "'alice', 'bob'" in prompt
        assert '<member_peer_card peer="alice">\nName: Alice' in prompt
        assert 'peer="bob"' not in prompt

    def test_no_members_leaves_the_prompt_unchanged(self):
        assert prompts.agent_system_prompt(
            "alice", "alice", ["card"], None
        ) == prompts.agent_system_prompt("alice", "alice", ["card"], None, members=())


class TestChatRoute:
    def test_chat_passes_the_expanded_set(
        self,
        client: TestClient,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, _ = sample_data
        ws = workspace.name
        assert (
            client.post(
                f"/v3/workspaces/{ws}/peers/team/members", json={"peer_id": "alice"}
            ).status_code
            == 201
        )

        captured: dict[str, Any] = {}

        async def fake_chat(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        with patch("src.routers.peers.agentic_chat", new=fake_chat):
            response = client.post(
                f"/v3/workspaces/{ws}/peers/team/chat", json={"query": "status?"}
            )
            assert response.status_code == 200, response.text
            assert captured["observed_set"].peers == ["team", "alice"]
            assert captured["observed_set"].pairs == [
                ("team", "team"),
                ("alice", "alice"),
            ]

            response = client.post(
                f"/v3/workspaces/{ws}/peers/bob/chat",
                json={"query": "status?", "target": "team"},
            )
            assert response.status_code == 200, response.text
            # A directional read keeps its observer for every member.
            assert captured["observed_set"].pairs == [
                ("bob", "team"),
                ("bob", "alice"),
            ]

    def test_depth_violation_is_a_422_before_streaming(
        self,
        client: TestClient,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, _ = sample_data
        response = client.post(
            f"/v3/workspaces/{workspace.name}/peers/team/chat",
            json={
                "query": "status?",
                "stream": True,
                "depth": settings.PEER.AGGREGATION_MAX_DEPTH + 1,
            },
        )
        assert response.status_code == 422
