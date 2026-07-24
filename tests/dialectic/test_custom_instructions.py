from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.dialectic.chat import agentic_chat
from src.dialectic.core import DialecticAgent
from src.dialectic.prompts import agent_system_prompt


def _system_prompt(agent: DialecticAgent) -> str:
    return agent.messages[0]["content"]


def test_agent_system_prompt_renders_custom_instructions() -> None:
    prompt = agent_system_prompt(
        "observer",
        "observed",
        None,
        None,
        "favor durable patterns, never dwell on sensitive details",
    )

    assert "CUSTOM INSTRUCTIONS:" in prompt
    assert "favor durable patterns, never dwell on sensitive details" in prompt


@pytest.mark.parametrize("custom_instructions", [None, "", "   "])
def test_agent_system_prompt_omits_empty_custom_instructions(
    custom_instructions: str | None,
) -> None:
    prompt = agent_system_prompt(
        "observer", "observed", None, None, custom_instructions
    )

    assert "CUSTOM INSTRUCTIONS:" not in prompt


def test_agent_system_prompt_preserves_multiline_custom_instructions() -> None:
    prompt = agent_system_prompt(
        "observer",
        "observed",
        None,
        None,
        "this is a growth mirror, not a scorecard\nnever dwell on sensitive details",
    )

    assert "\nCUSTOM INSTRUCTIONS:\n" in prompt
    assert "\nthis is a growth mirror, not a scorecard\n" in prompt
    assert "\nnever dwell on sensitive details\n" in prompt


def test_dialectic_agent_threads_custom_instructions_into_system_prompt() -> None:
    agent = DialecticAgent(
        workspace_name="workspace",
        session_name="session",
        observer="observer",
        observed="observed",
        custom_instructions="this is a growth mirror, not a performance scorecard",
    )

    assert "CUSTOM INSTRUCTIONS:" in _system_prompt(agent)
    assert "this is a growth mirror, not a performance scorecard" in _system_prompt(
        agent
    )


def test_dialectic_agent_without_custom_instructions_has_no_section() -> None:
    agent = DialecticAgent(
        workspace_name="workspace",
        session_name="session",
        observer="observer",
        observed="observed",
    )

    assert "CUSTOM INSTRUCTIONS:" not in _system_prompt(agent)


@pytest.mark.asyncio
async def test_agentic_chat_forwards_workspace_custom_instructions(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    test_workspace, test_peer = sample_data
    test_workspace.configuration = {
        "reasoning": {"custom_instructions": "favor durable patterns"}
    }
    await db_session.commit()

    with patch("src.dialectic.chat.DialecticAgent") as mock_agent:
        mock_agent.return_value.answer = AsyncMock(return_value="answer")
        await agentic_chat(
            workspace_name=test_workspace.name,
            session_name=None,
            query="what do you know about me?",
            observer=test_peer.name,
            observed=test_peer.name,
        )

    assert (
        mock_agent.call_args.kwargs["custom_instructions"] == "favor durable patterns"
    )


@pytest.mark.asyncio
async def test_agentic_chat_prefers_session_custom_instructions(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    test_workspace, test_peer = sample_data
    test_workspace.configuration = {
        "reasoning": {"custom_instructions": "favor durable patterns"}
    }
    test_session = models.Session(
        workspace_name=test_workspace.name,
        name=str(generate_nanoid()),
        configuration={"reasoning": {"custom_instructions": "keep answers terse"}},
    )
    db_session.add(test_session)
    await db_session.commit()

    with patch("src.dialectic.chat.DialecticAgent") as mock_agent:
        mock_agent.return_value.answer = AsyncMock(return_value="answer")
        await agentic_chat(
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            query="what do you know about me?",
            observer=test_peer.name,
            observed=test_peer.name,
        )

    assert mock_agent.call_args.kwargs["custom_instructions"] == "keep answers terse"
