from unittest.mock import AsyncMock

from src.dialectic.core import DialecticAgent
from src.utils.prompt_cache_layouts import (
    build_system_messages,
    merge_system_prompt_with_rolling_context,
    provider_default_prompt_cache_layout,
)


def test_provider_default_prompt_cache_layout_is_google_specific() -> None:
    assert provider_default_prompt_cache_layout("google") == "merged_system"
    assert provider_default_prompt_cache_layout("anthropic") == "split"
    assert provider_default_prompt_cache_layout("openai") == "split"


def test_build_system_messages_merges_google_history() -> None:
    messages = build_system_messages(
        "google",
        "stable instructions",
        "rolling history",
        wrapper_tag="rolling_history",
    )

    assert messages == [
        {
            "role": "system",
            "content": (
                "stable instructions\n\n"
                "<rolling_history>\nrolling history\n</rolling_history>"
            ),
        }
    ]


def test_build_system_messages_keeps_anthropic_history_separate() -> None:
    messages = build_system_messages(
        "anthropic",
        "stable instructions",
        "rolling history",
        wrapper_tag="rolling_history",
    )

    assert messages == [
        {"role": "system", "content": "stable instructions"},
        {"role": "system", "content": "rolling history"},
    ]


def test_merge_system_prompt_with_rolling_context_strips_noise() -> None:
    merged = merge_system_prompt_with_rolling_context(
        "\n stable instructions \n",
        "\n rolling history \n",
        wrapper_tag="rolling_history",
    )

    assert (
        merged
        == "stable instructions\n\n<rolling_history>\nrolling history\n</rolling_history>"
    )


def test_dialectic_agent_rebuilds_google_system_messages() -> None:
    agent = DialecticAgent(
        db=AsyncMock(),
        workspace_name="workspace",
        session_name="session",
        observer="Mira",
        observed="Jon",
        reasoning_level="low",
    )

    agent._provider = "google"
    agent._set_system_messages("rolling history")

    assert agent.messages == [
        {
            "role": "system",
            "content": (
                agent._base_system_prompt.strip()
                + "\n\n<rolling_history>\nrolling history\n</rolling_history>"
            ),
        }
    ]


def test_dialectic_agent_rebuilds_anthropic_system_messages() -> None:
    agent = DialecticAgent(
        db=AsyncMock(),
        workspace_name="workspace",
        session_name="session",
        observer="Mira",
        observed="Jon",
        reasoning_level="medium",
    )

    agent._provider = "anthropic"
    agent._set_system_messages("rolling history")

    assert agent.messages == [
        {"role": "system", "content": agent._base_system_prompt.strip()},
        {"role": "system", "content": "rolling history"},
    ]
