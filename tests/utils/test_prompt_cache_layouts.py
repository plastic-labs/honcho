from unittest.mock import AsyncMock

from src.dialectic.core import DialecticAgent
from src.utils.prompt_cache_layouts import (
    build_system_messages,
    merge_system_prompt_with_rolling_context,
    provider_default_prompt_cache_layout,
)
from src.utils.types import SupportedProviders


class DialecticAgentHarness(DialecticAgent):
    _provider: SupportedProviders

    @property
    def provider(self) -> SupportedProviders:
        return self._provider

    @provider.setter
    def provider(self, value: SupportedProviders) -> None:
        self._provider = value

    @property
    def base_system_prompt(self) -> str:
        return self._base_system_prompt

    def set_system_messages(self, session_history_section: str | None = None) -> None:
        self._set_system_messages(session_history_section)


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
    agent = DialecticAgentHarness(
        AsyncMock(),
        "workspace",
        "session",
        "Mira",
        "Jon",
        reasoning_level="low",
    )

    agent.provider = "google"
    agent.set_system_messages("rolling history")

    assert agent.messages == [
        {
            "role": "system",
            "content": (
                agent.base_system_prompt.strip()
                + "\n\n<rolling_history>\nrolling history\n</rolling_history>"
            ),
        }
    ]


def test_dialectic_agent_rebuilds_anthropic_system_messages() -> None:
    agent = DialecticAgentHarness(
        AsyncMock(),
        "workspace",
        "session",
        "Mira",
        "Jon",
        reasoning_level="medium",
    )

    agent.provider = "anthropic"
    agent.set_system_messages("rolling history")

    assert agent.messages == [
        {"role": "system", "content": agent.base_system_prompt.strip()},
        {"role": "system", "content": "rolling history"},
    ]
