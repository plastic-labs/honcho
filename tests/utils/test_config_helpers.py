from typing import Any

import pytest

from src import models
from src.config import settings
from src.schemas import MessageConfiguration, ReasoningConfiguration
from src.utils.config_helpers import get_configuration


def _workspace(configuration: dict[str, Any]) -> models.Workspace:
    return models.Workspace(name="workspace", configuration=configuration)


def _session(configuration: dict[str, Any]) -> models.Session:
    return models.Session(
        name="session",
        workspace_name="workspace",
        configuration=configuration,
    )


def test_preserves_workspace_custom_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DERIVER, "MAX_CUSTOM_INSTRUCTIONS_TOKENS", 100)

    workspace = _workspace(
        {
            "reasoning": {
                "custom_instructions": "Use the workspace-specific guidance.",
            }
        }
    )

    configuration = get_configuration(None, None, workspace)

    assert (
        configuration.reasoning.custom_instructions
        == "Use the workspace-specific guidance."
    )


def test_message_custom_instructions_override_session_and_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.DERIVER, "MAX_CUSTOM_INSTRUCTIONS_TOKENS", 100)

    workspace = _workspace(
        {
            "reasoning": {
                "custom_instructions": "Use the workspace-specific guidance.",
            }
        }
    )
    session = _session(
        {
            "reasoning": {
                "custom_instructions": "Use the session-specific guidance.",
            }
        }
    )
    message = MessageConfiguration(
        reasoning=ReasoningConfiguration(
            custom_instructions="Use the message-specific guidance.",
        ),
    )

    configuration = get_configuration(message, session, workspace)

    assert (
        configuration.reasoning.custom_instructions
        == "Use the message-specific guidance."
    )


@pytest.fixture
def instructions_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.DERIVER, "MAX_CUSTOM_INSTRUCTIONS_TOKENS", 100)


@pytest.mark.usefixtures("instructions_budget")
def test_shared_custom_instructions_reach_every_module() -> None:
    configuration = get_configuration(
        None, None, _workspace({"custom_instructions": "Write in German."})
    )

    assert configuration.reasoning.custom_instructions == "Write in German."
    assert configuration.summary.custom_instructions == "Write in German."
    assert configuration.dialectic.custom_instructions == "Write in German."
    assert configuration.dream.custom_instructions == "Write in German."
    # peer_card instructions are additive on top of dream and never fall back
    assert configuration.peer_card.custom_instructions is None


@pytest.mark.usefixtures("instructions_budget")
def test_module_custom_instructions_override_shared_at_same_level() -> None:
    configuration = get_configuration(
        None,
        None,
        _workspace(
            {
                "custom_instructions": "Write in German.",
                "summary": {"custom_instructions": "Use bullet points."},
                "peer_card": {"custom_instructions": "Track plan tier."},
            }
        ),
    )

    assert configuration.summary.custom_instructions == "Use bullet points."
    assert configuration.dialectic.custom_instructions == "Write in German."
    assert configuration.peer_card.custom_instructions == "Track plan tier."


@pytest.mark.usefixtures("instructions_budget")
def test_more_specific_level_wins_over_module_override() -> None:
    workspace = _workspace({"summary": {"custom_instructions": "Use bullet points."}})
    session = _session({"custom_instructions": "Write in Portuguese."})

    configuration = get_configuration(None, session, workspace)

    assert configuration.summary.custom_instructions == "Write in Portuguese."


@pytest.mark.usefixtures("instructions_budget")
def test_empty_custom_instructions_stop_fallback() -> None:
    workspace = _workspace({"custom_instructions": "Write in German."})
    session = _session({"dialectic": {"custom_instructions": ""}})

    configuration = get_configuration(None, session, workspace)

    assert configuration.dialectic.custom_instructions is None
    assert configuration.summary.custom_instructions == "Write in German."


@pytest.mark.usefixtures("instructions_budget")
def test_reasoning_custom_instructions_stay_deriver_only() -> None:
    configuration = get_configuration(
        None, None, _workspace({"reasoning": {"custom_instructions": "Deriver."}})
    )

    assert configuration.reasoning.custom_instructions == "Deriver."
    assert configuration.summary.custom_instructions is None
    assert configuration.dialectic.custom_instructions is None
    assert configuration.dream.custom_instructions is None
