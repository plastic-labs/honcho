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
