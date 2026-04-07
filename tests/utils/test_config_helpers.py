import pytest
from pydantic import ValidationError

from src import models
from src.config import settings
from src.schemas import MessageConfiguration, ReasoningConfiguration
from src.utils.config_helpers import get_configuration


class TestGetConfiguration:
    def test_preserves_workspace_custom_instructions(self) -> None:
        workspace = models.Workspace(
            name="workspace-1",
            configuration={
                "reasoning": {
                    "enabled": True,
                    "custom_instructions": "Focus on durable preferences.",
                }
            }
        )

        config = get_configuration(None, None, workspace)

        assert config.reasoning.enabled is True
        assert config.reasoning.custom_instructions == "Focus on durable preferences."

    def test_message_custom_instructions_override_session_and_workspace(self) -> None:
        workspace = models.Workspace(
            name="workspace-1",
            configuration={"reasoning": {"custom_instructions": "workspace scope"}}
        )
        session = models.Session(
            name="session-1",
            workspace_name="workspace-1",
            configuration={"reasoning": {"custom_instructions": "session scope"}}
        )
        message = MessageConfiguration(
            reasoning=ReasoningConfiguration(custom_instructions="message scope")
        )

        config = get_configuration(message, session, workspace)

        assert config.reasoning.custom_instructions == "message scope"

    def test_get_configuration_rejects_over_budget_custom_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            settings.DERIVER,
            "MAX_CUSTOM_INSTRUCTIONS_TOKENS",
            5,
        )

        workspace = models.Workspace(
            name="workspace-1",
            configuration={
                "reasoning": {
                    "enabled": True,
                    "custom_instructions": "focus on stable preferences and long-term plans",
                }
            },
        )

        with pytest.raises(ValidationError) as exc_info:
            get_configuration(None, None, workspace)

        assert "custom_instructions" in str(exc_info.value)
