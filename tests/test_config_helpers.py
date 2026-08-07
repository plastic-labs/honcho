from types import SimpleNamespace
from typing import Any

from src.schemas import MessageConfiguration, SummaryConfiguration
from src.utils.config_helpers import deep_update, get_configuration


def _configured_node(configuration: dict[str, Any]) -> Any:
    return SimpleNamespace(configuration=configuration)


class TestDeepUpdate:
    def test_summary_custom_instructions_none_clears_inherited_value(self) -> None:
        base = {
            "summary": {
                "enabled": True,
                "custom_instructions": "Write summaries in German.",
            }
        }

        deep_update(
            base,
            {"summary": {"enabled": None, "custom_instructions": None}},
        )

        assert base["summary"]["enabled"] is True
        assert base["summary"]["custom_instructions"] is None


class TestGetConfiguration:
    def test_session_can_clear_workspace_summary_custom_instructions(self) -> None:
        workspace = _configured_node(
            {"summary": {"custom_instructions": "Write summaries in German."}}
        )
        session = _configured_node({"summary": {"custom_instructions": None}})

        configuration = get_configuration(None, session, workspace)

        assert configuration.summary.custom_instructions is None

    def test_message_can_clear_session_summary_custom_instructions(self) -> None:
        session = _configured_node(
            {"summary": {"custom_instructions": "Write summaries in German."}}
        )
        message_configuration = MessageConfiguration(
            summary=SummaryConfiguration(custom_instructions=None)
        )

        configuration = get_configuration(message_configuration, session)

        assert configuration.summary.custom_instructions is None
