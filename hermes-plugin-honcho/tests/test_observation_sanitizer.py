"""Exercise opt-out matching with the pinned Hermes context sanitizer."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_plugin_honcho import HonchoMemoryProvider


@pytest.mark.parametrize(
    ("user_content", "stored_user", "no_observe"),
    [
        ("<memory-context>off the record</memory-context>real question", "real question", False),
        ("<memory-context>background</memory-context>off the record", "off the record", True),
    ],
)
def test_opt_out_matches_sanitized_user_content(user_content, stored_user, no_observe):
    provider = HonchoMemoryProvider()
    provider._session_key = "telegram:group1"
    provider._session_initialized = True
    provider._manager = MagicMock()
    provider._config = SimpleNamespace(message_max_chars=25000, observation_opt_out_phrases=["off the record"])
    session = provider._manager.get_or_create.return_value

    provider.sync_turn(user_content, "answer")
    if provider._sync_thread:
        provider._sync_thread.join(timeout=5)

    calls = session.add_message.call_args_list
    assert [(call.args[:2], call.kwargs["no_observe"]) for call in calls] == [
        (("user", stored_user), no_observe),
        (("assistant", "answer"), no_observe),
    ]
    provider._manager.save.assert_called_once_with(session)
