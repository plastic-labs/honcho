"""The fixed behaviour of the ideal plugin configuration (DEV-3258): one Honcho session per Hermes chat joined to
its scopes, every peer observing only itself, the directives and the user's card at session start, and recall
through ``honcho_chat``."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import HonchoClientConfig
from hermes_plugin_honcho.session import HonchoSessionManager


@pytest.mark.parametrize("existing", [False, True], ids=["new-session", "session-with-messages"])
def test_a_session_joins_its_scopes_once_and_every_peer_observes_only_itself(existing):
    client = MagicMock()
    sdk_session = client.session.return_value
    sdk_session.messages.return_value = SimpleNamespace(items=[object()] if existing else [])
    cfg = HonchoClientConfig(api_key="k", enabled=True, peer_name="user", ai_peer="hermes")
    with patch("hermes_plugin_honcho.session.get_honcho_client", return_value=client):
        mgr = HonchoSessionManager(config=cfg, scopes=["project", "laptop"])
        mgr.get_or_create("chat-1")

    # Scopes only reach a session that has no messages yet; sending them again would copy its memory twice.
    scoped = [c for c in client.session.call_args_list if "scopes" in c.kwargs]
    assert scoped == ([] if existing else [(("chat-1",), {"scopes": ["project", "laptop"]})])
    (user, user_cfg), (ai, ai_cfg) = sdk_session.add_peers.call_args[0][0]
    assert [(c.observe_me, c.observe_others) for c in (user_cfg, ai_cfg)] == [(True, False), (True, False)]
    # A session that already had messages may hold older flags, so they are set explicitly.
    assert sdk_session.set_peer_configuration.call_count == (2 if existing else 0)


def _initialized(card, **init_kwargs):
    cfg = HonchoClientConfig(api_key="k", enabled=True, peer_name="user")
    manager = MagicMock()
    manager.get_peer_card.return_value = card
    manager.dialectic_query.return_value = ""
    provider = HonchoMemoryProvider()
    with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
         patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
         patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=manager) as manager_cls, \
         patch("hermes_plugin_honcho.client.session_scopes", return_value=["project", "laptop"]), \
         patch("hermes_plugin_honcho.session_scopes", return_value=["project", "laptop"]):
        provider.initialize(session_id="20260412_171002_69bb38", **init_kwargs)
        provider._init_thread.join(timeout=5)
    if provider._prefetch_thread:
        provider._prefetch_thread.join(timeout=5)
    return provider, manager, manager_cls


def test_session_start_injects_the_directives_and_the_users_card():
    provider, manager, manager_cls = _initialized(["Name: user", "Prefers short answers"])

    block = provider.system_prompt_block()

    assert manager_cls.call_args.kwargs["scopes"] == ["project", "laptop"]
    assert "call `honcho_chat`" in block and "`project`, `laptop`" in block
    assert block.endswith("User profile: Name: user; Prefers short answers")
    manager.get_peer_card.assert_called_once_with(provider._session_key)


def test_honcho_chat_asks_about_the_user_or_the_workspace():
    provider, manager, _ = _initialized([])
    manager.dialectic_query.return_value = "user answer"
    manager.workspace_chat.return_value = "workspace answer"

    user = json.loads(provider.handle_tool_call("honcho_chat", {"query": "what do I prefer?", "scope": "project"}))
    workspace = json.loads(provider.handle_tool_call("honcho_chat", {"query": "who works here?", "workspace": True}))

    assert (user["result"], workspace["result"]) == ("user answer", "workspace answer")
    _, kwargs = manager.dialectic_query.call_args
    assert (kwargs["peer"], kwargs["reasoning_level"], kwargs["scope"]) == ("user", "medium", "project")
    manager.workspace_chat.assert_called_once_with("who works here?", reasoning_level="medium", scope=None)


@pytest.mark.parametrize("switch, gateway_key, same_session", [
    ({"reset": True, "reason": "new"}, None, False),
    ({"reason": "resume"}, None, False),
    ({"reason": "compression"}, None, True),
    ({"reset": True, "reason": "new"}, "agent:main:telegram:dm:42", True),
], ids=["/new", "/resume", "compression", "gateway-/new"])
def test_a_new_chat_gets_its_own_session(switch, gateway_key, same_session):
    """A Hermes chat is one Honcho session: /new, /resume and /branch open another, compression continues it,
    and a chat the gateway routes keeps its key's session."""
    provider, _, _ = _initialized([], **({"gateway_session_key": gateway_key} if gateway_key else {}))
    before = provider._session_key

    provider.on_session_switch("20260412_180000_aaaaaa", parent_session_id="20260412_171002_69bb38", **switch)

    assert (provider._session_key == before) is same_session
    if not same_session:
        assert provider._session_key == "20260412_180000_aaaaaa" and provider._last_dialectic_turn == -999
