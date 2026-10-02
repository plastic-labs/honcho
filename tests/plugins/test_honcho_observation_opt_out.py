"""Offline regressions for the handed-off Hermes Honcho plugin."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

_PLUGIN_NAME = "honcho_plugin_optout_test"
_PLUGIN_PATH = Path(__file__).resolve().parents[2] / "hermes-plugin-honcho"


def _stub(
    monkeypatch: pytest.MonkeyPatch, name: str, *, package: bool = False, **attrs: Any
) -> types.ModuleType:
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Load the plugin against narrow Hermes API stubs; no network or Hermes install is needed."""
    agent = _stub(monkeypatch, "agent", package=True)

    def sanitize_context(text: str) -> str:
        return re.sub(
            r"<memory-context>.*?</memory-context>", "", text, flags=re.DOTALL
        )

    def spawn_context_thread(target: Callable[..., Any], **kwargs: Any) -> None:
        del target, kwargs

    def a2a_key(author: dict[str, Any]) -> str:
        return str(author.get("id", ""))

    def get_secret(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)

    def redact_sensitive_text(text: str, **_kwargs: Any) -> str:
        return text

    def is_trivial_prompt(_text: str) -> bool:
        return False

    stub_modules: dict[str, dict[str, Any]] = {
        "agent.memory_manager": {"sanitize_context": sanitize_context},
        "agent.memory_provider": {
            "MemoryProvider": type("MemoryProvider", (), {}),
            "is_trivial_prompt": is_trivial_prompt,
            "spawn_context_thread": spawn_context_thread,
        },
        "agent.coding_context": {"INTERACTIVE_CODING_PLATFORMS": frozenset()},
        "agent.turn_author": {"a2a_key": a2a_key},
        "agent.secret_scope": {"get_secret": get_secret},
        "agent.redact": {"redact_sensitive_text": redact_sensitive_text},
    }
    for name, attrs in stub_modules.items():
        child = _stub(monkeypatch, name, **attrs)
        setattr(agent, name.rsplit(".", 1)[1], child)

    _stub(monkeypatch, "hermes_cli", package=True)
    profiles = _stub(
        monkeypatch,
        "hermes_cli.profiles",
        _get_default_hermes_home=lambda: Path.home() / ".hermes",
    )
    sys.modules["hermes_cli"].__dict__["profiles"] = profiles
    _stub(
        monkeypatch, "hermes_constants", get_hermes_home=lambda: Path.home() / ".hermes"
    )
    _stub(
        monkeypatch,
        "hermes_state_common",
        TITLE_SOURCE_DERIVED="derived",
        TITLE_SOURCE_LLM="llm",
    )

    class SingletonSlot:
        pass

    plugins = _stub(monkeypatch, "plugins", package=True)
    plugin_utils = _stub(
        monkeypatch, "plugins.plugin_utils", SingletonSlot=SingletonSlot
    )
    plugins.__dict__["plugin_utils"] = plugin_utils
    tools = _stub(monkeypatch, "tools", package=True)

    def tool_error(text: str) -> str:
        return text

    registry = _stub(monkeypatch, "tools.registry", tool_error=tool_error)
    tools.__dict__["registry"] = registry

    spec = importlib.util.spec_from_file_location(
        _PLUGIN_NAME,
        _PLUGIN_PATH / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_PATH)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _PLUGIN_NAME, package)
    spec.loader.exec_module(package)
    package.__dict__["session_api"] = importlib.import_module(f"{_PLUGIN_NAME}.session")
    try:
        yield package
    finally:
        for name in list(sys.modules):
            if name == _PLUGIN_NAME or name.startswith(f"{_PLUGIN_NAME}."):
                sys.modules.pop(name, None)


def _config(plugin: Any, path: Path, content: dict[str, Any]) -> Any:
    path.write_text(json.dumps(content), encoding="utf-8")
    return plugin.HonchoClientConfig.from_global_config(host="hermes", config_path=path)


def _provider(
    plugin: Any, phrases: list[str], *, limit: int = 25000
) -> tuple[Any, MagicMock]:
    provider = plugin.HonchoMemoryProvider.__new__(plugin.HonchoMemoryProvider)
    provider._config = SimpleNamespace(
        message_max_chars=limit, observation_opt_out_phrases=phrases
    )
    provider._session_key = "telegram:123"
    provider._turn_author = {}
    provider._sync_thread = None
    provider._manager = MagicMock()
    provider._manager.resolve_author_peer_id.return_value = None
    session = MagicMock()
    provider._manager.get_or_create.return_value = session
    provider._writes_enabled = lambda: True
    provider._ready_or_kick_init = lambda: True

    class Completed:
        def __init__(self, fn: Any) -> None:
            fn()

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            del timeout
            return None

    def spawn_write(fn: Callable[[], Any], *_args: Any) -> Completed:
        return Completed(fn)

    provider._spawn_write = spawn_write
    return provider, session


def test_config_opt_out_defaults_empty_and_host_presence_wins(
    plugin: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert (
        _config(plugin, tmp_path / "default.json", {}).observation_opt_out_phrases == []
    )
    assert _config(
        plugin,
        tmp_path / "root.json",
        {
            "observationOptOutPhrases": [" root phrase ", "", 7],
            "hosts": {"hermes": {"observationOptOutPhrases": ["host phrase"]}},
        },
    ).observation_opt_out_phrases == ["host phrase"]
    assert _config(
        plugin,
        tmp_path / "root-fallback.json",
        {
            "observationOptOutPhrases": ["root phrase"],
            "hosts": {"hermes": {"enabled": True}},
        },
    ).observation_opt_out_phrases == ["root phrase"]

    caplog.set_level("WARNING")
    cfg = _config(
        plugin,
        tmp_path / "invalid-host.json",
        {
            "observationOptOutPhrases": ["root phrase"],
            "hosts": {"hermes": {"observationOptOutPhrases": "off"}},
        },
    )
    assert cfg.observation_opt_out_phrases == []
    assert "expected a list of strings" in caplog.text


def test_opt_out_uses_casefold_and_marks_all_chunks_in_both_messages(
    plugin: Any,
) -> None:
    provider, session = _provider(plugin, ["straße"], limit=24)
    provider.sync_turn(
        "I walked down the STRASSE this morning", "Here is a longer assistant answer"
    )

    calls = session.add_message.call_args_list
    assert len(calls) > 2
    assert {call.args[0] for call in calls} == {"user", "assistant"}
    assert all(call.kwargs["no_observe"] is True for call in calls)
    provider._manager.save.assert_called_once_with(session)


def test_opt_out_matches_sanitized_user_text_not_injected_context(plugin: Any) -> None:
    provider, session = _provider(plugin, ["off the record"])
    provider.sync_turn(
        "<memory-context>off the record</memory-context>real question", "answer"
    )

    assert all(
        call.kwargs["no_observe"] is False
        for call in session.add_message.call_args_list
    )
    assert "off the record" not in session.add_message.call_args_list[0].args[1]


def test_sdk_message_configuration_is_a_mapping_and_content_remains_present(
    plugin: Any,
) -> None:
    from honcho import Peer

    content = "sensitive but searchable"
    message: dict[str, Any] = {
        "role": "user",
        "content": content,
        "no_observe": True,
    }
    config = plugin.session_api.HonchoSessionManager._message_configuration_for(message)
    peer = Peer(peer_id="u1", honcho=SimpleNamespace(workspace_id="test-workspace"))
    result = peer.message(content, configuration=config)

    assert type(config) is dict
    assert result.configuration is not None
    assert result.configuration.reasoning is not None
    assert result.configuration.reasoning.enabled is False
    assert result.content == content
    normal = peer.message("ordinary message")
    assert normal.configuration is None


def test_empty_assistant_keeps_user_message_and_respects_default_off(
    plugin: Any,
) -> None:
    provider, session = _provider(plugin, [])
    provider.sync_turn("ordinary user message", "")

    calls = session.add_message.call_args_list
    assert len(calls) == 1
    assert calls[0].args[:2] == ("user", "ordinary user message")
    assert calls[0].kwargs["no_observe"] is False
    provider._manager.save.assert_called_once_with(session)


def test_failed_opt_out_configuration_sends_nothing_and_keeps_message_unsynced(
    plugin: Any,
) -> None:
    manager = plugin.session_api.HonchoSessionManager()
    session = plugin.session_api.HonchoSession(
        key="k", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s"
    )
    session.add_message("user", "sensitive", no_observe=True)
    user_peer, assistant_peer, honcho_session = MagicMock(), MagicMock(), MagicMock()
    user_peer.message.side_effect = RuntimeError("SDK configuration rejected")

    def get_peer(peer_id: str) -> MagicMock:
        return {"u": user_peer, "a": assistant_peer}[peer_id]

    manager._get_or_create_peer = MagicMock(side_effect=get_peer)
    manager._sessions_cache["s"] = honcho_session

    def authed_call(_label: str, fn: Callable[[], Any]) -> Any:
        return fn()

    manager._authed_call = authed_call

    assert manager._flush_session(session) is False
    honcho_session.add_messages.assert_not_called()
    assert session.messages[0].get("_synced") is False


def test_successful_opt_out_batch_builds_real_sdk_messages_and_syncs_cursor(
    plugin: Any,
) -> None:
    from honcho import Peer

    manager = plugin.session_api.HonchoSessionManager()
    session = plugin.session_api.HonchoSession(
        key="k", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s"
    )
    session.add_message("user", "private user", no_observe=True)
    session.add_message("assistant", "private assistant", no_observe=True)
    session.add_message("user", "ordinary authored user", author_peer_id="author")
    session.add_message("assistant", "ordinary assistant")
    peers = {
        peer_id: Peer(
            peer_id=peer_id, honcho=SimpleNamespace(workspace_id="test-workspace")
        )
        for peer_id in ("u", "a", "author")
    }
    uploaded: list[list[Any]] = []
    joined: list[Any] = []
    honcho_session = SimpleNamespace(
        add_messages=lambda messages: uploaded.append(messages),
        add_peers=lambda peers_and_configs: joined.extend(peers_and_configs),
    )
    manager._get_or_create_peer = lambda peer_id: peers[peer_id]
    manager._authed_call = lambda _label, fn: fn()
    manager._sessions_cache["s"] = honcho_session

    assert manager._flush_session(session) is True

    assert len(uploaded) == 1
    batch = uploaded[0]
    assert [(message.peer_id, message.content) for message in batch] == [
        ("u", "private user"),
        ("a", "private assistant"),
        ("author", "ordinary authored user"),
        ("a", "ordinary assistant"),
    ]
    assert batch[0].configuration.reasoning.enabled is False
    assert batch[1].configuration.reasoning.enabled is False
    assert batch[2].configuration is None
    assert batch[3].configuration is None
    assert len(joined) == 1 and joined[0][0].id == "author"
    assert manager._joined_author_peers["s"] == {"author"}
    assert all(message["_synced"] is True for message in session.messages)
