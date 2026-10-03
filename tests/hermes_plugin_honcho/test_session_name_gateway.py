"""Unit tests for Hermes plugin gateway session-name resolution.

Hermes runtime modules are stubbed so this file can run without a Hermes
install (hermes-plugin-honcho imports agent.* / hermes_* at module load).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = REPO_ROOT / "hermes-plugin-honcho"
PACKAGE_NAME = "hermes_plugin_honcho_under_test"


def _install_stubs() -> None:
    stubs = {
        "agent": types.ModuleType("agent"),
        "agent.memory_provider": types.ModuleType("agent.memory_provider"),
        "agent.secret_scope": types.ModuleType("agent.secret_scope"),
        "hermes_cli": types.ModuleType("hermes_cli"),
        "hermes_cli.profiles": types.ModuleType("hermes_cli.profiles"),
        "hermes_constants": types.ModuleType("hermes_constants"),
        "hermes_state_common": types.ModuleType("hermes_state_common"),
    }
    stubs["agent.memory_provider"].spawn_context_thread = lambda *a, **k: None
    stubs["agent.secret_scope"].get_secret = lambda *a, **k: None
    stubs["hermes_cli.profiles"]._get_default_hermes_home = lambda: Path("/tmp")
    stubs["hermes_constants"].get_hermes_home = lambda: Path("/tmp")
    stubs["hermes_state_common"].TITLE_SOURCE_DERIVED = "derived"
    stubs["hermes_state_common"].TITLE_SOURCE_LLM = "llm"
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

    if PACKAGE_NAME not in sys.modules:
        pkg = types.ModuleType(PACKAGE_NAME)
        pkg.__path__ = [str(PLUGIN_ROOT)]
        sys.modules[PACKAGE_NAME] = pkg

        cache_mod = types.ModuleType(f"{PACKAGE_NAME}.client_cache")
        cache_mod._DEFAULT_HTTP_TIMEOUT = 30.0
        cache_mod._client_cache_key = lambda *a, **k: ""
        cache_mod._client_slots = {}
        cache_mod._client_slots_lock = __import__("threading").Lock()
        cache_mod._honcho_json_timeout_memo = {}
        cache_mod._refresh_oauth = lambda *a, **k: None
        cache_mod._slot_for = lambda *a, **k: None
        sys.modules[f"{PACKAGE_NAME}.client_cache"] = cache_mod

        spec = importlib.util.spec_from_file_location(
            f"{PACKAGE_NAME}.client",
            PLUGIN_ROOT / "client.py",
            submodule_search_locations=[str(PLUGIN_ROOT)],
        )
        assert spec and spec.loader
        client_mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{PACKAGE_NAME}.client"] = client_mod
        spec.loader.exec_module(client_mod)


@pytest.fixture(scope="module")
def HonchoClientConfig():
    _install_stubs()
    return sys.modules[f"{PACKAGE_NAME}.client"].HonchoClientConfig


class TestResolveSessionNameGatewayKey:
    """Default strategies keep a stable per-chat Honcho session. With
    sessionStrategy=per-session, the Hermes session id is appended so gateway
    /new starts a fresh Honcho session while chats remain isolated (#1285).
    """

    HONCHO_MAX = 100

    def test_per_session_composes_gateway_key_with_session_id(self, HonchoClientConfig):
        config = HonchoClientConfig(session_strategy="per-session")
        result = config.resolve_session_name(
            session_id="20260412_171002_69bb38",
            gateway_session_key="agent:main:telegram:dm:8439114563",
        )
        assert result == "agent-main-telegram-dm-8439114563-20260412_171002_69bb38"

    def test_non_per_session_keeps_stable_gateway_key(self, HonchoClientConfig):
        config = HonchoClientConfig(session_strategy="per-directory")
        result = config.resolve_session_name(
            session_id="20260412_171002_69bb38",
            gateway_session_key="agent:main:telegram:dm:8439114563",
        )
        assert result == "agent-main-telegram-dm-8439114563"

    def test_per_session_gateway_key_without_session_id_stays_stable(self, HonchoClientConfig):
        config = HonchoClientConfig(session_strategy="per-session")
        result = config.resolve_session_name(
            gateway_session_key="agent:main:telegram:dm:8439114563",
        )
        assert result == "agent-main-telegram-dm-8439114563"

    def test_gateway_key_sanitizes_special_chars(self, HonchoClientConfig):
        config = HonchoClientConfig(session_strategy="per-directory")
        result = config.resolve_session_name(
            gateway_session_key="agent:main:telegram:dm:8439114563",
        )
        assert result == "agent-main-telegram-dm-8439114563"

    def test_per_session_composed_long_key_stays_within_limit(self, HonchoClientConfig):
        key = "!roomid:matrix.example.org|" + "$event_" + ("a" * 300)
        config = HonchoClientConfig(session_strategy="per-session")
        result = config.resolve_session_name(
            gateway_session_key=key,
            session_id="20260412_171002_69bb38",
        )
        assert result is not None
        assert len(result) == self.HONCHO_MAX
        other = config.resolve_session_name(
            gateway_session_key=key,
            session_id="20260412_180000_abcdef",
        )
        assert other != result
        assert len(other) == self.HONCHO_MAX
        digest = hashlib.sha256(f"{key}:20260412_171002_69bb38".encode()).hexdigest()[:8]
        assert result.endswith(digest)
