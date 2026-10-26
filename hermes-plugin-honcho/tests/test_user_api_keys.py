"""Sender-bound credentials must never inherit another sender's key or base OAuth."""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import (
    HonchoClientConfig,
    close_honcho_clients,
    get_honcho_client,
)
from hermes_plugin_honcho.client_cache import _client_slots
from hermes_plugin_honcho.session import HonchoAuthError, HonchoSessionManager
from honcho.peer import Peer


@pytest.fixture(autouse=True)
def clients():
    close_honcho_clients()
    yield
    close_honcho_clients()


def config_file(tmp_path, **values):
    path = tmp_path / "honcho.json"
    path.write_text(json.dumps(values))
    return HonchoClientConfig.from_global_config(config_path=path)


def test_host_map_replaces_root_and_map_alone_enables(tmp_path):
    cfg = config_file(tmp_path, userApiKeys={"root": "key-root"},
                      hosts={"hermes": {"userApiKeys": {"a": "$KEY_A"}}})
    assert cfg.enabled
    assert cfg.user_api_keys == {"a": "$KEY_A"}
    cleared = config_file(tmp_path, userApiKeys={"root": "key-root"},
                          hosts={"hermes": {"userApiKeys": {}}})
    assert cleared.user_api_keys == {}
    assert not cleared.enabled


@pytest.mark.parametrize(("user_id", "alt", "expected"), [
    ("a", None, "key-a"),
    ("b", None, "key-b"),
    ("alias-only", None, "key-alias"),
    ("a", "b", "key-a"),
    ("alias-only", "b", "key-b"),  # all direct IDs precede alias targets
    ("unknown", "alias-only", "key-alias"),
    ("unknown", None, "base"),
    (None, None, "base"),
])
def test_precedence_and_shared_config_unchanged(user_id, alt, expected):
    cfg = HonchoClientConfig(enabled=True, api_key="base", pin_peer_name=True,
                             peer_name="owner", user_peer_aliases={"a": "alice", "b": "bob", "alias-only": "alice"},
                             user_api_keys={"a": "key-a", "b": "key-b", "alice": "key-alias", "owner": "owner-key"})
    bound = cfg.for_user(user_id, alt)
    assert bound.api_key == expected
    assert bound is not cfg
    assert cfg.api_key == "base"
    assert cfg.credential_user_id is None


@pytest.mark.parametrize("value", ["$MISSING", "", "  ", None, 42, {}])
def test_selected_missing_or_invalid_key_never_falls_back(tmp_path, value):
    cfg = config_file(tmp_path, apiKey="base", baseUrl="http://127.0.0.1:8959",
                      userApiKeys={"a": value, "alice": "alias-key"}, userPeerAliases={"a": "alice"})
    bound = cfg.for_user("a")
    assert bound.user_api_key_selected
    assert not bound.enabled
    assert not bound.api_key
    with pytest.raises(ValueError, match="inactive"):
        get_honcho_client(bound)


def test_secret_scope_is_captured_before_background_work(tmp_path, monkeypatch):
    monkeypatch.setenv("SCOPED_KEY", "wrong-ambient-key")
    cfg = config_file(tmp_path, userApiKeys={"a": "$SCOPED_KEY"})
    token = set_secret_scope({"SCOPED_KEY": "scoped-key"})
    try:
        bound = cfg.for_user("a")
    finally:
        reset_secret_scope(token)
    with ThreadPoolExecutor() as pool:
        client = pool.submit(get_honcho_client, bound).result()
    assert client._http.api_key == "scoped-key"
    assert bound.for_user("a") is bound
    assert cfg.api_key is None


def test_missing_secret_in_multiplex_scope_does_not_use_ambient_key(monkeypatch):
    import agent.secret_scope as secret_scope

    monkeypatch.setenv("USER_API_KEY", "wrong-profile-key")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    cfg = HonchoClientConfig(enabled=True, api_key="base",
                             user_api_keys={"a": "$USER_API_KEY"})
    token = set_secret_scope({})
    try:
        assert not cfg.for_user("a").enabled
        assert cfg.for_user("a").api_key is None
    finally:
        reset_secret_scope(token)


def test_bound_config_cannot_be_reused_for_another_sender():
    cfg = HonchoClientConfig(enabled=True, user_api_keys={"a": "key-a"})
    bound = cfg.for_user("a")
    with pytest.raises(ValueError, match="another user"):
        HonchoSessionManager(config=bound, runtime_user_peer_name="b")


@pytest.mark.parametrize("base_url", [None, "http://127.0.0.1:8959", "http://192.168.1.1:8959"])
def test_explicit_mapped_key_and_silent_unknown(tmp_path, monkeypatch, base_url):
    cfg = config_file(tmp_path, baseUrl=base_url, userApiKeys={"a": "key-a"}, recallMode="tools")
    monkeypatch.setattr(HonchoClientConfig, "from_global_config", lambda **kw: cfg)
    provider = HonchoMemoryProvider()
    assert provider.is_available()
    provider.initialize("session", user_id="unknown", platform="slack")
    provider.on_turn_start(1, "Synthetic user message", author_id="unknown")
    provider.sync_turn("Synthetic user message", "Synthetic answer")
    provider.queue_prefetch("Synthetic question")
    assert provider.prefetch("Synthetic question") == ""
    assert provider.system_prompt_block() == ""
    assert provider.get_tool_schemas() == []
    assert provider._pop_auth_notice() == ""
    assert provider._manager is None
    assert get_honcho_client(cfg.for_user("a"))._http.api_key == "key-a"


def test_no_map_preserves_local_behavior_and_explicit_disabled(tmp_path):
    local = config_file(tmp_path, baseUrl="http://127.0.0.1:8959")
    assert local.for_user("unknown") is local
    assert get_honcho_client(local)._http.api_key == "local"
    disabled = config_file(tmp_path, enabled=False, userApiKeys={"a": "key-a"})
    assert not disabled.for_user("a").enabled


def test_oauth_cannot_replace_selected_keys_on_build_cache_or_401(tmp_path, monkeypatch):
    import hermes_plugin_honcho.oauth as oauth

    refresh = Mock(return_value=("broad-oauth", True))
    forced = Mock(return_value="broad-oauth")
    monkeypatch.setattr(oauth, "ensure_fresh_token", refresh)
    monkeypatch.setattr(oauth, "force_refresh_token", forced)
    monkeypatch.setattr(oauth, "any_dead_grants", lambda: True)
    monkeypatch.setattr(oauth, "reauth_required", lambda *a: True)
    cfg = config_file(tmp_path, userApiKeys={"a": "key-a", "b": "key-b"},
                      hosts={"hermes": {"apiKey": "base", "oauth": {"refreshToken": "refresh-base"}}})
    a, b = (HonchoSessionManager(config=cfg, runtime_user_peer_name=uid) for uid in ("a", "b"))
    assert a.honcho._http.api_key == "key-a"
    assert b.honcho._http.api_key == "key-b"
    assert a.honcho is not b.honcho
    assert a.honcho._http.api_key == "key-a"
    assert not a._reauth_required()

    class Unauthorized(Exception):
        status_code = 401

    operation = Mock(side_effect=Unauthorized("HTTP 401"))
    with pytest.raises(HonchoAuthError):
        a._authed_call("test", operation)
    operation.assert_called_once()
    refresh.assert_not_called()
    forced.assert_not_called()
    assert cfg.api_key == "base"
    assert a.honcho._http.api_key == "key-a"


def test_clients_and_manager_caches_stay_isolated_under_concurrency(tmp_path):
    cfg = config_file(tmp_path, userApiKeys={"a": "key-a", "b": "key-b"})
    managers = [HonchoSessionManager(config=cfg, runtime_user_peer_name=uid) for uid in ("a", "b")]

    def get(index):
        manager = managers[index % 2]
        client = manager.honcho
        peer = manager._cached_sdk_object(manager._peers_cache, "shared-name", lambda: Peer("shared-name", client))
        return index % 2, client, peer

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(get, range(100)))
    for index, client, peer in results:
        assert client is managers[index].honcho
        assert client._http.api_key == ("key-a", "key-b")[index]
        assert peer is managers[index]._peers_cache["shared-name"]
    assert managers[0]._peers_cache["shared-name"] is not managers[1]._peers_cache["shared-name"]
    assert len(_client_slots) == 2
    assert cfg.api_key is None


def test_rotation_evicts_only_that_sender(tmp_path):
    cfg = config_file(tmp_path, userApiKeys={"a": "key-a", "b": "key-b"})
    old_a = get_honcho_client(cfg.for_user("a"))
    client_b = get_honcho_client(cfg.for_user("b"))
    cfg.user_api_keys["a"] = "key-a-new"
    new_a = get_honcho_client(cfg.for_user("a"))
    assert old_a is not new_a
    assert client_b is get_honcho_client(cfg.for_user("b"))
    assert new_a._http.api_key == "key-a-new"
    assert len(_client_slots) == 2


def test_provider_binds_before_lazy_init_and_signature_hides_keys(tmp_path, monkeypatch):
    cfg = config_file(tmp_path, userApiKeys={"a": "key-a"}, recallMode="tools")
    monkeypatch.setattr(HonchoClientConfig, "from_global_config", lambda **kw: cfg)
    provider = HonchoMemoryProvider()
    provider.initialize("session", user_id="a", platform="slack")
    assert provider._config.api_key == "key-a"
    assert cfg.api_key is None
    signature = provider.identity_signature()
    assert "key-a" not in json.dumps(signature)
    assert signature["user_api_keys"]


def test_signature_changes_on_credential_mapping_edit(tmp_path, monkeypatch):
    path = tmp_path / "honcho.json"
    monkeypatch.setattr("hermes_plugin_honcho.resolve_config_path", lambda: path)
    provider = HonchoMemoryProvider()
    path.write_text(json.dumps({"userApiKeys": {"a": "key-a"}}))
    before = provider.identity_signature()
    path.write_text(json.dumps({"userApiKeys": {"a": "replacement-key-a"}}))
    after = provider.identity_signature()
    assert before["user_api_keys"] != after["user_api_keys"]
    assert "replacement-key-a" not in json.dumps(after)
