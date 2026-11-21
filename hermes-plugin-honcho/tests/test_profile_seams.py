"""Profile create/rename hooks and the config panel's read_config_fields / write_config_fields."""

import json
from contextlib import contextmanager
from unittest import mock

import pytest
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

from hermes_plugin_honcho import HonchoMemoryProvider


@contextmanager
def _profile_home(path):
    """Bind a profile's home the way Hermes' notify_memory_provider does around a hook."""
    token = set_hermes_home_override(str(path))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_profile_create_rename_and_panel_round_trip():
    default_home = get_hermes_home()
    honcho_json = default_home / "honcho.json"
    honcho_json.write_text(json.dumps({
        "apiKey": "hch-v3-root",
        "hosts": {"hermes": {"workspace": "hermes", "peerName": "user", "recallMode": "tools"}},
    }))
    provider = HonchoMemoryProvider()

    coder = default_home / "profiles" / "coder"
    coder.mkdir(parents=True)
    with _profile_home(coder), mock.patch("hermes_plugin_honcho.cli._ensure_peer_exists", return_value=True):
        provider.on_profile_created("coder", str(coder))
    # The block lands in the file the profile reads; the profile gets no copy of everyone's credentials.
    assert not (coder / "honcho.json").exists()
    block = json.loads(honcho_json.read_text())["hosts"]["hermes_coder"]
    assert block == {"aiPeer": "coder", "workspace": "hermes", "peerName": "user", "recallMode": "tools", "enabled": True}

    writer = default_home / "profiles" / "writer"
    coder.rename(writer)
    with _profile_home(writer):
        provider.on_profile_renamed("coder", "writer", str(writer))
    hosts = json.loads(honcho_json.read_text())["hosts"]
    assert "hermes_coder" not in hosts
    assert hosts["hermes_writer"]["aiPeer"] == "coder"  # the peer keeps its identity

    stored = provider.read_config_fields(str(default_home))
    assert stored["values"]["workspace"] == "hermes"
    assert stored["values"]["apiKey"] == "hch-v3-root"  # host-scoped field falls back to the root
    assert stored["placeholders"] == {"workspace": "hermes", "aiPeer": "hermes"}

    provider.write_config_fields(
        {"workspace": "ws2", "baseUrl": "http://localhost:8000", "recallMode": None, "apiKey": "hch-v3-new"},
        str(default_home),
    )
    cfg = json.loads(honcho_json.read_text())
    assert cfg["baseUrl"] == "http://localhost:8000"  # root-scoped
    assert cfg["hosts"]["hermes"] == {"workspace": "ws2", "peerName": "user", "apiKey": "hch-v3-new"}
    assert cfg["hosts"]["hermes_writer"] == hosts["hermes_writer"]  # other hosts untouched


def test_panel_refuses_to_overwrite_an_unparseable_config():
    honcho_json = get_hermes_home() / "honcho.json"
    honcho_json.write_text('{"hosts": {"hermes": ')
    provider = HonchoMemoryProvider()
    # ValueError is what Hermes turns into a 400, in process and across the plugin host.
    with pytest.raises(ValueError):
        provider.read_config_fields(str(get_hermes_home()))
    with pytest.raises(ValueError):
        provider.write_config_fields({"workspace": "ws2"}, str(get_hermes_home()))
    assert honcho_json.read_text() == '{"hosts": {"hermes": '
