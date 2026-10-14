"""Client-identity headers the plugin attaches to every Honcho request."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import hermes_plugin_honcho.client as client_mod
from hermes_plugin_honcho.client import HonchoClientConfig, get_honcho_client, telemetry_headers


@pytest.fixture(autouse=True)
def _fresh_client_cache():
    client_mod.reset_honcho_client()
    yield
    client_mod.reset_honcho_client()


def test_plugin_header_carries_the_manifest_version(tmp_path, monkeypatch):
    (tmp_path / "plugin.yaml").write_text('name: honcho\nversion: "9.8.7"\n')
    monkeypatch.setattr(client_mod, "__file__", str(tmp_path / "client.py"))

    assert telemetry_headers()["X-Honcho-Plugin"] == "hermes-plugin-honcho/9.8.7"


def test_plugin_header_matches_the_shipped_manifest():
    manifest = Path(client_mod.__file__).with_name("plugin.yaml").read_text()
    version = next(line.split(":", 1)[1].strip() for line in manifest.splitlines() if line.startswith("version:"))

    assert telemetry_headers()["X-Honcho-Plugin"] == f"hermes-plugin-honcho/{version}"


@pytest.mark.parametrize("manifest", [None, "name: honcho\n", "version:\n"])
def test_plugin_header_falls_back_to_unknown(tmp_path, monkeypatch, manifest):
    if manifest is not None:
        (tmp_path / "plugin.yaml").write_text(manifest)
    monkeypatch.setattr(client_mod, "__file__", str(tmp_path / "client.py"))

    assert telemetry_headers()["X-Honcho-Plugin"] == "hermes-plugin-honcho/unknown"


def test_host_header_includes_the_hermes_version(monkeypatch):
    monkeypatch.setattr(client_mod, "_hermes_version", lambda: "0.21.3")

    assert telemetry_headers()["X-Honcho-Host"] == f"hermes/0.21.3 ({sys.platform})"


def test_host_header_omits_an_unresolvable_hermes_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli", None)  # makes `from hermes_cli import ...` raise

    assert client_mod._hermes_version() == ""
    assert telemetry_headers()["X-Honcho-Host"] == f"hermes ({sys.platform})"


def test_built_client_sends_the_telemetry_headers():
    cfg = HonchoClientConfig(api_key="k", base_url="http://localhost:8000", workspace_id="hermes")

    with patch("honcho.Honcho", return_value=MagicMock(name="Honcho")) as mock_honcho:
        get_honcho_client(cfg)

    assert mock_honcho.call_args.kwargs["default_headers"] == telemetry_headers()
