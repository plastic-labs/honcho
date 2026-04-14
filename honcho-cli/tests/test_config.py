"""Tests for config management."""

import json
import os

import pytest
from honcho_cli.config import CLIConfig


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    """Redirect CONFIG_FILE to tmp_path and clear HONCHO_* env vars."""
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    for key in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(key)
    return f


class TestLoad:
    def test_defaults_when_no_file(self, cfg_path):
        loaded = CLIConfig.load()
        assert loaded.base_url == "https://api.honcho.dev"
        assert loaded.api_key == ""
        assert loaded.workspace_id == ""

    def test_malformed_file_uses_defaults(self, cfg_path):
        cfg_path.write_text("not-json{{{")
        assert CLIConfig.load().api_key == ""

    def test_reads_environment_url(self, cfg_path):
        cfg_path.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        loaded = CLIConfig.load()
        assert loaded.base_url == "http://localhost:8000"
        assert loaded.api_key == "k"

    def test_api_key_and_base_url_from_env(self, cfg_path, monkeypatch):
        """HONCHO_API_KEY and HONCHO_BASE_URL override config file at runtime."""
        cfg_path.write_text(json.dumps({"environmentUrl": "https://api.honcho.dev"}))
        monkeypatch.setenv("HONCHO_API_KEY", "env-key")
        monkeypatch.setenv("HONCHO_BASE_URL", "http://localhost:8000")
        loaded = CLIConfig.load()
        assert loaded.api_key == "env-key"
        assert loaded.base_url == "http://localhost:8000"


class TestSave:
    def test_writes_only_cli_owned_keys(self, cfg_path):
        """apiKey + environmentUrl are written; workspace/peer/session are not."""
        CLIConfig(
            base_url="http://localhost:8000",
            api_key="test-key-123",
            workspace_id="my-ws",  # must NOT be persisted
            peer_id="user",
            session_id="s1",
        ).save()
        assert json.loads(cfg_path.read_text()) == {
            "environmentUrl": "http://localhost:8000",
            "apiKey": "test-key-123",
        }

    def test_preserves_foreign_keys(self, cfg_path):
        """Other tools' top-level keys (hosts, sessions, ...) are untouched."""
        seed = {
            "apiKey": "old-key",
            "environmentUrl": "https://api.honcho.dev",
            "saveMessages": True,
            "sessions": {"/Users/user": "home-chat"},
            "hosts": {"claude_code": {"peerName": "user", "workspace": "agents"}},
            "sessionStrategy": "chat-instance",
        }
        cfg_path.write_text(json.dumps(seed))
        cfg = CLIConfig.load()
        cfg.api_key = "new-key"
        cfg.save()

        on_disk = json.loads(cfg_path.read_text())
        assert on_disk["apiKey"] == "new-key"
        assert on_disk["environmentUrl"] == "https://api.honcho.dev"
        for k in ("saveMessages", "sessions", "hosts", "sessionStrategy"):
            assert on_disk[k] == seed[k]


@pytest.mark.parametrize(
    "api_key, check",
    [
        ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdef", lambda v: v.startswith("eyJhbGci") and v.endswith("cdef") and "..." in v),
        ("short", lambda v: v == "***"),
        ("", lambda v: v == ""),
    ],
)
def test_api_key_redaction(api_key, check):
    assert check(CLIConfig(api_key=api_key).redacted()["api_key"])
