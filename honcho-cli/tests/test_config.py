"""Tests for config management."""

import json
import os
import time
from pathlib import Path

import pytest
from honcho_cli.config import CLIConfig, OAuthTokens, _config_dir
from honcho_cli.oauth import TokenResponse


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    """Redirect CONFIG_FILE to tmp_path and clear HONCHO_* env vars."""
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    for key in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(key)
    return f


class TestConfigDir:
    def test_defaults_to_dot_honcho(self, monkeypatch):
        monkeypatch.delenv("HONCHO_CONFIG_DIR", raising=False)
        assert _config_dir() == Path.home() / ".honcho"

    def test_honcho_config_dir_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HONCHO_CONFIG_DIR", str(tmp_path / "profile"))
        assert _config_dir() == tmp_path / "profile"

    def test_expands_user_in_override(self, monkeypatch):
        monkeypatch.setenv("HONCHO_CONFIG_DIR", "~/.honcho-test")
        assert _config_dir() == Path.home() / ".honcho-test"


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

    def test_empty_env_var_popped_from_environ(self, cfg_path, monkeypatch):
        """Empty HONCHO_* vars are removed so the SDK doesn't crash on them."""
        cfg_path.write_text(json.dumps({"apiKey": "file-key"}))
        monkeypatch.setenv("HONCHO_API_KEY", "")
        loaded = CLIConfig.load()
        assert "HONCHO_API_KEY" not in os.environ
        assert loaded.api_key == "file-key"

    def test_garbage_access_expires_at_treated_as_expired(self, cfg_path):
        """Hand-edited/corrupt expiry degrades to the refresh path, not a crash."""
        cfg_path.write_text(json.dumps(
            {"oauth": {"accessToken": "x", "accessExpiresAt": "not-a-number"}}
        ))
        loaded = CLIConfig.load()
        assert loaded.oauth is not None
        assert loaded.oauth.access_valid() is False

    def test_numeric_string_access_expires_at_parses(self, cfg_path):
        cfg_path.write_text(json.dumps(
            {"oauth": {"accessToken": "x", "accessExpiresAt": "12345"}}
        ))
        loaded = CLIConfig.load()
        assert loaded.oauth is not None
        assert loaded.oauth.access_expires_at == 12345.0


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
    "api_key, expected",
    [
        # Long JWT: only last 4 chars visible, masked prefix.
        ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdef", "***cdef"),
        # Short value > 4 chars: still only last 4.
        ("abcdef", "***cdef"),
        # 4 or fewer chars: fully masked — don't leak the whole key.
        ("abcd", "***"),
        ("x", "***"),
    ],
)
def test_api_key_redaction_shows_last4_only(api_key, expected):
    """Redacted api_key must show ``***<last4>`` at most, never the header/body."""
    assert CLIConfig(api_key=api_key).redacted()["api_key"] == expected


def test_api_key_redaction_empty_omitted():
    """Empty api_key is omitted from redacted output entirely."""
    assert "api_key" not in CLIConfig(api_key="").redacted()


class TestOAuth:
    def _tokens(self, expires_at: float) -> OAuthTokens:
        return OAuthTokens(
            access_token="hch-at-x",
            refresh_token="hch-rt-x",
            access_expires_at=expires_at,
            client_id="honcho-cli",
            scope="write",
        )

    def test_round_trips_oauth_block(self, cfg_path):
        CLIConfig(base_url="http://localhost:8000", oauth=self._tokens(9999999999)).save()
        loaded = CLIConfig.load()
        assert loaded.oauth is not None
        assert loaded.oauth.access_token == "hch-at-x"
        assert loaded.oauth.refresh_token == "hch-rt-x"
        assert loaded.oauth.client_id == "honcho-cli"

    def test_oauth_persists_camelcase_keys(self, cfg_path):
        CLIConfig(base_url="http://localhost:8000", oauth=self._tokens(1234)).save()
        on_disk = json.loads(cfg_path.read_text())["oauth"]
        assert set(on_disk) == {"accessToken", "refreshToken", "accessExpiresAt", "clientId", "scope", "host"}

    def test_save_preserves_foreign_keys_with_oauth(self, cfg_path):
        cfg_path.write_text(json.dumps({"hosts": {"claude_code": {"peerName": "u"}}}))
        CLIConfig(base_url="http://localhost:8000", oauth=self._tokens(1234)).save()
        on_disk = json.loads(cfg_path.read_text())
        assert on_disk["hosts"] == {"claude_code": {"peerName": "u"}}
        assert "oauth" in on_disk

    def test_empty_oauth_is_dropped(self, cfg_path):
        cfg_path.write_text(json.dumps({"oauth": {"accessToken": "old"}}))
        CLIConfig(base_url="http://localhost:8000").save()
        assert "oauth" not in json.loads(cfg_path.read_text())

    def test_api_key_preserved_on_device_login(self, cfg_path):
        """apiKey is shared with sibling tools — device login must not delete it."""
        cfg_path.write_text(json.dumps({"apiKey": "shared-key"}))
        CLIConfig(base_url="http://localhost:8000", oauth=self._tokens(9999999999)).save()
        on_disk = json.loads(cfg_path.read_text())
        assert on_disk["apiKey"] == "shared-key"
        assert on_disk["oauth"]["accessToken"] == "hch-at-x"

    def test_resolved_api_key_prefers_live_oauth(self, cfg_path):
        cfg = CLIConfig(api_key="manual", oauth=self._tokens(9999999999))
        assert cfg.resolved_api_key() == "hch-at-x"

    def test_resolved_api_key_expired_oauth_falls_back_to_api_key(self, cfg_path):
        cfg = CLIConfig(api_key="manual", oauth=self._tokens(time.time() - 100))
        assert cfg.resolved_api_key() == "manual"

    def test_resolved_api_key_host_mismatch_falls_back_to_api_key(self, cfg_path):
        tokens = self._tokens(9999999999)
        tokens.host = "https://staging.example.com"
        cfg = CLIConfig(
            base_url="https://api.honcho.dev", api_key="manual", oauth=tokens
        )
        assert cfg.resolved_api_key() == "manual"

    def test_resolved_api_key_expired_oauth_wins_over_nothing(self, cfg_path):
        cfg = CLIConfig(oauth=self._tokens(time.time() - 100))
        assert cfg.resolved_api_key() == "hch-at-x"

    def test_resolved_api_key_falls_back_to_oauth(self, cfg_path):
        cfg = CLIConfig(oauth=self._tokens(9999999999))
        assert cfg.resolved_api_key() == "hch-at-x"

    def test_access_valid_expiry_and_skew(self):
        assert self._tokens(time.time() + 3600).access_valid()
        assert not self._tokens(time.time() - 10).access_valid()
        # inside the default 60s skew window → treated as invalid
        assert not self._tokens(time.time() + 30).access_valid()

    def test_access_valid_false_without_token(self):
        """A missing token is invalid even with a far-future expiry."""
        tokens = OAuthTokens(access_token="", access_expires_at=time.time() + 3600)
        assert tokens.access_valid() is False

    def test_from_response_keeps_prior_refresh_token_when_not_rotated(self):
        """Refresh-token rotation is optional (RFC 6749 §5.1) — keep the old one."""
        resp = TokenResponse(
            access_token="new-at", refresh_token="", expires_in=3600, scope=""
        )
        tokens = OAuthTokens.from_response(
            resp,
            client_id="honcho-cli",
            scope_fallback="write",
            refresh_fallback="prior-rt",
            host="https://staging.example.com",
        )
        assert tokens.refresh_token == "prior-rt"
        assert tokens.scope == "write"
        assert tokens.host == "https://staging.example.com"

    def test_host_round_trips_and_legacy_matches_all(self, cfg_path):
        tokens = self._tokens(9999999999)
        tokens.host = "https://staging.example.com"
        CLIConfig(base_url="https://staging.example.com", oauth=tokens).save()
        loaded = CLIConfig.load()
        assert loaded.oauth is not None
        assert loaded.oauth.host == "https://staging.example.com"
        # trailing-slash normalization + legacy blocks (no host) trust any host
        assert loaded.oauth.matches_host("https://staging.example.com/")
        assert not loaded.oauth.matches_host("https://api.honcho.dev")
        assert OAuthTokens(access_token="x").matches_host("https://anything.dev")

    def test_redacted_masks_oauth_token(self):
        red = CLIConfig(oauth=self._tokens(1234)).redacted()
        assert red["oauth"] == "***at-x"


def test_save_sets_600_permissions(cfg_path):
    """Config with plaintext API key must be owner-readable only on POSIX."""
    import stat
    CLIConfig(base_url="http://localhost:8000", api_key="sekret").save()
    mode = stat.S_IMODE(os.stat(cfg_path).st_mode)
    # chmod(0o600) → rw- --- ---
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
