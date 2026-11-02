"""Tests for the client factory's transparent OAuth refresh."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest
import typer
from honcho_cli import common
from honcho_cli.config import CLIConfig, OAuthTokens
from honcho_cli.main import app
from honcho_cli.oauth import OAuthFlowError, TokenResponse
from typer.testing import CliRunner


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    for k in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(k)
    return f


def _cfg(expires_at: float) -> CLIConfig:
    return CLIConfig(
        base_url="http://localhost:8000",
        oauth=OAuthTokens(
            access_token="old-at",
            refresh_token="old-rt",
            access_expires_at=expires_at,
            client_id="honcho-cli",
            scope="write",
        ),
    )


def test_valid_token_is_not_refreshed(cfg_path):
    config = _cfg(time.time() + 3600)
    with patch("honcho_cli.oauth.refresh_access_token") as refresh:
        common.maybe_refresh_token(config)
    refresh.assert_not_called()


def test_expired_token_refreshes_despite_manual_key(cfg_path):
    """OAuth wins over apiKey now, so the grant is kept alive even with a key set."""
    config = _cfg(time.time() - 100)
    config.api_key = "manual"
    rotated = TokenResponse(
        access_token="new-at", refresh_token="new-rt", expires_in=3600, scope="write"
    )
    with patch("honcho_cli.oauth.refresh_access_token", return_value=rotated) as refresh:
        common.maybe_refresh_token(config)
    refresh.assert_called_once()
    assert config.resolved_api_key() == "new-at"


def test_host_mismatch_skips_refresh(cfg_path):
    """A grant minted for another host is ignored — no refresh, apiKey covers this one."""
    config = _cfg(time.time() - 100)
    config.oauth.host = "https://staging.example.com"
    config.api_key = "manual"
    with patch("honcho_cli.oauth.refresh_access_token") as refresh:
        common.maybe_refresh_token(config)
    refresh.assert_not_called()
    assert config.resolved_api_key() == "manual"


def test_expired_token_refreshes_and_persists(cfg_path):
    config = _cfg(time.time() - 100)
    rotated = TokenResponse(
        access_token="new-at",
        refresh_token="new-rt",
        expires_in=3600,
        scope="write",
    )
    with patch("honcho_cli.oauth.refresh_access_token", return_value=rotated) as refresh:
        common.maybe_refresh_token(config)

    # used the stored refresh token + client_id
    _endpoints, sent_rt = refresh.call_args.args
    assert sent_rt == "old-rt"
    assert _endpoints.client_id == "honcho-cli"

    # in-memory config updated with the rotated pair
    assert config.oauth.access_token == "new-at"
    assert config.oauth.refresh_token == "new-rt"
    assert config.oauth.access_valid()

    # rotation persisted to disk before reuse
    on_disk = json.loads(cfg_path.read_text())["oauth"]
    assert on_disk["accessToken"] == "new-at"
    assert on_disk["refreshToken"] == "new-rt"


def test_refresh_does_not_persist_env_overrides(cfg_path, monkeypatch):
    """A refresh writes the new grant only, never the HONCHO_BASE_URL overlay.

    Sibling tools read ``environmentUrl`` and ``apiKey`` from the shared file,
    so a leaked override silently moves them to another server or org. (With
    HONCHO_API_KEY set the refresh is skipped altogether.) A legacy grant with
    no recorded host is trusted on any URL, so it refreshes under the override.
    """
    original = {
        "apiKey": "file-key",
        "environmentUrl": "http://localhost:8000",
        "oauth": {
            "accessToken": "old-at",
            "refreshToken": "old-rt",
            "accessExpiresAt": time.time() - 100,
            "clientId": "honcho-cli",
            "scope": "write",
        },
        "hosts": {"harness": {"workspace": "harness-honcho"}},
    }
    cfg_path.write_text(json.dumps(original))
    monkeypatch.setenv("HONCHO_BASE_URL", "http://127.0.0.1:9000")
    rotated = TokenResponse(
        access_token="new-at", refresh_token="new-rt", expires_in=3600, scope="write"
    )
    client = MagicMock()
    client.workspaces.return_value = ["ws-a"]
    with (
        patch("honcho_cli.oauth.refresh_access_token", return_value=rotated),
        patch("honcho_cli.common.Honcho", return_value=client),
    ):
        result = CliRunner().invoke(app, ["workspace", "list"])
    assert result.exit_code == 0, result.output

    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["apiKey"] == "file-key"
    assert on_disk["environmentUrl"] == "http://localhost:8000"
    assert on_disk["hosts"] == original["hosts"]
    assert on_disk["oauth"]["accessToken"] == "new-at"
    assert on_disk["oauth"]["refreshToken"] == "new-rt"
    # bound to the override it was minted against, not left hostless for the saved URL
    assert on_disk["oauth"]["host"] == "http://127.0.0.1:9000"


def test_env_api_key_beats_live_oauth_grant(cfg_path, monkeypatch):
    """HONCHO_API_KEY is an explicit per-run choice, so it outranks the saved grant."""
    cfg_path.write_text(json.dumps({
        "environmentUrl": "http://localhost:8000",
        "oauth": {
            "accessToken": "oauth-at",
            "refreshToken": "oauth-rt",
            "accessExpiresAt": time.time() + 3600,
            "host": "http://localhost:8000",
        },
    }))
    monkeypatch.setenv("HONCHO_API_KEY", "other-org-key")
    client = MagicMock()
    client.workspaces.return_value = ["ws-a"]
    with patch("honcho_cli.common.Honcho", return_value=client) as honcho:
        result = CliRunner().invoke(app, ["workspace", "list"])
    assert result.exit_code == 0, result.output
    assert honcho.call_args.kwargs["api_key"] == "other-org-key"


def test_env_api_key_skips_refresh(cfg_path, monkeypatch):
    """The env key wins, so an expired grant it won't use is neither refreshed nor saved."""
    cfg_path.write_text(json.dumps(
        {"oauth": {"accessToken": "old-at", "refreshToken": "old-rt", "accessExpiresAt": 0}}
    ))
    before = cfg_path.read_text()
    monkeypatch.setenv("HONCHO_API_KEY", "other-org-key")
    config = CLIConfig.load()
    with patch("honcho_cli.oauth.refresh_access_token") as refresh:
        common.maybe_refresh_token(config)
    refresh.assert_not_called()
    assert cfg_path.read_text() == before


def test_refresh_failure_exits(cfg_path):
    config = _cfg(time.time() - 100)
    with patch("honcho_cli.oauth.refresh_access_token", side_effect=OAuthFlowError("invalid_grant")):
        with pytest.raises(typer.Exit):
            common.maybe_refresh_token(config)


def test_refresh_failure_falls_back_to_api_key(cfg_path):
    """A dead grant degrades to the saved apiKey instead of aborting."""
    config = _cfg(time.time() - 100)
    config.api_key = "manual"
    with patch("honcho_cli.oauth.refresh_access_token", side_effect=OAuthFlowError("invalid_grant")):
        common.maybe_refresh_token(config)  # must not raise
    assert config.resolved_api_key() == "manual"


def test_missing_refresh_token_exits(cfg_path):
    config = _cfg(time.time() - 100)
    config.oauth.refresh_token = ""
    with pytest.raises(typer.Exit):
        common.maybe_refresh_token(config)


def test_missing_refresh_token_falls_back_to_api_key(cfg_path):
    config = _cfg(time.time() - 100)
    config.oauth.refresh_token = ""
    config.api_key = "manual"
    common.maybe_refresh_token(config)  # must not raise
    assert config.resolved_api_key() == "manual"
