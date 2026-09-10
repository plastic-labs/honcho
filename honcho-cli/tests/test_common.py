"""Tests for the client factory's transparent OAuth refresh."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest
import typer
from honcho_cli import common
from honcho_cli.config import CLIConfig, OAuthTokens
from honcho_cli.oauth import OAuthFlowError, TokenResponse


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
