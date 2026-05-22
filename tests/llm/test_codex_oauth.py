from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest

from src.llm import codex_oauth


def _fake_jwt(claims: dict[str, object]) -> str:
    def encode(part: dict[str, object]) -> str:
        raw = json.dumps(part).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode(claims)}.sig"


def test_resolve_codex_oauth_credentials_reads_codex_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_dir = tmp_path / "codex"
    auth_dir.mkdir()
    access_token = _fake_jwt(
        {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_test",
            },
        }
    )
    (auth_dir / "auth.json").write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-token",
                }
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(auth_dir))

    credentials = codex_oauth.resolve_codex_oauth_credentials()

    assert credentials.access_token == access_token
    assert credentials.base_url == codex_oauth.DEFAULT_CODEX_BASE_URL
    assert credentials.auth_path == auth_dir / "auth.json"
    assert credentials.default_headers["originator"] == "codex_cli_rs"
    assert credentials.default_headers["ChatGPT-Account-ID"] == "acct_test"


def test_resolve_codex_oauth_credentials_refreshes_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    expired_token = _fake_jwt({"exp": int(time.time()) - 60})
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": expired_token,
                    "refresh_token": "old-refresh",
                }
            }
        )
    )

    refreshed_token = _fake_jwt({"exp": int(time.time()) + 3600})

    def fake_refresh(refresh_token: str, *, timeout_seconds: float) -> dict[str, str]:
        assert refresh_token == "old-refresh"
        assert timeout_seconds == 3.0
        return {
            "access_token": refreshed_token,
            "refresh_token": "new-refresh",
        }

    monkeypatch.setattr(codex_oauth, "refresh_codex_access_token", fake_refresh)

    credentials = codex_oauth.resolve_codex_oauth_credentials(
        auth_path=str(auth_path),
        refresh_timeout_seconds=3.0,
    )

    saved = json.loads(auth_path.read_text())
    assert credentials.access_token == refreshed_token
    assert saved["tokens"]["access_token"] == refreshed_token
    assert saved["tokens"]["refresh_token"] == "new-refresh"
