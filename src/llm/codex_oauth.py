from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from src.exceptions import ValidationException

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows-only fallback
    fcntl = None  # type: ignore[assignment]


DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"


@dataclass(frozen=True, slots=True)
class CodexOAuthCredentials:
    access_token: str
    base_url: str
    default_headers: dict[str, str]
    auth_path: Path


def resolve_codex_auth_path(configured_path: str | None = None) -> Path:
    """Resolve the Codex OAuth token file.

    Codex CLI stores OAuth credentials in ``$CODEX_HOME/auth.json`` or
    ``~/.codex/auth.json``. A configured directory is treated like CODEX_HOME;
    a configured file path is used directly.
    """

    raw_path = (configured_path or "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_dir() or raw_path.endswith(("/", "\\")):
            path = path / "auth.json"
        return path

    codex_home = os.getenv("CODEX_HOME", "").strip()
    if codex_home:
        return Path(codex_home).expanduser() / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def build_codex_default_headers(access_token: str) -> dict[str, str]:
    """Build first-party Codex headers used by the ChatGPT Codex endpoint."""

    headers = {
        "User-Agent": "codex_cli_rs/0.0.0 (Honcho)",
        "originator": "codex_cli_rs",
    }
    claims = _decode_jwt_claims(access_token)
    auth_claims = claims.get("https://api.openai.com/auth")
    account_id: Any = None
    if isinstance(auth_claims, dict):
        auth_claims = cast(dict[str, Any], auth_claims)
        account_id = auth_claims.get("chatgpt_account_id")
    if not isinstance(account_id, str):
        account_id = claims.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id.strip():
        headers["ChatGPT-Account-ID"] = account_id.strip()
    return headers


def resolve_codex_oauth_credentials(
    *,
    auth_path: str | None = None,
    base_url: str | None = None,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = 120,
    refresh_timeout_seconds: float = 20.0,
) -> CodexOAuthCredentials:
    path = resolve_codex_auth_path(auth_path)
    if not path.is_file():
        message = (
            f"Missing Codex OAuth credentials at {path}. "
            + "Run Codex login so ~/.codex/auth.json exists, or configure "
            + "codex_auth_path."
        )
        raise ValidationException(
            message
        )

    with _auth_file_lock(path):
        payload = _read_auth_payload(path)
        tokens = _extract_tokens(payload)
        access_token = tokens["access_token"]

        if refresh_if_expiring and _access_token_is_expiring(
            access_token,
            refresh_skew_seconds,
        ):
            refresh_token = tokens.get("refresh_token")
            if not refresh_token:
                message = (
                    "Codex OAuth access token is expiring and no refresh_token "
                    + "is present. Re-run Codex login."
                )
                raise ValidationException(
                    message
                )
            refreshed = refresh_codex_access_token(
                refresh_token,
                timeout_seconds=refresh_timeout_seconds,
            )
            tokens.update(refreshed)
            payload["tokens"] = tokens
            payload["last_refresh"] = int(time.time())
            _write_auth_payload(path, payload)
            access_token = tokens["access_token"]

    resolved_base_url = (base_url or DEFAULT_CODEX_BASE_URL).strip().rstrip("/")
    return CodexOAuthCredentials(
        access_token=access_token,
        base_url=resolved_base_url,
        default_headers=build_codex_default_headers(access_token),
        auth_path=path,
    )


def refresh_codex_access_token(
    refresh_token: str,
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, str]:
    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code != 200:
        message = f"Codex OAuth token refresh failed with status {response.status_code}."
        try:
            raw_body: Any = response.json()
        except ValueError:
            body = None
        else:
            body = cast(dict[str, Any], raw_body) if isinstance(raw_body, dict) else None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                error = cast(dict[str, Any], error)
                detail = error.get("message") or error.get("code") or error.get("type")
            else:
                detail = body.get("error_description") or body.get("message") or error
            if isinstance(detail, str) and detail.strip():
                message = f"Codex OAuth token refresh failed: {detail.strip()}"
        raise ValidationException(message)

    try:
        raw_refresh_body: Any = response.json()
    except ValueError as exc:
        raise ValidationException("Codex OAuth token refresh returned invalid JSON") from exc
    body = (
        cast(dict[str, Any], raw_refresh_body)
        if isinstance(raw_refresh_body, dict)
        else {}
    )

    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise ValidationException(
            "Codex OAuth token refresh response did not include access_token"
        )

    updated = {"access_token": access_token.strip()}
    new_refresh = body.get("refresh_token")
    if isinstance(new_refresh, str) and new_refresh.strip():
        updated["refresh_token"] = new_refresh.strip()
    else:
        updated["refresh_token"] = refresh_token
    return updated


def _extract_tokens(payload: dict[str, Any]) -> dict[str, str]:
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise ValidationException(
            "Codex OAuth auth.json is missing a tokens object. Re-run Codex login."
        )
    tokens = cast(dict[str, Any], tokens)
    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise ValidationException(
            "Codex OAuth auth.json is missing access_token. Re-run Codex login."
        )
    refresh_token = tokens.get("refresh_token")
    return {
        "access_token": access_token.strip(),
        **(
            {"refresh_token": refresh_token.strip()}
            if isinstance(refresh_token, str) and refresh_token.strip()
            else {}
        ),
    }


def _read_auth_payload(path: Path) -> dict[str, Any]:
    try:
        raw_payload: Any = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationException(f"Unable to read Codex OAuth credentials at {path}") from exc
    if not isinstance(raw_payload, dict):
        raise ValidationException(f"Codex OAuth credentials at {path} are invalid")
    return cast(dict[str, Any], raw_payload)


def _write_auth_payload(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        raise ValidationException(f"Unable to update Codex OAuth credentials at {path}") from exc


def _access_token_is_expiring(access_token: str, skew_seconds: int) -> bool:
    exp = _decode_jwt_claims(access_token).get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return time.time() + max(0, int(skew_seconds)) >= float(exp)


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded: Any = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, TypeError):
        return {}
    return cast(dict[str, Any], decoded) if isinstance(decoded, dict) else {}


@contextmanager
def _auth_file_lock(path: Path) -> Generator[None, None, None]:
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
