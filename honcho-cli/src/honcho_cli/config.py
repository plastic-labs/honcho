"""Configuration management for Honcho CLI.

Config stored at ``~/.honcho/config.json`` with env var overrides.

The CLI owns these top-level keys in that file:

    apiKey          -- Honcho admin JWT (manual auth)
    environmentUrl  -- Honcho API URL (full URL, e.g. https://api.honcho.dev)
    oauth           -- OAuth device-grant tokens (accessToken, refreshToken,
                       accessExpiresAt, clientId, scope), written by device login

All other top-level keys (``hosts``, ``sessions``, ``saveMessages``,
``sessionStrategy``, …) are written by sibling Honcho tools and are
preserved untouched on save.

Workspace / peer / session scoping is intentionally *not* persisted here —
pass ``-w`` / ``-p`` / ``-s`` flags or set ``HONCHO_WORKSPACE_ID`` /
``HONCHO_PEER_ID`` / ``HONCHO_SESSION_ID`` per command instead.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from honcho_cli.oauth import TokenResponse

CONFIG_DIR = Path.home() / ".honcho"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_BASE_URL = "https://api.honcho.dev"


def _redact_token(token: str) -> str:
    """Show ``***<last4>`` — enough to compare tokens without leaking the body."""
    if not token:
        return ""
    return "***" + token[-4:] if len(token) > 4 else "***"


def _coerce_epoch(value: object) -> float:
    """Parse a persisted epoch-seconds value, treating garbage as expired (0)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0

# Env var mapping for runtime overrides.
#
# Resolution order: flag > env var > config file > default.
ENV_MAP: dict[str, str] = {
    "api_key": "HONCHO_API_KEY",
    "base_url": "HONCHO_BASE_URL",
    "workspace_id": "HONCHO_WORKSPACE_ID",
    "peer_id": "HONCHO_PEER_ID",
    "session_id": "HONCHO_SESSION_ID",
}


@dataclass
class OAuthTokens:
    """Device-grant tokens persisted under the config ``oauth`` key."""

    access_token: str = ""
    refresh_token: str = ""
    access_expires_at: float = 0.0  # epoch seconds
    client_id: str = ""
    scope: str = ""

    def access_valid(self, skew: int = 60) -> bool:
        """True while the access token is present and not within ``skew`` of expiry."""
        return bool(self.access_token) and time.time() < self.access_expires_at - skew

    @classmethod
    def from_response(
        cls,
        resp: TokenResponse,
        *,
        client_id: str,
        scope_fallback: str = "",
        refresh_fallback: str = "",
    ) -> OAuthTokens:
        """Build persisted tokens from a token response.

        ``refresh_fallback`` keeps the prior refresh token when the server
        doesn't rotate one (optional on the refresh grant, RFC 6749 §5.1).
        """
        return cls(
            access_token=resp.access_token,
            refresh_token=resp.refresh_token or refresh_fallback,
            access_expires_at=time.time() + resp.expires_in,
            client_id=client_id,
            scope=resp.scope or scope_fallback,
        )


@dataclass
class CLIConfig:
    """CLI configuration with layered resolution: flag > env > file > default.

    ``workspace_id`` / ``peer_id`` / ``session_id`` exist on this dataclass so
    flag/env overrides flow through ``get_client_kwargs()``, but they are
    never read from or written to the config file — they're per-command.
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    workspace_id: str = ""
    peer_id: str = ""
    session_id: str = ""
    oauth: OAuthTokens | None = None

    def resolved_api_key(self) -> str:
        """The key handed to the SDK: manual apiKey wins, else the OAuth token."""
        if self.api_key:
            return self.api_key
        if self.oauth and self.oauth.access_token:
            return self.oauth.access_token
        return ""

    @classmethod
    def load(cls) -> CLIConfig:
        """Load config from file, then overlay env vars."""
        config = cls()

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}

            if isinstance(data, dict):
                url = data.get("environmentUrl")
                if isinstance(url, str) and url:
                    config.base_url = url
                key = data.get("apiKey")
                if isinstance(key, str):
                    config.api_key = key
                oauth = data.get("oauth")
                if isinstance(oauth, dict) and oauth.get("accessToken"):
                    config.oauth = OAuthTokens(
                        access_token=str(oauth.get("accessToken", "")),
                        refresh_token=str(oauth.get("refreshToken", "")),
                        access_expires_at=_coerce_epoch(oauth.get("accessExpiresAt")),
                        client_id=str(oauth.get("clientId", "")),
                        scope=str(oauth.get("scope", "")),
                    )

        for fld_name, env_var in ENV_MAP.items():
            val = os.environ.get(env_var)
            if val:
                setattr(config, fld_name, val)
            elif val == "":
                # SDK reads these env vars directly and crashes on empty
                # strings with a Pydantic ValidationError. Drop them so the
                # SDK falls back to kwargs / defaults.
                os.environ.pop(env_var, None)

        return config

    def save(self) -> None:
        """Write ``apiKey`` + ``environmentUrl`` to config.json.

        Preserves unrelated top-level keys (``hosts``, ``sessions``,
        ``saveMessages``, ``sessionStrategy``, …) that other tools write.
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        data: dict = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
            except (json.JSONDecodeError, OSError):
                data = {}

        data["environmentUrl"] = self.base_url
        if self.api_key:
            data["apiKey"] = self.api_key
        else:
            data.pop("apiKey", None)

        if self.oauth and self.oauth.access_token:
            data["oauth"] = {
                "accessToken": self.oauth.access_token,
                "refreshToken": self.oauth.refresh_token,
                "accessExpiresAt": self.oauth.access_expires_at,
                "clientId": self.oauth.client_id,
                "scope": self.oauth.scope,
            }
        else:
            data.pop("oauth", None)

        CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")
        # API key in plaintext — restrict to the owner on multi-user hosts.
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except OSError:
            pass

    def redacted(self) -> dict[str, str]:
        """Return config dict with api_key redacted.

        Only includes fields that have a value set — per-command fields
        (workspace_id, peer_id, session_id) are omitted when empty.
        """
        d: dict[str, str] = {}
        for fld in fields(self):
            val = getattr(self, fld.name)
            if not val:
                continue
            if fld.name == "api_key":
                # Show ``***<last4>`` only — enough to compare keys without
                # leaking the header or body of the JWT.
                d[fld.name] = _redact_token(val)
            elif fld.name == "oauth":
                d[fld.name] = _redact_token(val.access_token)
            else:
                d[fld.name] = val
        return d


def get_client_kwargs(config: CLIConfig) -> dict:
    """Build kwargs for Honcho client from config."""
    kwargs: dict = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    api_key = config.resolved_api_key()
    if api_key:
        kwargs["api_key"] = api_key
    if config.workspace_id:
        kwargs["workspace_id"] = config.workspace_id
    return kwargs
