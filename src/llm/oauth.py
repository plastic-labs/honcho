"""OAuth token management for OpenAI-compatible providers.

Implements RFC 8628 device code flow for initial authentication and
automatic access-token refresh using a stored refresh token.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://auth.openai.com/oauth/token"
_DEVICE_CODE_URL = "https://auth.openai.com/oauth/device/code"
_DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_SCOPES = "openid profile email offline_access"
_AUDIENCE = "https://api.openai.com/v1"

# Refresh the token when fewer than this many seconds remain before expiry.
_REFRESH_BUFFER_SECONDS = 300


class OAuthTokenManager:
    """Holds an OAuth access token and refreshes it automatically.

    Instantiate once at startup (after calling ``refresh()`` to obtain the
    initial token), then pass to ``OAuthOpenAI``.  A background task should
    call ``run_refresh_loop()`` to keep the token alive.
    """

    def __init__(
        self,
        refresh_token: str,
        client_id: str = _DEFAULT_CLIENT_ID,
        token_file: str | None = None,
    ) -> None:
        self._token_file = Path(token_file) if token_file else None
        # If a token file exists, it holds the most recently rotated token and
        # takes precedence over the value passed in (which may be stale).
        if self._token_file and self._token_file.exists():
            stored = self._token_file.read_text().strip()
            if stored:
                refresh_token = stored
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._access_token: str = ""
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def access_token(self) -> str:
        return self._access_token

    async def refresh(self) -> None:
        """Exchange the refresh token for a new access token."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": self._client_id,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        try:
            self._access_token = data["access_token"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"OAuth token response missing 'access_token': {data}"
            ) from exc
        if "refresh_token" in data:
            self._refresh_token = data["refresh_token"]
            if self._token_file is not None:
                # Write atomically so a crash mid-write doesn't corrupt the file.
                tmp = self._token_file.with_suffix(".tmp")
                tmp.write_text(self._refresh_token)
                tmp.replace(self._token_file)
                logger.debug("Persisted rotated refresh token to %s", self._token_file)
        expires_in: int = data.get("expires_in", 600)
        self._expires_at = time.time() + expires_in
        logger.info(
            "OAuth access token refreshed; expires in %ds", expires_in
        )

    async def refresh_if_needed(self) -> None:
        """Refresh only when within the buffer window of expiry."""
        if time.time() < self._expires_at - _REFRESH_BUFFER_SECONDS:
            return
        async with self._lock:
            if time.time() < self._expires_at - _REFRESH_BUFFER_SECONDS:
                return
            await self.refresh()

    async def run_refresh_loop(self) -> None:
        """Background task: check every 60 seconds and refresh as needed."""
        while True:
            try:
                await self.refresh_if_needed()
            except Exception:
                logger.exception("OAuth token refresh failed; will retry")
            await asyncio.sleep(60)


class OAuthOpenAI(AsyncOpenAI):
    """AsyncOpenAI variant that reads its bearer token from an OAuthTokenManager.

    Both ``_bearer_auth`` (openai >= 2.x) and ``auth_headers`` (openai < 2.x)
    are overridden so the correct hook is called regardless of SDK version.
    The properties are evaluated per-request, so a single cached instance
    automatically picks up refreshed tokens without cache invalidation.
    """

    def __init__(self, token_manager: OAuthTokenManager, **kwargs: Any) -> None:
        # Pass a non-empty placeholder so the SDK does not raise at construction.
        kwargs.setdefault("api_key", "__oauth__")
        super().__init__(**kwargs)
        self._token_manager = token_manager

    @property
    def _bearer_auth(self) -> dict[str, str]:  # openai >= 2.x
        return {"Authorization": f"Bearer {self._token_manager.access_token}"}

    @property
    def auth_headers(self) -> dict[str, str]:  # type: ignore[override]  # openai < 2.x
        return {"Authorization": f"Bearer {self._token_manager.access_token}"}
