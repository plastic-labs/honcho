"""Tests for src/llm/oauth.py — OAuthTokenManager and OAuthOpenAI."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.oauth import OAuthOpenAI, OAuthTokenManager


# ---------------------------------------------------------------------------
# OAuthTokenManager
# ---------------------------------------------------------------------------


def _make_token_response(
    access_token: str = "access-123",
    expires_in: int = 600,
    refresh_token: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "access_token": access_token,
        "expires_in": expires_in,
    }
    if refresh_token is not None:
        data["refresh_token"] = refresh_token
    return data


def _mock_httpx_post(response_data: dict[str, Any], status_code: int = 200) -> Any:
    """Return an async context manager mock whose .post() returns a fake response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client


@pytest.mark.asyncio
async def test_refresh_updates_access_token() -> None:
    manager = OAuthTokenManager(refresh_token="rt-abc", client_id="client-1")
    assert manager.access_token == ""

    with patch(
        "src.llm.oauth.httpx.AsyncClient",
        return_value=_mock_httpx_post(_make_token_response("new-token", 3600)),
    ):
        await manager.refresh()

    assert manager.access_token == "new-token"


@pytest.mark.asyncio
async def test_refresh_updates_refresh_token_when_rotated() -> None:
    manager = OAuthTokenManager(refresh_token="rt-old", client_id="client-1")

    with patch(
        "src.llm.oauth.httpx.AsyncClient",
        return_value=_mock_httpx_post(
            _make_token_response("tok", refresh_token="rt-new")
        ),
    ):
        await manager.refresh()

    # Access the private attribute to verify rotation.
    assert manager._refresh_token == "rt-new"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_refresh_if_needed_skips_when_token_is_fresh() -> None:
    manager = OAuthTokenManager(refresh_token="rt-abc", client_id="client-1")
    # Manually set a far-future expiry so the token appears fresh.
    manager._access_token = "existing-token"  # pyright: ignore[reportPrivateUsage]
    manager._expires_at = time.time() + 3600  # pyright: ignore[reportPrivateUsage]

    mock_client = _mock_httpx_post(_make_token_response())
    with patch("src.llm.oauth.httpx.AsyncClient", return_value=mock_client):
        await manager.refresh_if_needed()

    mock_client.post.assert_not_called()
    assert manager.access_token == "existing-token"


@pytest.mark.asyncio
async def test_refresh_if_needed_refreshes_when_near_expiry() -> None:
    manager = OAuthTokenManager(refresh_token="rt-abc", client_id="client-1")
    manager._access_token = "old-token"  # pyright: ignore[reportPrivateUsage]
    # Expire in 60 s — within the 300 s buffer.
    manager._expires_at = time.time() + 60  # pyright: ignore[reportPrivateUsage]

    with patch(
        "src.llm.oauth.httpx.AsyncClient",
        return_value=_mock_httpx_post(_make_token_response("refreshed-token")),
    ):
        await manager.refresh_if_needed()

    assert manager.access_token == "refreshed-token"


@pytest.mark.asyncio
async def test_refresh_if_needed_is_race_safe() -> None:
    """Concurrent refresh_if_needed calls should only trigger one HTTP request."""
    manager = OAuthTokenManager(refresh_token="rt-abc", client_id="client-1")
    # Expired token.
    manager._expires_at = time.time() - 1  # pyright: ignore[reportPrivateUsage]

    call_count = 0

    async def fake_refresh() -> None:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)
        manager._access_token = "token"  # pyright: ignore[reportPrivateUsage]
        manager._expires_at = time.time() + 3600  # pyright: ignore[reportPrivateUsage]

    with patch.object(manager, "refresh", side_effect=fake_refresh):
        await asyncio.gather(
            manager.refresh_if_needed(),
            manager.refresh_if_needed(),
            manager.refresh_if_needed(),
        )

    assert call_count == 1


@pytest.mark.asyncio
async def test_run_refresh_loop_calls_refresh_if_needed() -> None:
    manager = OAuthTokenManager(refresh_token="rt-abc", client_id="client-1")
    call_count = 0

    async def fake_refresh_if_needed() -> None:
        nonlocal call_count
        call_count += 1

    with patch.object(manager, "refresh_if_needed", side_effect=fake_refresh_if_needed):
        with patch("src.llm.oauth.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Let the loop run for 2 iterations then cancel.
            async def stop_after_two(*_: Any) -> None:
                if call_count >= 2:
                    raise asyncio.CancelledError

            mock_sleep.side_effect = stop_after_two

            with pytest.raises(asyncio.CancelledError):
                await manager.run_refresh_loop()

    assert call_count >= 2


# ---------------------------------------------------------------------------
# OAuthOpenAI
# ---------------------------------------------------------------------------


def test_oauth_openai_auth_headers_reflects_current_token() -> None:
    manager = OAuthTokenManager(refresh_token="rt-abc", client_id="client-1")
    manager._access_token = "initial-token"  # pyright: ignore[reportPrivateUsage]

    openai_client = OAuthOpenAI(token_manager=manager, base_url="http://localhost/v1")

    assert openai_client.auth_headers == {"Authorization": "Bearer initial-token"}

    # Simulate a token refresh.
    manager._access_token = "refreshed-token"  # pyright: ignore[reportPrivateUsage]

    assert openai_client.auth_headers == {"Authorization": "Bearer refreshed-token"}


def test_oauth_openai_does_not_require_api_key_env_var() -> None:
    """OAuthOpenAI must construct without a real api_key."""
    manager = OAuthTokenManager(refresh_token="rt-abc", client_id="client-1")
    manager._access_token = "tok"  # pyright: ignore[reportPrivateUsage]

    client = OAuthOpenAI(token_manager=manager, base_url="https://api.openai.com/v1")
    assert client is not None
