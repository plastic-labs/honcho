from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
import pytest

from src.config import ModelConfig
from src.llm.backends.openai_codex import OpenAICodexBackend, OpenAICodexClient
from src.llm.registry import client_for_model_config


def _jwt(*, exp: int | None = None, account_id: str = "acct_test") -> str:
    header = {"alg": "none", "typ": "JWT"}
    payload: dict[str, Any] = {
        "https://api.openai.com/auth.chatgpt_account_id": account_id,
    }
    if exp is not None:
        payload["exp"] = exp

    def enc(obj: dict[str, Any]) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{enc(header)}.{enc(payload)}.sig"


def _sse_completed(text: str = "ok") -> bytes:
    events = [
        {"type": "response.output_text.delta", "delta": text},
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 2, "output_tokens": 1}},
        },
    ]
    return b"".join(
        f"data: {json.dumps(event)}\n\n".encode() for event in events
    )


@pytest.mark.asyncio
async def test_codex_backend_refreshes_expiring_access_token_before_request() -> None:
    old_token = _jwt(exp=int(time.time()) - 60, account_id="acct_old")
    new_token = _jwt(exp=int(time.time()) + 3600, account_id="acct_new")
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if str(request.url) == "https://auth.openai.com/oauth/token":
            form = request.content.decode()
            assert "grant_type=refresh_token" in form
            assert "refresh_token=refresh-1" in form
            return httpx.Response(
                200,
                json={"access_token": new_token, "refresh_token": "refresh-2"},
            )
        assert request.headers["authorization"] == f"Bearer {new_token}"
        assert request.headers["chatgpt-account-id"] == "acct_new"
        return httpx.Response(200, content=_sse_completed())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICodexClient(
        api_key=old_token,
        refresh_token="refresh-1",
        http_client=http_client,
    )
    backend = OpenAICodexBackend(client)

    result = await backend.complete(model="gpt-5", messages=[], max_tokens=32)

    await http_client.aclose()
    assert result.content == "ok"
    assert client.api_key == new_token
    assert client.refresh_token == "refresh-2"
    assert [request.url.host for request in calls] == ["auth.openai.com", "chatgpt.com"]


@pytest.mark.asyncio
async def test_codex_backend_refreshes_and_retries_once_after_401() -> None:
    old_token = _jwt(exp=int(time.time()) + 3600, account_id="acct_old")
    new_token = _jwt(exp=int(time.time()) + 7200, account_id="acct_new")
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        if str(request.url) == "https://auth.openai.com/oauth/token":
            return httpx.Response(200, json={"access_token": new_token})
        request_count += 1
        if request_count == 1:
            return httpx.Response(401, json={"error": "expired"})
        assert request.headers["authorization"] == f"Bearer {new_token}"
        assert request.headers["chatgpt-account-id"] == "acct_new"
        return httpx.Response(200, content=_sse_completed("recovered"))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICodexClient(
        api_key=old_token,
        refresh_token="refresh-1",
        http_client=http_client,
    )
    backend = OpenAICodexBackend(client)

    result = await backend.complete(model="gpt-5", messages=[], max_tokens=32)

    await http_client.aclose()
    assert result.content == "recovered"
    assert client.api_key == new_token
    assert request_count == 2



def test_registry_uses_per_model_refresh_token_without_default_fast_path() -> None:
    config = ModelConfig(
        model="gpt-5",
        transport="openai_codex",
        api_key=_jwt(exp=int(time.time()) + 3600),
        provider_params={"refresh_token": "per-model-refresh"},
    )

    client = client_for_model_config("openai_codex", config)

    assert isinstance(client, OpenAICodexClient)
    assert client.refresh_token == "per-model-refresh"


def test_registry_refresh_token_env_falls_back_to_global_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config as config_module

    monkeypatch.delenv("MISSING_CODEX_REFRESH_TOKEN", raising=False)
    monkeypatch.setattr(
        config_module.settings.LLM,
        "OPENAI_CODEX_REFRESH_TOKEN",
        "global-refresh",
    )
    config = ModelConfig(
        model="gpt-5",
        transport="openai_codex",
        api_key=_jwt(exp=int(time.time()) + 3600),
        provider_params={"refresh_token_env": "MISSING_CODEX_REFRESH_TOKEN"},
    )

    client = client_for_model_config("openai_codex", config)

    assert isinstance(client, OpenAICodexClient)
    assert client.refresh_token == "global-refresh"


def test_registry_returns_fresh_codex_clients_to_avoid_token_state_leakage() -> None:
    config = ModelConfig(
        model="gpt-5",
        transport="openai_codex",
        api_key=_jwt(exp=int(time.time()) + 3600),
        provider_params={"refresh_token": "refresh"},
    )

    first = client_for_model_config("openai_codex", config)
    second = client_for_model_config("openai_codex", config)

    assert isinstance(first, OpenAICodexClient)
    assert isinstance(second, OpenAICodexClient)
    assert first is not second
