"""Decode gateway-shaped HTTP usage with the real SDK and full Honcho call."""

from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from tenacity import wait_none

from src.config import ModelConfig
from src.llm import api
from src.llm.registry import CLIENTS


def _no_retry_wait(**_kwargs: Any) -> wait_none:
    return wait_none()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "expected_input", "expected_output"),
    [
        pytest.param(
            {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            12,
            7,
            id="full-usage",
        ),
        pytest.param({"total_tokens": 19}, 0, 0, id="both-counts-omitted"),
        pytest.param(
            {"completion_tokens": 7, "total_tokens": 19},
            0,
            7,
            id="prompt-count-omitted",
        ),
        pytest.param(
            {"prompt_tokens": 12, "total_tokens": 19},
            12,
            0,
            id="completion-count-omitted",
        ),
        pytest.param(None, 0, 0, id="usage-absent"),
    ],
)
async def test_omitted_usage_survives_real_sdk_and_full_call_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    usage: dict[str, int] | None,
    expected_input: int,
    expected_output: int,
) -> None:
    provider_requests = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal provider_requests
        provider_requests += 1
        assert request.url.path == "/v1/chat/completions"
        payload: dict[str, Any] = {
            "id": "chatcmpl-usage-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello"},
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return httpx.Response(200, json=payload)

    # Preserve the real three-attempt retry path, but do not sleep on a regression.
    monkeypatch.setattr(api, "wait_exponential", _no_retry_wait)
    async with AsyncOpenAI(
        api_key="non-secret-test-key",
        base_url="http://127.0.0.1:1460/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as client:
        monkeypatch.setitem(CLIENTS, "openai", client)
        response = await api.honcho_llm_call(
            model_config=ModelConfig(model="gpt-4.1-nano", transport="openai"),
            prompt="Hello",
            max_tokens=16,
            retry_attempts=3,
        )

    assert response.content == "Hello"
    assert response.input_tokens == expected_input
    assert response.output_tokens == expected_output
    assert provider_requests == 1
