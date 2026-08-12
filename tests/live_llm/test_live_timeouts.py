from __future__ import annotations

import time
from typing import Any

import anthropic
import httpx
import openai
import pytest

from src.llm.request_builder import execute_completion

from .conftest import make_backend, require_provider_key, wrap_async_method
from .model_matrix import LiveModelSpec, ProviderName, get_live_model_specs

pytestmark = [pytest.mark.live_llm]

GENEROUS_TIMEOUT_SECONDS = 120
TIGHT_TIMEOUT_SECONDS = 0.01
# Well under the 600s client default; generous enough to absorb SDK retries.
TIGHT_TIMEOUT_WALL_CLOCK_LIMIT_SECONDS = 30

TIMEOUT_EXCEPTIONS: dict[ProviderName, tuple[type[BaseException], ...]] = {
    "anthropic": (anthropic.APITimeoutError,),
    "openai": (openai.APITimeoutError,),
    # google-genai raises httpx or aiohttp timeouts depending on its transport;
    # aiohttp surfaces as asyncio.TimeoutError (== builtins.TimeoutError).
    "gemini": (httpx.TimeoutException, TimeoutError),
}

PROVIDER_MARKS = {
    "anthropic": pytest.mark.requires_anthropic,
    "openai": pytest.mark.requires_openai,
    "gemini": pytest.mark.requires_gemini,
}


def representative_specs() -> list[Any]:
    """One spec per provider — timeout plumbing is transport-level, not model-level."""
    params: list[Any] = []
    for provider in ("anthropic", "openai", "gemini"):
        specs = get_live_model_specs(provider=provider)
        if not specs:
            continue
        params.append(
            pytest.param(specs[0], marks=PROVIDER_MARKS[provider], id=specs[0].id)
        )
    return params


def assert_timeout_reached_sdk(
    model_spec: LiveModelSpec, call_kwargs: dict[str, Any], timeout_seconds: float
) -> None:
    if model_spec.provider == "gemini":
        http_options = call_kwargs["config"]["http_options"]
        assert http_options.timeout == int(timeout_seconds * 1000)
    else:
        assert call_kwargs["timeout"] == timeout_seconds


def sdk_call_target(backend: Any, model_spec: LiveModelSpec) -> tuple[Any, str]:
    if model_spec.provider == "gemini":
        return backend._client.aio.models, "generate_content"
    if model_spec.provider == "anthropic":
        return backend._client.messages, "create"
    return backend._client.chat.completions, "create"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", representative_specs())
async def test_live_provider_timeout_reaches_the_wire(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(
        model_spec, provider_params={"timeout": GENEROUS_TIMEOUT_SECONDS}
    )
    target, attribute = sdk_call_target(backend, model_spec)
    calls = wrap_async_method(monkeypatch, target, attribute)

    result = await execute_completion(
        backend,
        config,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=256,
    )

    assert isinstance(result.content, str)
    assert result.content.strip()
    assert len(calls) == 1
    assert_timeout_reached_sdk(model_spec, calls[0]["kwargs"], GENEROUS_TIMEOUT_SECONDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", representative_specs())
async def test_live_tight_provider_timeout_aborts_request(
    model_spec: LiveModelSpec,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(
        model_spec, provider_params={"timeout": TIGHT_TIMEOUT_SECONDS}
    )

    started = time.monotonic()
    with pytest.raises(TIMEOUT_EXCEPTIONS[model_spec.provider]):
        await execute_completion(
            backend,
            config,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=256,
        )
    elapsed = time.monotonic() - started

    assert (
        elapsed < TIGHT_TIMEOUT_WALL_CLOCK_LIMIT_SECONDS
    ), f"tight timeout took {elapsed:.1f}s — per-request timeout likely not applied"
