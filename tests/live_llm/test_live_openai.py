from __future__ import annotations

import pytest

from src.llm.request_builder import execute_completion

from .conftest import (
    StructuredLiveResponse,
    make_backend,
    make_large_system_prompt,
    require_provider_key,
    wrap_async_method,
)
from .model_matrix import LiveModelSpec, get_live_model_specs

pytestmark = [pytest.mark.live_llm, pytest.mark.requires_openai]

_GPT4_SPECS = tuple(
    spec
    for spec in get_live_model_specs(provider="openai")
    if spec.family == "gpt_4_class"
)
_GPT5_SPECS = tuple(
    spec
    for spec in get_live_model_specs(provider="openai")
    if spec.family == "gpt_5_class"
)
_JSON_OBJECT_SPECS = tuple(
    spec
    for spec in get_live_model_specs(provider="openai")
    if spec.family == "openai_json_object"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", _GPT4_SPECS, ids=lambda spec: spec.id)
async def test_live_openai_gpt4_structured_output_and_prefix_caching(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec)
    parse_calls = wrap_async_method(
        monkeypatch,
        backend._client.chat.completions,
        "parse",
    )

    messages = [
        {
            "role": "system",
            "content": make_large_system_prompt(label=f"openai-{model_spec.family}"),
        },
        {
            "role": "user",
            "content": (
                "Return valid JSON with provider='openai', "
                f"family='{model_spec.family}', and answer='cache-ok'."
            ),
        },
    ]

    first = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=256,
        response_format=StructuredLiveResponse,
    )
    second = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=256,
        response_format=StructuredLiveResponse,
    )

    assert isinstance(first.content, StructuredLiveResponse)
    assert first.content.provider == "openai"
    assert first.content.family == model_spec.family
    assert isinstance(second.content, StructuredLiveResponse)
    assert second.cache_read_input_tokens > 0

    assert parse_calls[0]["kwargs"]["response_format"] is StructuredLiveResponse
    assert "max_tokens" in parse_calls[0]["kwargs"]
    assert "max_completion_tokens" not in parse_calls[0]["kwargs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", _GPT5_SPECS, ids=lambda spec: spec.id)
async def test_live_openai_gpt5_reasoning_structured_output_and_prefix_caching(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec, reasoning_effort="minimal")
    parse_calls = wrap_async_method(
        monkeypatch,
        backend._client.chat.completions,
        "parse",
    )

    messages = [
        {
            "role": "system",
            "content": make_large_system_prompt(label=f"openai-{model_spec.family}"),
        },
        {
            "role": "user",
            "content": (
                "Return valid JSON with provider='openai', "
                f"family='{model_spec.family}', and answer='reasoning-ok'."
            ),
        },
    ]

    first = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=1024,
        response_format=StructuredLiveResponse,
    )
    second = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=1024,
        response_format=StructuredLiveResponse,
    )

    assert isinstance(first.content, StructuredLiveResponse)
    assert first.content.provider == "openai"
    assert first.content.family == model_spec.family
    assert isinstance(second.content, StructuredLiveResponse)
    assert second.cache_read_input_tokens > 0

    assert parse_calls[0]["kwargs"]["response_format"] is StructuredLiveResponse
    assert parse_calls[0]["kwargs"]["reasoning_effort"] == "minimal"
    assert "max_completion_tokens" in parse_calls[0]["kwargs"]
    assert "max_tokens" not in parse_calls[0]["kwargs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", _JSON_OBJECT_SPECS, ids=lambda spec: spec.id)
async def test_live_openai_json_object_structured_output(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For OpenAI-compatible providers without json_schema support, json_object
    mode must skip parse(), request {"type": "json_object"}, and still produce a
    valid structured object (the #797 fix, proven against a real provider).

    Configure: LLM_OPENAI_BASE_URL + LLM_OPENAI_API_KEY pointed at the target
    provider, and LIVE_LLM_OPENAI_JSON_OBJECT_MODELS=<model>.
    """
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec, structured_output_mode="json_object")
    parse_calls = wrap_async_method(
        monkeypatch, backend._client.chat.completions, "parse"
    )
    create_calls = wrap_async_method(
        monkeypatch, backend._client.chat.completions, "create"
    )

    messages = [
        {
            "role": "system",
            "content": "You answer questions about a test run.",
        },
        {
            "role": "user",
            "content": (
                "Return provider='openai', "
                f"family='{model_spec.family}', and answer='json-object-ok'."
            ),
        },
    ]

    result = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=512,
        response_format=StructuredLiveResponse,
    )

    assert isinstance(result.content, StructuredLiveResponse)
    assert result.content.provider == "openai"
    assert parse_calls == []
    assert create_calls, "expected a chat.completions.create call"
    assert create_calls[0]["kwargs"]["response_format"] == {"type": "json_object"}
