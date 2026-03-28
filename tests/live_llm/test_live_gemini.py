from __future__ import annotations

import pytest

from src.llm.caching import PromptCachePolicy
from src.llm.history_adapters import GeminiHistoryAdapter
from src.llm.request_builder import execute_completion

from .conftest import (
    StructuredLiveResponse,
    execute_local_tool,
    favorite_prime_tools,
    make_backend,
    make_large_system_prompt,
    require_provider_key,
    wrap_async_method,
)
from .model_matrix import get_live_model_specs

pytestmark = [pytest.mark.live_llm, pytest.mark.requires_gemini]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="gemini", feature="structured_output"),
    ids=lambda spec: spec.id,
)
async def test_live_gemini_structured_output_and_explicit_cache_reuse(
    model_spec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec, temperature=0)
    cache_create_calls = wrap_async_method(
        monkeypatch,
        backend._client.aio.caches,
        "create",
    )
    generate_calls = wrap_async_method(
        monkeypatch,
        backend._client.aio.models,
        "generate_content",
    )
    cache_policy = PromptCachePolicy(mode="gemini_cached_content", ttl_seconds=300)

    messages = [
        {
            "role": "system",
            "content": make_large_system_prompt(label=f"gemini-{model_spec.family}"),
        },
        {
            "role": "user",
            "content": (
                "Return valid JSON with provider='gemini', "
                f"family='{model_spec.family}', and answer='cache-ok'. "
                "Return JSON only, with no prose or markdown."
            ),
        },
    ]

    first = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=512,
        response_format=StructuredLiveResponse,
        cache_policy=cache_policy,
    )
    second = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=512,
        response_format=StructuredLiveResponse,
        cache_policy=cache_policy,
    )

    assert isinstance(first.content, StructuredLiveResponse)
    assert first.content.provider == "gemini"
    assert first.content.family == model_spec.family
    assert isinstance(second.content, StructuredLiveResponse)

    assert len(cache_create_calls) == 1
    assert len(generate_calls) == 2
    first_cached_content = generate_calls[0]["kwargs"]["config"]["cached_content"]
    second_cached_content = generate_calls[1]["kwargs"]["config"]["cached_content"]
    assert first_cached_content == second_cached_content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="gemini", feature="thinking"),
    ids=lambda spec: spec.id,
)
async def test_live_gemini_thinking_and_tool_replay(
    model_spec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(
        model_spec,
        thinking_budget_tokens=512,
        temperature=0,
    )
    generate_calls = wrap_async_method(
        monkeypatch,
        backend._client.aio.models,
        "generate_content",
    )
    tools = favorite_prime_tools()
    adapter = GeminiHistoryAdapter()

    initial_messages = [
        {
            "role": "user",
            "content": (
                "Before answering, call the get_favorite_prime tool exactly once. "
                "Do not answer with plain text on this turn. "
                "After the tool result arrives, answer with the exact text "
                "'13 is prime.'"
            ),
        }
    ]

    first = await execute_completion(
        backend,
        config,
        messages=initial_messages,
        max_tokens=512,
        tools=tools,
        tool_choice="required",
    )

    assert generate_calls[0]["kwargs"]["config"]["thinking_config"] == {
        "thinking_budget": 512,
    }
    assert first.tool_calls, "Gemini should issue a tool call in the first turn"
    assert any(
        tool_call.thought_signature for tool_call in first.tool_calls
    ), "Gemini tool replay should preserve thought signatures"

    tool_call = first.tool_calls[0]
    tool_result = execute_local_tool(tool_call.name, tool_call.input)
    replay_messages = initial_messages + [
        adapter.format_assistant_tool_message(first),
        *adapter.format_tool_results(
            [
                {
                    "tool_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "result": tool_result,
                }
            ]
        ),
    ]

    second = await execute_completion(
        backend,
        config,
        messages=replay_messages,
        max_tokens=512,
        tools=tools,
        tool_choice="none",
    )

    assert generate_calls[1]["kwargs"]["config"]["thinking_config"] == {
        "thinking_budget": 512,
    }
    assert isinstance(second.content, str)
    assert "13" in second.content
    assert "prime" in second.content.lower()
