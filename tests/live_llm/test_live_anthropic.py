from __future__ import annotations

import pytest

from src.llm.history_adapters import AnthropicHistoryAdapter
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

pytestmark = [pytest.mark.live_llm, pytest.mark.requires_anthropic]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="anthropic"),
    ids=lambda spec: spec.id,
)
async def test_live_anthropic_structured_output_and_prefix_caching(
    model_spec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec)
    create_calls = wrap_async_method(monkeypatch, backend._client.messages, "create")

    messages = [
        {
            "role": "system",
            "content": make_large_system_prompt(label=f"anthropic-{model_spec.family}"),
        },
        {
            "role": "user",
            "content": (
                "Return valid JSON with provider='anthropic', "
                f"family='{model_spec.family}', and answer='cache-ok'."
            ),
        },
    ]

    results = []
    for _ in range(3):
        results.append(
            await execute_completion(
                backend,
                config,
                messages=messages,
                max_tokens=256,
                response_format=StructuredLiveResponse,
            )
        )
        if len(results) >= 2 and results[-1].cache_read_input_tokens > 0:
            break

    first = results[0]
    later_results = results[1:]

    assert isinstance(first.content, StructuredLiveResponse)
    assert first.content.provider == "anthropic"
    assert first.content.family == model_spec.family
    assert later_results, "Anthropic caching validation requires at least two calls"
    for result in later_results:
        assert isinstance(result.content, StructuredLiveResponse)
    assert any(
        result.cache_read_input_tokens > 0 for result in later_results
    ), "Anthropic prompt caching did not report a cache hit after repeated identical requests"

    assert len(create_calls) == len(results)
    for call in create_calls:
        assert call["kwargs"]["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="anthropic"),
    ids=lambda spec: spec.id,
)
async def test_live_anthropic_thinking_and_tool_replay(
    model_spec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec, thinking_budget_tokens=1024)
    create_calls = wrap_async_method(monkeypatch, backend._client.messages, "create")
    tools = favorite_prime_tools()
    adapter = AnthropicHistoryAdapter()

    initial_messages = [
        {
            "role": "user",
            "content": (
                "Before answering, call the get_favorite_prime tool exactly once. "
                "After you receive the tool result, answer in one sentence that includes "
                "the number and the word 'prime'."
            ),
        }
    ]

    first = await execute_completion(
        backend,
        config,
        messages=initial_messages,
        max_tokens=2048,
        tools=tools,
    )

    assert create_calls[0]["kwargs"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert first.tool_calls, "Anthropic should issue a tool call in the first turn"
    assert first.thinking_blocks, "Anthropic thinking blocks should be preserved"

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
        max_tokens=2048,
        tools=tools,
    )

    assert create_calls[1]["kwargs"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert isinstance(second.content, str)
    assert "13" in second.content
    assert "prime" in second.content.lower()
