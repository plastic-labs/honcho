from __future__ import annotations

import pytest

from src.llm.backend import CompletionResult
from src.llm.capabilities import anthropic_capabilities
from src.llm.request_builder import execute_completion

from .conftest import (
    REASONING_TOOL_PROMPT,
    StructuredLiveResponse,
    execute_local_tool,
    favorite_prime_tools,
    make_backend,
    make_large_system_prompt,
    replay_tool_round,
    require_provider_key,
    wrap_async_method,
)
from .model_matrix import LiveModelSpec, get_live_model_specs

pytestmark = [pytest.mark.live_llm, pytest.mark.requires_anthropic]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="anthropic"),
    ids=lambda spec: spec.id,
)
async def test_live_anthropic_structured_output_and_prefix_caching(
    model_spec: LiveModelSpec,
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

    results: list[CompletionResult] = []
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
    assert any(result.cache_read_input_tokens > 0 for result in later_results), (
        "Anthropic prompt caching did not report a cache hit after repeated identical requests"
    )

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
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec, thinking_budget_tokens=1024)
    create_calls = wrap_async_method(monkeypatch, backend._client.messages, "create")
    tools = favorite_prime_tools()
    # A budget model gets the budget; an adaptive model rejects one, so the
    # budget only switches adaptive thinking on.
    expected_thinking = (
        {"type": "enabled", "budget_tokens": 1024}
        if anthropic_capabilities(model_spec.model).thinking == "budget"
        else {"type": "adaptive"}
    )

    initial_messages = [{"role": "user", "content": REASONING_TOOL_PROMPT}]

    # Adaptive models decide per turn whether to think, so give the first turn
    # a few tries to produce a thinking block worth replaying.
    first = None
    for _ in range(3):
        first = await execute_completion(
            backend,
            config,
            messages=initial_messages,
            max_tokens=2048,
            tools=tools,
        )
        if first.thinking_blocks and first.tool_calls:
            break
    assert first is not None

    assert create_calls[0]["kwargs"]["thinking"] == expected_thinking
    assert first.tool_calls, "Anthropic should issue a tool call in the first turn"
    assert first.thinking_blocks, "Anthropic thinking blocks should be preserved"

    tool_call = first.tool_calls[0]
    tool_result = execute_local_tool(tool_call.name, tool_call.input)
    replay_messages = replay_tool_round(config, initial_messages, first, tool_result)

    second = await execute_completion(
        backend,
        config,
        messages=replay_messages,
        max_tokens=2048,
        tools=tools,
    )

    assert create_calls[-1]["kwargs"]["thinking"] == expected_thinking
    assert isinstance(second.content, str)
    assert "13" in second.content
    assert "prime" in second.content.lower()
