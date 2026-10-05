from __future__ import annotations

import pytest

from src.llm.history_adapters import OpenAIHistoryAdapter
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
_TOOL_REPLAY_SPECS = tuple(
    spec
    for spec in get_live_model_specs(provider="openai")
    if spec.supports_tool_replay
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
    # Only the original gpt-5 generation accepts 'minimal'; gpt-5.1+ replaced
    # it with 'none'. 'low' is valid everywhere else, including future models.
    is_base_gpt5 = model_spec.model == "gpt-5" or model_spec.model.startswith("gpt-5-")
    reasoning_effort = "minimal" if is_base_gpt5 else "low"
    backend, config = make_backend(model_spec, reasoning_effort=reasoning_effort)
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
    assert parse_calls[0]["kwargs"]["reasoning_effort"] == reasoning_effort
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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", _TOOL_REPLAY_SPECS, ids=lambda spec: spec.id)
async def test_live_openai_tool_replay_preserves_null_content(
    model_spec: LiveModelSpec,
) -> None:
    """Tool-call turns with provider content=null must stay null through
    normalize + history replay, and the continuation must still succeed."""
    require_provider_key(model_spec)
    # Leave reasoning_effort unset: gpt-5.4 rejects function tools with any
    # explicit reasoning_effort other than 'none' on /v1/chat/completions.
    backend, config = make_backend(model_spec)
    tools = favorite_prime_tools()
    adapter = OpenAIHistoryAdapter()

    initial_messages = [
        {
            "role": "user",
            "content": (
                "Before answering, call the get_favorite_prime tool exactly once. "
                "Do not answer with plain text on this turn. "
                "After you receive the tool result, answer in one sentence that "
                "includes the number and the word 'prime'."
            ),
        }
    ]

    first = await execute_completion(
        backend,
        config,
        messages=initial_messages,
        max_tokens=1024,
        tools=tools,
        tool_choice="required",
    )

    assert first.tool_calls, "OpenAI should issue a tool call in the first turn"
    raw_message = first.raw_response.choices[0].message
    raw_content = raw_message.content
    if raw_content is None:
        assert first.content is None
    else:
        assert first.content == raw_content

    assistant_message = adapter.format_assistant_tool_message(first)
    assert assistant_message["content"] is (
        first.content if isinstance(first.content, str) else None
    )
    if raw_content is None:
        assert assistant_message["content"] is None

    tool_call = first.tool_calls[0]
    tool_result = execute_local_tool(tool_call.name, tool_call.input)
    replay_messages = initial_messages + [
        assistant_message,
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
        max_tokens=1024,
        tools=tools,
        tool_choice="auto",
    )

    assert not second.tool_calls, "continuation should answer without another tool call"
    assert isinstance(second.content, str)
    assert "13" in second.content
    assert "prime" in second.content.lower()
