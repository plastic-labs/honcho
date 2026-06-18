import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from src.llm.backends.openai import OpenAIBackend
from src.utils.representation import PromptRepresentation


def _await_kwargs(mock_method: Any) -> dict[str, Any]:
    await_args = mock_method.await_args
    if await_args is None:
        raise AssertionError("Expected the mocked method to have been awaited")
    return await_args.kwargs


async def _empty_stream() -> AsyncIterator[Any]:
    chunks: list[Any] = []  # async generator that yields nothing
    for chunk in chunks:
        yield chunk


class _StructuredResponse(BaseModel):
    answer: str


def _structured_create_return(content: str, parsed: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=content,
                    parsed=parsed,
                    tool_calls=[],
                    reasoning_details=[],
                    refusal=None,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details=None,
        ),
    )


@pytest.mark.asyncio
async def test_openai_backend_uses_gpt5_params_and_extracts_reasoning() -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="Hello from GPT-5",
                        tool_calls=[],
                        reasoning_details=[
                            SimpleNamespace(
                                content="reasoning summary",
                                model_dump=lambda: {
                                    "type": "reasoning",
                                    "content": "reasoning summary",
                                },
                            )
                        ],
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            ),
        )
    )

    backend = OpenAIBackend(client)
    result = await backend.complete(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        thinking_effort="high",
    )

    assert result.content == "Hello from GPT-5"
    assert result.thinking_content == "reasoning summary"
    assert result.reasoning_details == [
        {"type": "reasoning", "content": "reasoning summary"}
    ]
    assert result.cache_read_input_tokens == 4

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert call["model"] == "gpt-5-mini"
    assert call["max_completion_tokens"] == 100
    assert call["reasoning_effort"] == "high"
    assert "max_tokens" not in call


@pytest.mark.asyncio
async def test_openai_backend_passes_thinking_effort_through_for_non_gpt5_models() -> (
    None
):
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="Hello from GPT-4.1",
                        tool_calls=[],
                        reasoning_details=[],
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=None,
            ),
        )
    )

    backend = OpenAIBackend(client)
    await backend.complete(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        thinking_effort="low",
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert call["model"] == "gpt-4.1"
    assert call["max_tokens"] == 100
    assert call["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_openai_backend_does_not_treat_proxy_models_with_gpt5_substring_as_gpt5() -> (
    None
):
    """Regression: proxy/deployment names containing 'gpt-5' must use `max_tokens`.

    Flexible OpenAI-compatible configuration means operators commonly route through
    proxies/Azure deployments with IDs like `azure-gpt-5-deployment` or
    `my-gpt-5-proxy`. A naive substring check would incorrectly send
    `max_completion_tokens` (a GPT-5-only parameter) to those endpoints.
    """
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=[],
                        reasoning_details=[],
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=None,
            ),
        )
    )

    backend = OpenAIBackend(client)
    await backend.complete(
        model="my-gpt-5-proxy",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert call["max_tokens"] == 100
    assert "max_completion_tokens" not in call


@pytest.mark.asyncio
async def test_openai_backend_passes_thinking_budget_via_extra_body() -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=[],
                        reasoning_details=[],
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=None,
            ),
        )
    )

    backend = OpenAIBackend(client)
    await backend.complete(
        model="x-ai/grok-4.1-fast",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        thinking_budget_tokens=256,
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert call["extra_body"] == {"reasoning": {"max_tokens": 256}}


@pytest.mark.asyncio
async def test_openai_backend_skips_extra_body_when_thinking_budget_zero() -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=[],
                        reasoning_details=[],
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=None,
            ),
        )
    )

    backend = OpenAIBackend(client)
    await backend.complete(
        model="x-ai/grok-4.1-fast",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        thinking_budget_tokens=0,
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert "extra_body" not in call


@pytest.mark.asyncio
async def test_openai_backend_converts_anthropic_style_tools() -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="Used tools",
                        tool_calls=[],
                        reasoning_details=[],
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=None,
            ),
        )
    )

    backend = OpenAIBackend(client)
    await backend.complete(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[
            {
                "name": "get_weather",
                "description": "Lookup weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        tool_choice="required",
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Lookup weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    assert call["tool_choice"] == "required"


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "gpt-5-turbo",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5-preview",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-preview",
    ],
)
def test_openai_reasoning_models_use_max_completion_tokens(model: str) -> None:
    """Reasoning model families (gpt-5 incl. x.y versions, o1/o3/o4) must send
    max_completion_tokens, not max_tokens — OpenAI rejects max_tokens for them
    with 400 unsupported_parameter."""
    from src.llm.backends.openai import (
        _uses_max_completion_tokens,  # pyright: ignore[reportPrivateUsage]
    )

    assert _uses_max_completion_tokens(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "some-proxy-model",
    ],
)
def test_openai_classic_models_use_max_tokens(model: str) -> None:
    """Non-reasoning OpenAI and OpenAI-compatible proxy models stay on
    the classic max_tokens parameter."""
    from src.llm.backends.openai import (
        _uses_max_completion_tokens,  # pyright: ignore[reportPrivateUsage]
    )

    assert _uses_max_completion_tokens(model) is False


@pytest.mark.asyncio
async def test_structured_output_default_mode_uses_parse() -> None:
    """Default json_schema mode uses parse(), not the json_object path."""
    client = Mock()
    client.chat.completions.parse = AsyncMock(
        return_value=_structured_create_return(
            '{"answer": "ok"}', parsed=_StructuredResponse(answer="ok")
        )
    )
    client.chat.completions.create = AsyncMock()

    backend = OpenAIBackend(client)
    result = await backend.complete(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=_StructuredResponse,
    )

    assert client.chat.completions.parse.await_count == 1
    assert client.chat.completions.create.await_count == 0
    parse_call = _await_kwargs(client.chat.completions.parse)
    assert parse_call["response_format"] is _StructuredResponse
    assert isinstance(result.content, _StructuredResponse)


@pytest.mark.asyncio
async def test_structured_output_parse_failure_returns_empty_without_second_request() -> (
    None
):
    """parse() failure returns an empty representation, no second request (#797)."""
    client = Mock()
    client.chat.completions.parse = AsyncMock(
        side_effect=json.JSONDecodeError("Expecting value", "not json", 0)
    )
    client.chat.completions.create = AsyncMock()

    backend = OpenAIBackend(client)
    result = await backend.complete(
        model="glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=PromptRepresentation,
    )

    assert client.chat.completions.parse.await_count == 1
    assert client.chat.completions.create.await_count == 0  # no second request
    assert isinstance(result.content, PromptRepresentation)
    assert result.content.explicit == []


@pytest.mark.asyncio
async def test_structured_output_json_object_mode_request_shape() -> None:
    """json_object mode skips parse(), requests json_object, injects the schema."""
    client = Mock()
    client.chat.completions.parse = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_structured_create_return('{"answer": "ok"}')
    )

    backend = OpenAIBackend(client)
    result = await backend.complete(
        model="glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=_StructuredResponse,
        extra_params={"structured_output_mode": "json_object"},
    )

    assert client.chat.completions.parse.await_count == 0
    assert client.chat.completions.create.await_count == 1
    call = _await_kwargs(client.chat.completions.create)
    assert call["response_format"] == {"type": "json_object"}

    system_messages = [m for m in call["messages"] if m["role"] == "system"]
    assert system_messages, "expected a system message carrying the schema"
    system_content = system_messages[0]["content"]
    assert "JSON" in system_content
    assert "answer" in system_content  # schema property serialized in
    assert isinstance(result.content, _StructuredResponse)
    assert result.content.answer == "ok"


@pytest.mark.asyncio
async def test_structured_output_json_object_mode_repairs_markdown() -> None:
    """A provider that ignores json_object and returns prose must not crash —
    PromptRepresentation repairs to an empty representation, not an exception."""
    client = Mock()
    client.chat.completions.parse = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_structured_create_return(
            "Sure! Here are the facts:\n- the user likes coffee"
        )
    )

    backend = OpenAIBackend(client)
    result = await backend.complete(
        model="glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=PromptRepresentation,
        extra_params={"structured_output_mode": "json_object"},
    )

    assert isinstance(result.content, PromptRepresentation)


@pytest.mark.asyncio
async def test_structured_output_json_object_mode_does_not_mutate_messages() -> None:
    """The schema-injection helper must copy, never mutate the caller's list."""
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=_structured_create_return('{"answer": "ok"}')
    )

    backend = OpenAIBackend(client)
    original_messages = [{"role": "user", "content": "Hello"}]
    await backend.complete(
        model="glm-4.6",
        messages=original_messages,
        max_tokens=100,
        response_format=_StructuredResponse,
        extra_params={"structured_output_mode": "json_object"},
    )

    assert original_messages == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_stream_structured_output_default_mode_uses_json_schema() -> None:
    """Streaming in default mode converts the model to a json_schema dict."""
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=_empty_stream())

    backend = OpenAIBackend(client)
    async for _ in backend.stream(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=_StructuredResponse,
    ):
        pass

    call = _await_kwargs(client.chat.completions.create)
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == "_StructuredResponse"


@pytest.mark.asyncio
async def test_stream_structured_output_json_object_mode() -> None:
    """Streaming in json_object mode requests json_object + injects the schema."""
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=_empty_stream())

    backend = OpenAIBackend(client)
    async for _ in backend.stream(
        model="glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=_StructuredResponse,
        extra_params={"structured_output_mode": "json_object"},
    ):
        pass

    call = _await_kwargs(client.chat.completions.create)
    assert call["response_format"] == {"type": "json_object"}
    system_messages = [m for m in call["messages"] if m["role"] == "system"]
    assert system_messages and "JSON" in system_messages[0]["content"]
