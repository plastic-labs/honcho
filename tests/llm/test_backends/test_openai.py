from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.exceptions import ValidationException
from src.llm.backends.openai import OpenAIBackend


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
async def test_openai_backend_omits_stop_for_gpt5_reasoning_models() -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="Hello from GPT-5",
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
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        stop=["\n\n\n\n"],
        thinking_effort="low",
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert call["model"] == "gpt-5.4-mini"
    assert call["max_completion_tokens"] == 100
    assert call["reasoning_effort"] == "low"
    assert "stop" not in call


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
async def test_openai_backend_rejects_thinking_budget_tokens() -> None:
    backend = OpenAIBackend(Mock())

    with pytest.raises(
        ValidationException, match="does not support thinking_budget_tokens"
    ):
        await backend.complete(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            thinking_budget_tokens=256,
        )


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
