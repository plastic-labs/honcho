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
async def test_openai_backend_forwards_provider_params_extra_body() -> None:
    """Operator-supplied extra_body in provider_params reaches the OpenAI SDK call.

    This is the escape hatch for OpenAI-compatible proxies that translate to
    other providers (litellm → Vertex AI Anthropic) and need provider-native
    body fields (e.g. Anthropic's `thinking`).
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
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        extra_params={
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}}
        },
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert call["extra_body"] == {
        "thinking": {"type": "enabled", "budget_tokens": 4096}
    }


@pytest.mark.asyncio
async def test_openai_backend_forwards_provider_params_extra_headers_and_query() -> (
    None
):
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
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        extra_params={
            "extra_headers": {"X-Proxy-Route": "vertex"},
            "extra_query": {"trace_id": "abc123"},
        },
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    assert call["extra_headers"] == {"X-Proxy-Route": "vertex"}
    assert call["extra_query"] == {"trace_id": "abc123"}


@pytest.mark.asyncio
async def test_openai_backend_operator_extra_body_wins_over_auto_injection() -> None:
    """When the operator supplies extra_body.reasoning, it must replace the
    value that thinking_budget_tokens would otherwise auto-inject (operator-wins
    shallow merge).
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
        model="x-ai/grok-4.1-fast",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        thinking_budget_tokens=256,
        extra_params={
            "extra_body": {"reasoning": {"effort": "high", "max_tokens": 9999}}
        },
    )

    await_args = client.chat.completions.create.await_args
    if await_args is None:
        raise AssertionError("Expected OpenAI create call")
    call = await_args.kwargs
    # Operator's whole `reasoning` dict replaces Honcho's auto-injected one.
    assert call["extra_body"] == {"reasoning": {"effort": "high", "max_tokens": 9999}}


@pytest.mark.asyncio
async def test_openai_backend_rejects_non_mapping_passthrough() -> None:
    """A non-mapping passthrough (operator misconfiguration) raises a clear
    ValidationException instead of an opaque TypeError deep in the transport.
    """
    client = Mock()
    client.chat.completions.create = AsyncMock()

    backend = OpenAIBackend(client)
    with pytest.raises(ValidationException, match=r"provider_params\.extra_headers"):
        await backend.complete(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            extra_params={"extra_headers": [["X-Foo", "bar"]]},
        )

    client.chat.completions.create.assert_not_awaited()


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
