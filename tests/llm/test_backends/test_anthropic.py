from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from pydantic import BaseModel

from src.exceptions import ValidationException
from src.llm.backends.anthropic import AnthropicBackend


@pytest.mark.asyncio
async def test_anthropic_backend_extracts_text_thinking_and_tool_calls() -> None:
    client = Mock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[
                ThinkingBlock(
                    type="thinking",
                    thinking="internal reasoning",
                    signature="sig_123",
                ),
                TextBlock(type="text", text="Hello from Anthropic"),
                ToolUseBlock(
                    type="tool_use",
                    id="tool_1",
                    name="search",
                    input={"query": "honcho"},
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=2,
            ),
            stop_reason="tool_use",
        )
    )

    backend = AnthropicBackend(client)
    result = await backend.complete(
        model="claude-haiku-4-5",
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=100,
        tools=[
            {
                "name": "search",
                "description": "Search for information",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ],
        thinking_budget_tokens=2048,
        tool_choice="required",
    )

    assert result.content == "Hello from Anthropic"
    assert result.thinking_content == "internal reasoning"
    assert result.thinking_blocks == [
        {
            "type": "thinking",
            "thinking": "internal reasoning",
            "signature": "sig_123",
        }
    ]
    assert result.tool_calls[0].name == "search"
    assert result.input_tokens == 15
    assert result.output_tokens == 5
    assert result.finish_reason == "tool_use"

    await_args = client.messages.create.await_args
    if await_args is None:
        raise AssertionError("Expected Anthropic client call")
    call = await_args.kwargs
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"][0]["text"] == "System prompt"
    assert call["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert call["tool_choice"] == {"type": "any"}


class StructuredResponse(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_anthropic_backend_skips_assistant_prefill_for_claude_4_models() -> None:
    client = Mock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[TextBlock(type="text", text='{"answer":"ok"}')],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            stop_reason="end_turn",
        )
    )

    backend = AnthropicBackend(client)
    result = await backend.complete(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
    )

    assert isinstance(result.content, StructuredResponse)
    assert result.content.answer == "ok"
    await_args = client.messages.create.await_args
    if await_args is None:
        raise AssertionError("Expected Anthropic client call")
    call = await_args.kwargs
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"
    assert call["messages"][0]["content"].startswith("Hello\n\nRespond with valid JSON")


@pytest.mark.asyncio
async def test_anthropic_backend_ignores_thinking_effort() -> None:
    client = Mock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[TextBlock(type="text", text="ok")],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            stop_reason="end_turn",
        )
    )

    backend = AnthropicBackend(client)
    await backend.complete(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        thinking_effort="high",
    )

    await_args = client.messages.create.await_args
    if await_args is None:
        raise AssertionError("Expected Anthropic client call")
    call = await_args.kwargs
    assert "thinking" not in call
    assert "reasoning_effort" not in call


def _text_response_client() -> Mock:
    """Mock Anthropic client returning a minimal text completion."""
    client = Mock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[TextBlock(type="text", text="ok")],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            stop_reason="end_turn",
        )
    )
    return client


def _create_call_kwargs(client: Mock) -> dict[str, Any]:
    await_args = client.messages.create.await_args
    if await_args is None:
        raise AssertionError("Expected Anthropic client call")
    return await_args.kwargs


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-5-0",  # future Opus keeps using adaptive thinking
        "claude-opus-4-8-20260120",  # date-suffixed variant
    ],
)
@pytest.mark.asyncio
async def test_anthropic_backend_uses_adaptive_thinking(model: str) -> None:
    # These models reject the legacy {"type": "enabled", "budget_tokens": N}
    # shape with HTTP 400; the backend must send adaptive thinking instead.
    client = _text_response_client()
    backend = AnthropicBackend(client)
    await backend.complete(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32000,
        thinking_budget_tokens=16000,
    )

    call = _create_call_kwargs(client)
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ],
)
@pytest.mark.asyncio
async def test_anthropic_backend_keeps_legacy_thinking(model: str) -> None:
    # Models that still accept the legacy budget format must be unchanged.
    client = _text_response_client()
    backend = AnthropicBackend(client)
    await backend.complete(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32000,
        thinking_budget_tokens=16000,
    )

    call = _create_call_kwargs(client)
    assert call["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert "output_config" not in call


@pytest.mark.asyncio
async def test_anthropic_backend_adaptive_thinking_respects_explicit_effort() -> None:
    # An explicit thinking_effort overrides the budget-derived effort bucket.
    client = _text_response_client()
    backend = AnthropicBackend(client)
    await backend.complete(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32000,
        thinking_budget_tokens=16000,
        thinking_effort="low",
    )

    call = _create_call_kwargs(client)
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "low"}


class _FakeAnthropicStream:
    """Minimal async-context-manager stand-in for client.messages.stream()."""

    def __init__(self, final_message: Any) -> None:
        self._final_message: Any = final_message

    async def __aenter__(self) -> "_FakeAnthropicStream":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        del exc_info
        return False

    def __aiter__(self) -> "_FakeAnthropicStream":
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        return self._final_message


@pytest.mark.asyncio
async def test_anthropic_backend_stream_uses_adaptive_thinking_for_opus_4_8() -> None:
    # The streaming path must apply the same adaptive thinking format.
    final_message = SimpleNamespace(
        usage=SimpleNamespace(output_tokens=5),
        stop_reason="end_turn",
    )
    client = Mock()
    client.messages.stream = Mock(return_value=_FakeAnthropicStream(final_message))

    backend = AnthropicBackend(client)
    chunks = [
        chunk
        async for chunk in backend.stream(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=32000,
            thinking_budget_tokens=16000,
        )
    ]

    assert chunks[-1].is_done is True
    stream_call = client.messages.stream.call_args
    if stream_call is None:
        raise AssertionError("Expected Anthropic stream call")
    call = stream_call.kwargs
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}


@pytest.mark.parametrize(
    ("thinking_budget_tokens", "expected_effort"),
    [
        (2048, "low"),
        (8000, "medium"),
        (16000, "high"),
        (32000, "xhigh"),
    ],
)
@pytest.mark.asyncio
async def test_anthropic_backend_maps_budget_to_effort(
    thinking_budget_tokens: int, expected_effort: str
) -> None:
    # With no explicit effort, the legacy budget is bucketed into an effort
    # level so existing budget-based configs keep a comparable thinking depth.
    client = _text_response_client()
    backend = AnthropicBackend(client)
    await backend.complete(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64000,
        thinking_budget_tokens=thinking_budget_tokens,
    )

    call = _create_call_kwargs(client)
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": expected_effort}


@pytest.mark.asyncio
async def test_anthropic_backend_rejects_unknown_thinking_effort() -> None:
    # An unrecognized effort is a caller error, not a silent fallback.
    backend = AnthropicBackend(_text_response_client())
    with pytest.raises(ValidationException):
        await backend.complete(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=32000,
            thinking_budget_tokens=16000,
            thinking_effort="turbo",
        )


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-opus-4-6"])
@pytest.mark.asyncio
async def test_anthropic_backend_rejects_negative_budget(model: str) -> None:
    # Negative budgets must not be bucketed (adaptive) or forwarded (legacy).
    backend = AnthropicBackend(_text_response_client())
    with pytest.raises(ValidationException):
        await backend.complete(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=32000,
            thinking_budget_tokens=-1,
        )


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-8", "claude-opus-4-6", "claude-haiku-4-5"],
)
@pytest.mark.parametrize("thinking_budget_tokens", [None, 0])
@pytest.mark.asyncio
async def test_anthropic_backend_no_thinking_for_zero_or_none_budget(
    model: str, thinking_budget_tokens: int | None
) -> None:
    # 0 is a valid "disable thinking" sentinel (config permits it); neither 0
    # nor None raises, and no thinking params are sent.
    client = _text_response_client()
    backend = AnthropicBackend(client)
    await backend.complete(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32000,
        thinking_budget_tokens=thinking_budget_tokens,
    )

    call = _create_call_kwargs(client)
    assert "thinking" not in call
    assert "output_config" not in call
