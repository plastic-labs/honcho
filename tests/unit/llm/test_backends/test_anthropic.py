from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from pydantic import BaseModel

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
async def test_anthropic_backend_rejects_thinking_effort() -> None:
    backend = AnthropicBackend(Mock())

    with pytest.raises(ValueError, match="does not support thinking_effort"):
        await backend.complete(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            thinking_effort="high",
        )
