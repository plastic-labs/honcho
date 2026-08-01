from types import SimpleNamespace
from typing import Any
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
async def test_anthropic_backend_forwards_provider_params_passthroughs() -> None:
    """provider_params.extra_body/extra_headers/extra_query reach the Anthropic
    SDK call as kwargs of the same name (the SDK's documented passthrough).
    """
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
        extra_params={
            "extra_body": {"anthropic_beta": ["context-1m-2025-01-15"]},
            "extra_headers": {"X-Proxy-Route": "vertex"},
            "extra_query": {"trace_id": "abc123"},
        },
    )

    await_args = client.messages.create.await_args
    if await_args is None:
        raise AssertionError("Expected Anthropic client call")
    call = await_args.kwargs
    assert call["extra_body"] == {"anthropic_beta": ["context-1m-2025-01-15"]}
    assert call["extra_headers"] == {"X-Proxy-Route": "vertex"}
    assert call["extra_query"] == {"trace_id": "abc123"}


@pytest.mark.asyncio
async def test_anthropic_backend_stream_forwards_provider_params_passthroughs() -> None:
    """The stream() path forwards provider_params passthroughs to the SDK the
    same way complete() does — it has its own merge block, so cover it too.
    """

    class _FakeStream:
        async def __aenter__(self) -> "_FakeStream":
            return self

        async def __aexit__(self, *_: object) -> bool:
            return False

        def __aiter__(self) -> "_FakeStream":
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

        async def get_final_message(self) -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(output_tokens=5),
                stop_reason="end_turn",
            )

    client = Mock()
    client.messages.stream = Mock(return_value=_FakeStream())

    backend = AnthropicBackend(client)
    chunks = [
        chunk
        async for chunk in backend.stream(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            extra_params={
                "extra_body": {"anthropic_beta": ["context-1m-2025-01-15"]},
                "extra_headers": {"X-Proxy-Route": "vertex"},
                "extra_query": {"trace_id": "abc123"},
            },
        )
    ]

    assert chunks  # the terminal is_done chunk is always emitted
    call = client.messages.stream.call_args
    if call is None:
        raise AssertionError("Expected Anthropic stream call")
    kwargs = call.kwargs
    assert kwargs["extra_body"] == {"anthropic_beta": ["context-1m-2025-01-15"]}
    assert kwargs["extra_headers"] == {"X-Proxy-Route": "vertex"}
    assert kwargs["extra_query"] == {"trace_id": "abc123"}


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


def _make_client(content_blocks: list[Any]) -> Mock:
    client = Mock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=content_blocks,
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


SEARCH_TOOL = {
    "name": "search",
    "description": "Search for information",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    },
}


@pytest.mark.asyncio
async def test_anthropic_backend_no_prefill_when_tools_present() -> None:
    """With tools + response_format, the '{' prefill must be skipped and the
    schema instruction must be conditional, so tool_use blocks stay reachable."""
    client = _make_client([TextBlock(type="text", text='{"answer":"ok"}')])

    backend = AnthropicBackend(client)
    result = await backend.complete(
        # claude-3-5 supports prefill, so only the tools guard prevents it here
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[SEARCH_TOOL],
        response_format=StructuredResponse,
    )

    assert isinstance(result.content, StructuredResponse)
    call = client.messages.create.await_args.kwargs
    assert call["messages"][-1]["role"] == "user"  # no assistant '{' prefill
    assert (
        "If not responding with a tool call, respond with valid JSON"
        in call["messages"][0]["content"]
    )


@pytest.mark.asyncio
async def test_anthropic_backend_prefill_unchanged_without_tools() -> None:
    """Tool-less structured calls keep the prefill + unconditional wording."""
    client = _make_client([TextBlock(type="text", text='"answer":"ok"}')])

    backend = AnthropicBackend(client)
    result = await backend.complete(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
    )

    assert isinstance(result.content, StructuredResponse)
    call = client.messages.create.await_args.kwargs
    assert call["messages"][-1] == {"role": "assistant", "content": "{"}
    instruction = call["messages"][0]["content"]
    assert "\n\nRespond with valid JSON matching this schema:" in instruction
    assert "If not responding with a tool call" not in instruction


@pytest.mark.asyncio
async def test_anthropic_backend_repairs_malformed_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured output that fails JSON parsing falls back to
    repair_response_model_json, whose result becomes the response content."""
    # After the '{' prefill is prepended this is still invalid JSON.
    client = _make_client([TextBlock(type="text", text='"answer": not-json')])

    repaired = StructuredResponse(answer="fixed")
    repair_calls: list[tuple[str, type[BaseModel], str]] = []

    def _fake_repair(
        raw: str, response_format: type[BaseModel], model_name: str
    ) -> StructuredResponse:
        repair_calls.append((raw, response_format, model_name))
        return repaired

    monkeypatch.setattr(
        "src.llm.backends.anthropic.repair_response_model_json", _fake_repair
    )

    backend = AnthropicBackend(client)
    result = await backend.complete(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
    )

    assert result.content is repaired
    assert repair_calls == [
        ('{"answer": not-json', StructuredResponse, "claude-3-5-sonnet-latest")
    ]


@pytest.mark.asyncio
async def test_anthropic_backend_skips_parsing_on_tool_call_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool-call response with response_format set must not attempt JSON
    parsing — the repair fallback raises on the empty text of tool-call
    turns, which would fail every intermediate tool iteration."""
    client = _make_client(
        [
            ToolUseBlock(
                type="tool_use",
                id="tool_1",
                name="search",
                input={"query": "honcho"},
            )
        ]
    )

    def _fail_repair(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("repair must not run for tool-call turns")

    monkeypatch.setattr(
        "src.llm.backends.anthropic.repair_response_model_json", _fail_repair
    )

    backend = AnthropicBackend(client)
    result = await backend.complete(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[SEARCH_TOOL],
        response_format=StructuredResponse,
    )

    assert result.content == ""  # raw (empty) text, not a parsed model
    assert result.tool_calls[0].name == "search"


@pytest.mark.asyncio
async def test_anthropic_backend_stream_no_prefill_when_tools_present() -> None:
    """The streaming path applies the same tools guard."""

    class _FakeStream:
        def __init__(self) -> None:
            self._chunks: list[SimpleNamespace] = [
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(text='{"answer":"ok"}'),
                )
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._chunks:
                return self._chunks.pop(0)
            raise StopAsyncIteration

        async def get_final_message(self):
            return SimpleNamespace(
                stop_reason="end_turn", usage=SimpleNamespace(output_tokens=5)
            )

    client = Mock()
    client.messages.stream = Mock(return_value=_FakeStream())

    backend = AnthropicBackend(client)
    chunks = [
        chunk
        async for chunk in backend.stream(
            model="claude-3-5-sonnet-latest",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            tools=[SEARCH_TOOL],
            response_format=StructuredResponse,
        )
    ]

    assert chunks[0].content == '{"answer":"ok"}'
    call = client.messages.stream.call_args.kwargs
    assert call["messages"][-1]["role"] == "user"  # no assistant '{' prefill
    assert (
        "If not responding with a tool call, respond with valid JSON"
        in call["messages"][0]["content"]
    )
