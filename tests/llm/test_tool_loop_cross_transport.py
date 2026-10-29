"""Tool loops whose history moves between transports, through the real executor.

The tool loop keeps one typed history and each attempt renders it for its own
model. When a fallback on another transport answers a round, the primary's
next round must still receive its own wire shape: no `role: "tool"` messages
for Anthropic, no Anthropic content blocks for OpenAI, and no reasoning that
only the producing provider can read.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.config import ModelConfig, ModelTransport, ResolvedFallbackConfig
from src.llm import honcho_llm_call
from src.llm.backend import CompletionResult, ToolCallResult
from src.llm.types import BudgetLine, HonchoLLMCallResponse
from tests.llm.conftest import FakeBackend

SEARCH_TOOL = {
    "name": "search",
    "description": "Search memory",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}
SAVE_TOOL = {
    "name": "save",
    "description": "Save a conclusion",
    "input_schema": {"type": "object"},
}
MODELS: dict[ModelTransport, str] = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-5.5",
    "gemini": "gemini-3-pro",
}
THINKING = {"type": "thinking", "thinking": "look it up", "signature": "sig"}
DETAIL = {"type": "reasoning.encrypted", "data": "enc", "index": 0}


class _FailingFirst(FakeBackend):
    """Raises on its first call, then answers from its script."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses)
        self.failed: bool = False

    async def complete(self, **kwargs: Any) -> CompletionResult:
        if not self.failed:
            self.failed = True
            self.calls.append(kwargs)
            raise RuntimeError("primary overloaded")
        return await super().complete(**kwargs)


def _tool_turn(transport: ModelTransport, name: str = "search") -> CompletionResult:
    """A tool-call turn carrying the reasoning `transport` produces."""
    return CompletionResult(
        content=None if transport == "openai" else "",
        tool_calls=[
            ToolCallResult(
                id=f"call_{name}",
                name=name,
                input={"q": "tea"},
                thought_signature=b"sig" if transport == "gemini" else None,
            )
        ],
        thinking_blocks=[THINKING] if transport == "anthropic" else [],
        reasoning_details=[DETAIL] if transport == "openai" else [],
    )


def _assert_valid_for(
    transport: ModelTransport, messages: list[dict[str, Any]]
) -> None:
    """Every message is in `transport`'s own shape, and tool calls are answered."""
    if transport == "anthropic":
        for i, message in enumerate(messages):
            assert message["role"] in {"system", "user", "assistant"}
            content = message["content"]
            if isinstance(content, str):
                continue
            types = {block["type"] for block in content}
            assert types <= {"text", "tool_use", "tool_result", "thinking"}
            calls = [b["id"] for b in content if b["type"] == "tool_use"]
            if calls:
                results = messages[i + 1]["content"]
                assert [b["tool_use_id"] for b in results] == calls
    elif transport == "openai":
        for i, message in enumerate(messages):
            assert message["role"] in {"system", "user", "assistant", "tool"}
            assert not isinstance(message.get("content"), list)
            assert "parts" not in message
            calls = [c["id"] for c in message.get("tool_calls", [])]
            following = [
                m["tool_call_id"] for m in messages[i + 1 : i + 1 + len(calls)]
            ]
            assert following == calls
    else:
        for message in messages:
            assert message["role"] in {"system", "user", "assistant", "model"}
            assert not isinstance(message.get("content"), list)
            for part in message.get("parts", []):
                assert set(part) <= {
                    "text",
                    "function_call",
                    "function_response",
                    "thought_signature",
                }


async def _run(
    backends: dict[ModelTransport, FakeBackend],
    model_config: ModelConfig,
    *,
    max_tool_iterations: int,
    retry_attempts: int = 1,
    budget_line: BudgetLine | None = None,
    final_call_tools: list[dict[str, Any]] | None = None,
) -> tuple[HonchoLLMCallResponse[Any], list[str]]:
    executed: list[str] = []

    async def executor(name: str, _input: dict[str, Any]) -> str:
        executed.append(name)
        return f"{name} result"

    def backend_for(provider: ModelTransport, _client: object) -> FakeBackend:
        return backends[provider]

    with (
        patch(
            "src.llm.executor.backend_for_provider",
            side_effect=backend_for,
        ),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        response = await honcho_llm_call(
            model_config=model_config,
            prompt="",
            messages=[
                {"role": "system", "content": "You recall what users like."},
                {"role": "user", "content": "What does she drink?"},
            ],
            max_tokens=256,
            tools=[SEARCH_TOOL],
            tool_choice="auto",
            tool_executor=executor,
            max_tool_iterations=max_tool_iterations,
            retry_attempts=retry_attempts,
            budget_line=budget_line,
            final_call_tools=final_call_tools,
        )
    return response, executed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary", "fallback"),
    [("anthropic", "openai"), ("openai", "anthropic"), ("anthropic", "gemini")],
)
async def test_primary_resumes_after_a_fallback_on_another_transport(
    primary: ModelTransport, fallback: ModelTransport
) -> None:
    """Round one's primary call fails and the fallback answers with a tool call;
    round two goes back to the primary, which gets the fallback's turn in its own
    shape and none of the fallback's reasoning."""
    backends: dict[ModelTransport, FakeBackend] = {
        primary: _FailingFirst([CompletionResult(content="She drinks tea.")]),
        fallback: FakeBackend([_tool_turn(fallback)]),
    }
    model_config = ModelConfig(
        model=MODELS[primary],
        transport=primary,
        api_key="test-key",
        fallback=ResolvedFallbackConfig(
            model=MODELS[fallback], transport=fallback, api_key="test-key"
        ),
    )

    response, executed = await _run(
        backends, model_config, max_tool_iterations=3, retry_attempts=2
    )

    assert response.content == "She drinks tea."
    assert executed == ["search"]
    resumed = backends[primary].calls[-1]["messages"]
    _assert_valid_for(primary, resumed)
    assert "enc" not in repr(resumed)
    assert "look it up" not in repr(resumed)
    assert "b'sig'" not in repr(resumed)
    # The fallback's tool call and its result both reach the primary.
    assert "call_search" in repr(resumed)
    assert "search result" in repr(resumed)


def _budget_line(used: int, limit: int) -> str:
    return f"{used} of {limit} rounds used."


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["anthropic", "openai", "gemini"])
async def test_dreamer_shaped_loop_reaches_synthesis_with_final_call_tools(
    transport: ModelTransport,
) -> None:
    """System + user, tools, a budget line after each round, and a final call
    that may only save: every call stays in the transport's shape, and the
    final call runs only the offered tool."""
    backend = FakeBackend(
        [
            _tool_turn(transport),
            _tool_turn(transport),
            _tool_turn(transport, name="save"),
        ]
    )

    response, executed = await _run(
        {transport: backend},
        ModelConfig(model=MODELS[transport], transport=transport, api_key="k"),
        max_tool_iterations=2,
        budget_line=_budget_line,
        final_call_tools=[SAVE_TOOL],
    )

    assert response.hit_iteration_limit
    assert executed == ["search", "search", "save"]
    for call in backend.calls:
        _assert_valid_for(transport, call["messages"])
    final = backend.calls[-1]
    assert [tool["name"] for tool in final["tools"]] == ["save"]
    assert "2 of 2 rounds used." in repr(final["messages"])
    assert "maximum number of tool calls" in repr(final["messages"][-1])
