"""Replayed thinking blocks after the tool loop edits history or changes tools.

Models with preserved thinking (Claude 5.5-era) reject a replayed thinking
block once the tool set or anything before it differs from what it was
produced under, so the loop drops the blocks at those points. Older Claude
models keep them: they require the current tool-use turn's thinking.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.config import ModelConfig
from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    ProviderClient,
    StreamingResponseWithMetadata,
)

SEARCH_TOOL = {
    "name": "search",
    "description": "Search",
    "input_schema": {"type": "object"},
}
THINKING_BLOCK = {"type": "thinking", "thinking": "plan", "signature": "sig"}


def _plan_for(model: str) -> Callable[[], AttemptPlan]:
    def make() -> AttemptPlan:
        return AttemptPlan(
            provider="anthropic",
            model=model,
            client=cast(ProviderClient, object()),
            thinking_budget_tokens=None,
            reasoning_effort=None,
            selected_config=ModelConfig(model=model, transport="anthropic"),
            attempt=1,
            retry_attempts=1,
            is_fallback=False,
        )

    return make


def _tool_call_turn(call_id: str) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="",
        output_tokens=5,
        finish_reasons=["tool_use"],
        tool_calls_made=[{"id": call_id, "name": "search", "input": {"q": "x"}}],
        thinking_blocks=[THINKING_BLOCK],
    )


def _answer() -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="done", output_tokens=3, finish_reasons=["end_turn"]
    )


def _thinking_blocks_in(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") in {"thinking", "redacted_thinking"}
    )


async def _run_loop(
    model: str,
    responses: list[HonchoLLMCallResponse[Any]],
    *,
    max_tool_iterations: int,
    max_input_tokens: int | None = None,
    stream_final: bool = False,
) -> tuple[
    list[dict[str, Any]],
    HonchoLLMCallResponse[Any] | StreamingResponseWithMetadata,
]:
    """Run the loop; return each call's messages and tools, and the result."""
    calls: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake_call(*_args: Any, **kwargs: Any) -> Any:
        calls.append(
            {"messages": deepcopy(kwargs["messages"]), "tools": kwargs.get("tools")}
        )
        if kwargs.get("stream"):

            async def chunks() -> AsyncIterator[HonchoLLMCallStreamChunk]:
                yield HonchoLLMCallStreamChunk(content="done", is_done=True)

            return chunks()
        return next(remaining)

    async def execute(_name: str, _input: dict[str, Any]) -> str:
        return "result"

    with patch.object(tool_loop, "honcho_llm_call_inner", new=fake_call):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=4096,
            messages=[{"role": "user", "content": "hi"}],
            tools=[SEARCH_TOOL],
            tool_choice="auto",
            tool_executor=execute,
            max_tool_iterations=max_tool_iterations,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=max_input_tokens,
            get_attempt_plan=_plan_for(model),
            before_retry_callback=lambda _retry_state: None,
            stream_final=stream_final,
            telemetry=None,
        )
        if isinstance(result, StreamingResponseWithMetadata):
            async for _chunk in result:
                pass
    return calls, result


@pytest.mark.asyncio
async def test_synthesis_drops_thinking_for_preserved_thinking_model() -> None:
    calls, _ = await _run_loop(
        "claude-sonnet-5-5", [_tool_call_turn("t1"), _answer()], max_tool_iterations=1
    )

    synthesis = calls[-1]
    assert synthesis["tools"] is None
    assert _thinking_blocks_in(synthesis["messages"]) == 0
    # The turn's tool_use block survives the strip.
    assistant = synthesis["messages"][1]
    assert [block["type"] for block in assistant["content"]] == ["tool_use"]


@pytest.mark.asyncio
async def test_synthesis_keeps_thinking_for_older_claude() -> None:
    calls, _ = await _run_loop(
        "claude-sonnet-4-5", [_tool_call_turn("t1"), _answer()], max_tool_iterations=1
    )

    assert _thinking_blocks_in(calls[-1]["messages"]) == 1


@pytest.mark.asyncio
async def test_unchanged_rounds_keep_thinking() -> None:
    """Rounds that only append to history replay thinking unchanged."""
    calls, _ = await _run_loop(
        "claude-sonnet-5-5",
        [_tool_call_turn("t1"), _tool_call_turn("t2"), _answer()],
        max_tool_iterations=5,
    )

    assert [_thinking_blocks_in(call["messages"]) for call in calls] == [0, 1, 2]


@pytest.mark.asyncio
async def test_truncation_drops_thinking_for_preserved_thinking_model() -> None:
    def cut(messages: list[dict[str, Any]], _max_tokens: int) -> list[dict[str, Any]]:
        return list(messages)  # a new list: the truncation changed history

    with patch("src.llm.conversation.truncate_messages_to_fit", side_effect=cut):
        calls, _ = await _run_loop(
            "claude-haiku-5-5",
            [_tool_call_turn("t1"), _answer()],
            max_tool_iterations=5,
            max_input_tokens=1_000,
        )

    assert _thinking_blocks_in(calls[1]["messages"]) == 0


@pytest.mark.asyncio
async def test_streamed_final_answer_drops_thinking() -> None:
    calls, _ = await _run_loop(
        "claude-sonnet-5-5",
        [_tool_call_turn("t1"), _answer()],
        max_tool_iterations=5,
        stream_final=True,
    )

    streamed = calls[-1]
    assert streamed["tools"] is None
    assert _thinking_blocks_in(streamed["messages"]) == 0
