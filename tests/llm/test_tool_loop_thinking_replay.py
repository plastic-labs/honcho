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
from tenacity import wait_none

from src.config import ModelConfig
from src.llm import tool_loop
from src.llm.errors import llm_retry_condition
from src.llm.runtime import AttemptPlan, current_attempt
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
    model: str | Callable[[], AttemptPlan],
    responses: list[HonchoLLMCallResponse[Any]],
    *,
    max_tool_iterations: int,
    max_input_tokens: int | None = None,
    stream_final: bool = False,
    fail_call: int | None = None,
    retry_attempts: int = 1,
) -> tuple[
    list[dict[str, Any]],
    HonchoLLMCallResponse[Any] | StreamingResponseWithMetadata,
]:
    """Run the loop; return each call's model, messages and tools, and the result.

    `fail_call` (0-based) makes that provider call raise, so a retry runs.
    """
    calls: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake_call(*args: Any, **kwargs: Any) -> Any:
        calls.append(
            {
                "model": args[1],
                "messages": deepcopy(kwargs["messages"]),
                "tools": kwargs.get("tools"),
            }
        )
        if len(calls) - 1 == fail_call:
            raise RuntimeError("provider unavailable")
        if kwargs.get("stream"):

            async def chunks() -> AsyncIterator[HonchoLLMCallStreamChunk]:
                yield HonchoLLMCallStreamChunk(content="done", is_done=True)

            return chunks()
        return next(remaining)

    async def execute(_name: str, _input: dict[str, Any]) -> str:
        return "result"

    def next_attempt(retry_state: Any) -> None:
        current_attempt.set(retry_state.attempt_number + 1)

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=fake_call),
        patch.object(tool_loop, "wait_exponential", return_value=wait_none()),
    ):
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
            enable_retry=retry_attempts > 1,
            retry_attempts=retry_attempts,
            max_input_tokens=max_input_tokens,
            get_attempt_plan=_plan_for(model) if isinstance(model, str) else model,
            before_retry_callback=next_attempt,
            retry_condition=llm_retry_condition(
                can_fall_back=retry_attempts > 1,
                retry_attempts=retry_attempts,
                model="anthropic/test",
            ),
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
async def test_streamed_synthesis_drops_thinking() -> None:
    calls, _ = await _run_loop(
        "claude-sonnet-5-5",
        [_tool_call_turn("t1")],
        max_tool_iterations=1,
        stream_final=True,
    )

    streamed = calls[-1]
    assert streamed["tools"] is None
    assert _thinking_blocks_in(streamed["messages"]) == 0


def _cut_on_call(n: int) -> Callable[[list[dict[str, Any]], int], list[dict[str, Any]]]:
    """A truncation that changes history only on its `n`-th invocation (1-based)."""
    seen = 0

    def cut(messages: list[dict[str, Any]], _max_tokens: int) -> list[dict[str, Any]]:
        nonlocal seen
        seen += 1
        return list(messages) if seen == n else messages

    return cut


@pytest.mark.asyncio
async def test_thinking_produced_after_an_edit_is_kept() -> None:
    """Only blocks from before the edit go stale; later ones match the new prefix."""
    with patch(
        "src.llm.conversation.truncate_messages_to_fit", side_effect=_cut_on_call(2)
    ):
        calls, _ = await _run_loop(
            "claude-sonnet-5-5",
            [_tool_call_turn("t1"), _tool_call_turn("t2"), _answer()],
            max_tool_iterations=5,
            max_input_tokens=1_000,
        )

    assert [_thinking_blocks_in(call["messages"]) for call in calls] == [0, 0, 1]
    # The stale turn keeps its tool_use block.
    assert [block["type"] for block in calls[2]["messages"][1]["content"]] == [
        "tool_use"
    ]


@pytest.mark.asyncio
async def test_fallback_to_preserved_thinking_model_gets_stale_thinking_stripped() -> (
    None
):
    """The strip is decided per attempt: a Claude 4 primary keeps its blocks,
    and a Claude 5.5 fallback on the retry gets them without."""
    primary = _plan_for("claude-sonnet-4-5")
    fallback = _plan_for("claude-opus-5-5")

    def plan() -> AttemptPlan:
        return primary() if current_attempt.get() < 2 else fallback()

    with patch(
        "src.llm.conversation.truncate_messages_to_fit", side_effect=_cut_on_call(2)
    ):
        calls, _ = await _run_loop(
            plan,
            [_tool_call_turn("t1"), _answer()],
            max_tool_iterations=5,
            max_input_tokens=1_000,
            fail_call=1,
            retry_attempts=2,
        )

    assert [call["model"] for call in calls] == [
        "claude-sonnet-4-5",
        "claude-sonnet-4-5",
        "claude-opus-5-5",
    ]
    assert [_thinking_blocks_in(call["messages"]) for call in calls] == [0, 1, 0]
