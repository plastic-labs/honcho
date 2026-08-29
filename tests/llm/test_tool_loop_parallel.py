# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportArgumentType=false
"""Tool calls emitted in a single assistant turn execute concurrently.

The model routinely asks for several independent reads at once — a dialectic
turn typically emits `search_memory` and `search_messages` together. Running
them one after another makes the turn cost the sum of their latencies when it
only needs to cost the slowest: measured against production, a `minimal`
dialectic spent 1.2s in `search_memory` and then a further 1.9s in
`search_messages`, both pure reads.

Tool handlers open their own short-lived sessions via `tracked_db()` and only
the mutating handlers (`create_observations`, `update_peer_card`,
`delete_observations`) take `ctx.db_lock`, so concurrent reads are safe. The
per-call telemetry ContextVars are set inside each task, and `asyncio` copies
the context per task, so sequence numbers and last-tool metadata stay bound to
their own call instead of racing on one shared context.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import MAX_CONCURRENT_TOOL_CALLS, execute_tool_loop
from src.llm.types import HonchoLLMCallResponse

TOOL_DELAY_SECONDS = 0.2


def _make_plan() -> AttemptPlan:
    return AttemptPlan(
        provider="anthropic",
        model="claude-sonnet-4-5",
        client=object(),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=None,
        attempt=1,
        retry_attempts=1,
        is_fallback=False,
    )


def _response(tool_calls: list[dict[str, Any]]) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="done",
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        finish_reasons=["stop"],
        tool_calls_made=tool_calls,
    )


class _Tracker:
    """Counts how many tool executions are in flight at once."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def execute(self, _name: str, _input: dict[str, Any]) -> str:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(TOOL_DELAY_SECONDS)
        finally:
            self.in_flight -= 1
        return "ok"


async def _run(tracker: _Tracker, n_calls: int = 2) -> HonchoLLMCallResponse[Any]:
    names = ["search_memory", "search_messages"]
    calls = [
        {"name": names[i % len(names)], "input": {"query": str(i)}, "id": f"t{i}"}
        for i in range(n_calls)
    ]
    responses = iter([_response(calls), _response([])])

    async def _call(*_args: Any, **_kwargs: Any) -> HonchoLLMCallResponse[Any]:
        return next(responses)

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_call),
        patch("src.llm.conversation.count_message_tokens", return_value=10),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=lambda msgs, _cap: msgs,
        ),
    ):
        return await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "q"}],
            tools=[
                {
                    "name": "search_memory",
                    "description": "",
                    "input_schema": {"type": "object"},
                },
                {
                    "name": "search_messages",
                    "description": "",
                    "input_schema": {"type": "object"},
                },
            ],
            tool_choice="auto",
            tool_executor=tracker.execute,
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=1000,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )


@pytest.mark.asyncio
async def test_tool_calls_in_one_iteration_run_concurrently():
    """Two tools requested in one turn overlap rather than queueing."""
    tracker = _Tracker()

    started = asyncio.get_running_loop().time()
    await _run(tracker)
    elapsed = asyncio.get_running_loop().time() - started

    assert tracker.peak == 2, f"tools ran sequentially (peak in-flight {tracker.peak})"
    # Sequential would be >= 2 * delay; concurrent stays near one delay.
    assert elapsed < TOOL_DELAY_SECONDS * 1.8, (
        f"elapsed {elapsed:.3f}s suggests serial execution"
    )


@pytest.mark.asyncio
async def test_results_keep_request_order():
    """Concurrency must not reorder results relative to the calls."""
    tracker = _Tracker()

    result = await _run(tracker)

    names = [call["tool_name"] for call in result.tool_calls_made]
    assert names == ["search_memory", "search_messages"]


@pytest.mark.asyncio
async def test_fan_out_is_capped():
    """A turn asking for many tools does not stampede the database.

    Production has produced 18 tool calls in a single iteration; firing all of
    them at once would mean that many concurrent embedding + pgvector queries
    on one instance.
    """
    tracker = _Tracker()

    await _run(tracker, n_calls=18)

    assert tracker.peak <= MAX_CONCURRENT_TOOL_CALLS, (
        f"fan-out reached {tracker.peak}, above the {MAX_CONCURRENT_TOOL_CALLS} cap"
    )
    assert tracker.peak > 1, "cap must not serialise execution entirely"
