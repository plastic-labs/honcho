# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""Isolation tests for empty-final-response recovery in `execute_tool_loop`.

These prove — independent of any deployment, stored data, or LLM
nondeterminism — that an empty final answer (whether `""`, blank, or `None`)
with no tool calls is recovered rather than surfaced as `null`.

The recovery is decoupled from the iteration budget: while a retry and an
iteration remain, one in-loop nudge is tried; otherwise the loop breaks to the
forced synthesis call. The dialectic `minimal` tier (`MAX_TOOL_ITERATIONS=1`)
always takes the synthesis path because it has no iteration to loop on, and a
repeat empty response after the single nudge is routed to synthesis too. The
synthesis call is counted in `iterations`.

Covered here: empty-string and `None` at a single iteration; a repeat empty
after the nudge at two iterations; and the in-budget nudge control.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse


def _make_plan() -> AttemptPlan:
    """Return a minimal AttemptPlan the mocked inner call passes straight through."""
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


def _resp(content: str | None) -> HonchoLLMCallResponse[Any]:
    """Build a no-tool-call LLM response carrying the given content (may be None)."""
    return HonchoLLMCallResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        finish_reasons=["stop"],
        tool_calls_made=[],
    )


class _ScriptedLLM:
    """Mock inner-LLM driven by a script of per-tool-call contents.

    Each tool-enabled call (``tools`` passed) returns the next scripted content;
    empty/`None` entries drive the empty-response paths. The forced synthesis
    call (``tools=None``) always returns a fixed sentinel, so a test can tell
    recovery happened via synthesis rather than an in-loop nudge.
    """

    def __init__(
        self,
        tool_call_contents: list[str | None],
        synthesis: str = "recovered via synthesis",
    ) -> None:
        """Store the scripted tool-call contents and the synthesis sentinel."""
        self._contents = list(tool_call_contents)
        self._synthesis = synthesis
        self.tool_calls = 0
        self.synthesis_calls = 0

    async def __call__(self, *_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        """Return the next scripted content, or the sentinel when ``tools`` is None."""
        if kwargs.get("tools") is None:  # forced final synthesis call
            self.synthesis_calls += 1
            return _resp(self._synthesis)
        idx = self.tool_calls
        self.tool_calls += 1
        content = self._contents[idx] if idx < len(self._contents) else ""
        return _resp(content)


_TOOLS = [{"name": "noop", "description": "no-op", "input_schema": {"type": "object"}}]


async def _run(max_tool_iterations: int, mock: _ScriptedLLM) -> Any:
    """Drive execute_tool_loop with the mock at the given iteration budget."""
    return await execute_tool_loop(
        prompt="hi",
        max_tokens=250,
        messages=[{"role": "user", "content": "What do you know about me?"}],
        tools=_TOOLS,
        tool_choice="auto",
        tool_executor=lambda _name, _input: "",
        max_tool_iterations=max_tool_iterations,
        response_model=None,
        json_mode=False,
        temperature=None,
        stop_seqs=None,
        verbosity=None,
        enable_retry=False,
        retry_attempts=1,
        max_input_tokens=None,
        get_attempt_plan=_make_plan,
        before_retry_callback=lambda _r: None,
        stream_final=False,
        telemetry=None,
    )


@pytest.mark.asyncio
async def test_empty_string_recovered_at_single_iteration():
    """minimal tier: an empty ("") tool-less answer is recovered via synthesis,
    and `iterations` counts both the empty call and the synthesis call."""
    mock = _ScriptedLLM([""])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tool_loop, "honcho_llm_call_inner", mock)
        result = await _run(1, mock)

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.content == "recovered via synthesis"
    assert mock.synthesis_calls == 1
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_none_content_recovered_at_single_iteration():
    """A `None` completion (not just "") is treated as empty and recovered via
    synthesis — without this the string-only predicate would let it through."""
    mock = _ScriptedLLM([None])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tool_loop, "honcho_llm_call_inner", mock)
        result = await _run(1, mock)

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.content == "recovered via synthesis"
    assert mock.synthesis_calls == 1
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_repeat_empty_after_nudge_recovered_via_synthesis():
    """A second empty response after the single in-loop nudge is still recovered
    via synthesis rather than returned as null (initial + nudged retry +
    synthesis == 3 LLM calls)."""
    mock = _ScriptedLLM(["", ""])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tool_loop, "honcho_llm_call_inner", mock)
        result = await _run(2, mock)

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.content == "recovered via synthesis"
    assert mock.synthesis_calls == 1
    assert result.iterations == 3


@pytest.mark.asyncio
async def test_nudge_recovers_within_budget_without_synthesis():
    """Control: with an iteration to spare, a single empty response is recovered
    by the in-loop nudge, no synthesis call needed."""
    mock = _ScriptedLLM(["", "recovered via retry"])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tool_loop, "honcho_llm_call_inner", mock)
        result = await _run(2, mock)

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.content == "recovered via retry"
    assert mock.synthesis_calls == 0
    assert result.iterations == 2
