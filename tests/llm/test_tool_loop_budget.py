# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""The `budget_line` option on `execute_tool_loop`."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

import pytest

from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse

_TOOLS = [{"name": "look", "description": "look", "input_schema": {"type": "object"}}]


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


def _response(tool_names: list[str]) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="" if tool_names else "done",
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        finish_reasons=["tool_use" if tool_names else "stop"],
        tool_calls_made=[
            {"id": f"t{i}", "name": name, "input": {}}
            for i, name in enumerate(tool_names)
        ],
    )


class _ScriptedModel:
    """Replays one tool-call list per turn and snapshots the messages sent."""

    def __init__(self, turns: list[list[str]]) -> None:
        self.turns: list[list[str]] = list(turns)
        self.seen: list[list[dict[str, Any]]] = []

    async def __call__(self, *_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        self.seen.append(copy.deepcopy(kwargs["messages"]))
        return _response(self.turns.pop(0) if self.turns else [])


async def _echo_executor(name: str, _input: dict[str, Any]) -> str:
    return f"{name} result"


async def _run(model: _ScriptedModel, *, max_tool_iterations: int = 5, **kw: Any):
    with patch.object(tool_loop, "honcho_llm_call_inner", new=model):
        return await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
            tools=_TOOLS,
            tool_choice=None,
            tool_executor=_echo_executor,
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
            **kw,
        )


def _user_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [
        m["content"]
        for m in messages
        if m["role"] == "user" and isinstance(m["content"], str)
    ]


@pytest.mark.asyncio
async def test_budget_line_follows_each_round_of_tool_results() -> None:
    model = _ScriptedModel([["look"], ["look"], []])
    await _run(model, budget_line=lambda used, limit: f"used {used} of {limit}")

    assert _user_texts(model.seen[1])[-1] == "used 1 of 5"
    # Earlier lines stay put, so the prefix sent on the previous call is
    # unchanged and remains cacheable.
    assert _user_texts(model.seen[2])[-2:] == ["used 1 of 5", "used 2 of 5"]
    assert model.seen[2][: len(model.seen[1])] == model.seen[1]
    # The line is the last message the model sees.
    assert model.seen[2][-1] == {"role": "user", "content": "used 2 of 5"}


@pytest.mark.asyncio
async def test_none_from_budget_line_appends_nothing() -> None:
    model = _ScriptedModel([["look"], []])
    await _run(model, budget_line=lambda _used, _limit: None)
    assert _user_texts(model.seen[1]) == ["q"]


@pytest.mark.asyncio
async def test_no_budget_line_by_default() -> None:
    model = _ScriptedModel([["look"], []])
    await _run(model)
    assert _user_texts(model.seen[1]) == ["q"]
