# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""`force_tools_until` gating of a forced tool_choice in `execute_tool_loop`."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse

_TOOLS = [
    {"name": name, "description": name, "input_schema": {"type": "object"}}
    for name in ("orient", "recall")
]


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
    """Replays one tool-call list per turn and records the tool_choice seen."""

    turns: list[list[str]]
    choices: list[Any]

    def __init__(self, turns: list[list[str]]) -> None:
        self.turns = list(turns)
        self.choices = []

    async def __call__(self, *_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        self.choices.append(kwargs.get("tool_choice"))
        return _response(self.turns.pop(0) if self.turns else [])


async def _echo_executor(name: str, _input: dict[str, Any]) -> str:
    return f"{name} result"


async def _run(model: _ScriptedModel, **overrides: Any) -> None:
    with patch.object(tool_loop, "honcho_llm_call_inner", new=model):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "q"}],
            tools=_TOOLS,
            tool_choice="required",
            tool_executor=overrides.pop("tool_executor", _echo_executor),
            max_tool_iterations=10,
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
            **overrides,
        )


@pytest.mark.asyncio
async def test_without_a_gate_any_tool_call_relaxes_after_one_round() -> None:
    model = _ScriptedModel([["orient"], []])
    await _run(model)
    assert model.choices == ["required", "auto"]


@pytest.mark.asyncio
async def test_orientation_call_keeps_the_forced_choice() -> None:
    model = _ScriptedModel([["orient"], ["recall"], []])
    await _run(model, force_tools_until={"recall"})
    assert model.choices == ["required", "required", "auto"]


@pytest.mark.asyncio
async def test_recall_call_relaxes_immediately() -> None:
    model = _ScriptedModel([["recall"], []])
    await _run(model, force_tools_until={"recall"})
    assert model.choices == ["required", "auto"]


@pytest.mark.asyncio
async def test_round_cap_relaxes_a_model_that_keeps_dodging() -> None:
    model = _ScriptedModel([["orient"], ["orient"], ["orient"], []])
    await _run(model, force_tools_until={"recall"}, max_forced_iterations=3)
    assert model.choices == ["required", "required", "required", "auto"]


@pytest.mark.asyncio
async def test_failed_recall_call_does_not_satisfy_the_gate() -> None:
    calls: list[str] = []

    async def flaky_executor(name: str, _input: dict[str, Any]) -> str:
        calls.append(name)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "ok"

    model = _ScriptedModel([["recall"], ["recall"], []])
    await _run(model, force_tools_until={"recall"}, tool_executor=flaky_executor)
    assert model.choices == ["required", "required", "auto"]


@pytest.mark.asyncio
async def test_executor_reported_error_does_not_satisfy_the_gate() -> None:
    """The real executor returns handler failures as strings, not exceptions."""
    from src.utils.agent_tools import create_tool_executor

    outcomes = iter(["ERROR: Query exceeds maximum token limit", "1 result"])

    async def recall_handler(_ctx: Any, _input: dict[str, Any]) -> str:
        return next(outcomes)

    executor = await create_tool_executor(
        workspace_name="w",
        observer="a",
        observed="a",
        handler_resolver=lambda name: recall_handler if name == "recall" else None,
    )
    model = _ScriptedModel([["recall"], ["recall"], []])
    await _run(model, force_tools_until={"recall"}, tool_executor=executor)
    assert model.choices == ["required", "required", "auto"]
