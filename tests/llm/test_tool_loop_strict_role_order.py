from __future__ import annotations

from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.config import ModelConfig
from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse, ProviderClient


def _openai_plan() -> AttemptPlan:
    return AttemptPlan(
        provider="openai",
        model="mistral-small",
        client=cast(ProviderClient, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(
            model="mistral-small",
            transport="openai",
        ),
        attempt=1,
        retry_attempts=1,
        is_fallback=False,
    )


def _response(
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content=content,
        output_tokens=1,
        finish_reasons=["tool_calls" if tool_calls else "stop"],
        tool_calls_made=tool_calls or [],
    )


def _tool_call() -> dict[str, Any]:
    return {
        "id": "call_1",
        "name": "search",
        "input": {"query": "honcho"},
    }


def _reject_tool_to_user(messages: list[dict[str, Any]]) -> None:
    for previous, current in zip(messages, messages[1:], strict=False):
        if previous.get("role") == "tool" and current.get("role") == "user":
            raise RuntimeError("Unexpected role 'user' after role 'tool'")


async def _execute_search(_name: str, _input: dict[str, Any]) -> str:
    return "result"


@pytest.mark.asyncio
async def test_max_iteration_synthesis_continues_after_tool_result() -> None:
    calls: list[list[dict[str, Any]]] = []
    responses = iter([_response("", tool_calls=[_tool_call()]), _response("done")])

    async def strict_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        messages = deepcopy(kwargs["messages"])
        calls.append(messages)
        _reject_tool_to_user(messages)
        return next(responses)

    with patch.object(tool_loop, "honcho_llm_call_inner", new=strict_call):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=_execute_search,
            max_tool_iterations=1,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=_openai_plan,
            before_retry_callback=lambda _retry_state: None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert [message["role"] for message in calls[-1][-2:]] == ["tool", "assistant"]
    assert "maximum number of tool calls" in calls[-1][-1]["content"]


@pytest.mark.asyncio
async def test_empty_response_retry_continues_after_tool_result() -> None:
    calls: list[list[dict[str, Any]]] = []
    responses = iter(
        [
            _response("", tool_calls=[_tool_call()]),
            _response(""),
            _response("done"),
        ]
    )

    async def strict_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        messages = deepcopy(kwargs["messages"])
        calls.append(messages)
        _reject_tool_to_user(messages)
        return next(responses)

    with patch.object(tool_loop, "honcho_llm_call_inner", new=strict_call):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=_execute_search,
            max_tool_iterations=3,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=_openai_plan,
            before_retry_callback=lambda _retry_state: None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert [message["role"] for message in calls[-1][-2:]] == ["tool", "assistant"]
    assert "last response was empty" in calls[-1][-1]["content"]
