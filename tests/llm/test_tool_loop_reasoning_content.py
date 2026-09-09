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


def _make_plan() -> AttemptPlan:
    return AttemptPlan(
        provider="openai",
        model="deepseek-v4-pro",
        client=cast(ProviderClient, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(
            model="deepseek-v4-pro",
            transport="openai",
        ),
        attempt=1,
        retry_attempts=1,
        is_fallback=False,
    )


@pytest.mark.asyncio
async def test_tool_loop_replays_reasoning_content_on_continuation() -> None:
    calls: list[list[dict[str, Any]]] = []
    responses = iter(
        [
            HonchoLLMCallResponse(
                content="",
                output_tokens=5,
                finish_reasons=["tool_calls"],
                tool_calls_made=[
                    {
                        "id": "call_1",
                        "name": "search",
                        "input": {"query": "honcho"},
                    }
                ],
                thinking_content="DeepSeek reasoning",
            ),
            HonchoLLMCallResponse(
                content="done",
                output_tokens=3,
                finish_reasons=["stop"],
                tool_calls_made=[],
            ),
        ]
    )

    async def fake_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        calls.append(deepcopy(kwargs["messages"]))
        return next(responses)

    async def execute_search(_name: str, _input: dict[str, Any]) -> str:
        return "result"

    with patch.object(tool_loop, "honcho_llm_call_inner", new=fake_call):
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
            tool_executor=execute_search,
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _retry_state: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert len(calls) == 2
    assert calls[1][1]["reasoning_content"] == "DeepSeek reasoning"
    assert calls[1][1]["tool_calls"][0]["function"]["name"] == "search"
    assert calls[1][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "result",
    }


@pytest.mark.asyncio
async def test_tool_loop_replays_null_assistant_content_on_continuation() -> None:
    calls: list[list[dict[str, Any]]] = []
    responses = iter(
        [
            HonchoLLMCallResponse(
                content=None,
                output_tokens=5,
                finish_reasons=["tool_calls"],
                tool_calls_made=[
                    {
                        "id": "call_1",
                        "name": "search",
                        "input": {"query": "honcho"},
                    }
                ],
                reasoning_details=[
                    {
                        "type": "reasoning.encrypted",
                        "data": "opaque",
                        "format": "openai-responses-v1",
                        "id": "binding",
                        "index": 0,
                    }
                ],
            ),
            HonchoLLMCallResponse(
                content="done",
                output_tokens=3,
                finish_reasons=["stop"],
                tool_calls_made=[],
            ),
        ]
    )

    async def fake_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        calls.append(deepcopy(kwargs["messages"]))
        return next(responses)

    async def execute_search(_name: str, _input: dict[str, Any]) -> str:
        return "result"

    with patch.object(tool_loop, "honcho_llm_call_inner", new=fake_call):
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
            tool_executor=execute_search,
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _retry_state: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert len(calls) == 2
    assert calls[1][1]["content"] is None
    assert calls[1][1]["reasoning_details"][0]["data"] == "opaque"
    assert calls[1][1]["tool_calls"][0]["function"]["name"] == "search"
