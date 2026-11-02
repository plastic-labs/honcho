"""Under stream_final, the tool loop streams the answer it has, as the plain loop returns it."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from src.config import ModelConfig
from src.llm import tool_loop
from src.llm.errors import llm_retry_condition
from src.llm.runtime import AttemptPlan
from src.llm.structured_output import StructuredOutputError
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    ProviderClient,
    StreamingResponseWithMetadata,
)

TOOLS = [
    {"name": "search", "description": "Search", "input_schema": {"type": "object"}}
]


def _plan() -> AttemptPlan:
    return AttemptPlan(
        provider="openai",
        model="m",
        client=cast(ProviderClient, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(model="m", transport="openai"),
        attempt=1,
        retry_attempts=1,
        is_fallback=False,
    )


def _tool_round() -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="",
        input_tokens=100,
        output_tokens=5,
        finish_reasons=["tool_calls"],
        tool_calls_made=[{"id": "t1", "name": "search", "input": {}}],
    )


def _answer(content: Any) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content=content, input_tokens=120, output_tokens=7, finish_reasons=["stop"]
    )


async def _run(
    responses: list[HonchoLLMCallResponse[Any]],
    *,
    stream_final: bool,
    max_tool_iterations: int = 3,
    response_model: type[BaseModel] | None = None,
) -> tuple[list[dict[str, Any]], Any, str]:
    """Drive the loop over scripted responses: (call kwargs, result, answer text)."""
    calls: list[dict[str, Any]] = []
    scripted = iter(responses)

    async def fake_call(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if kwargs.get("stream"):

            async def chunks() -> AsyncIterator[HonchoLLMCallStreamChunk]:
                yield HonchoLLMCallStreamChunk(content="synthesized", output_tokens=4)

            return chunks()
        return next(scripted)

    async def execute(_name: str, _input: dict[str, Any]) -> str:
        return "result"

    with patch.object(tool_loop, "honcho_llm_call_inner", new=fake_call):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
            tool_choice="auto",
            tool_executor=execute,
            max_tool_iterations=max_tool_iterations,
            response_model=response_model,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=_plan,
            before_retry_callback=lambda _retry_state: None,
            retry_condition=llm_retry_condition(
                can_fall_back=False, retry_attempts=1, model="test/model"
            ),
            stream_final=stream_final,
            telemetry=None,
        )
        if stream_final:
            assert isinstance(result, StreamingResponseWithMetadata)
            text = "".join([chunk.content async for chunk in result])
        else:
            assert isinstance(result, HonchoLLMCallResponse)
            text = str(result.content)
    return calls, result, text


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_rounds", [0, 1], ids=["no-tools", "after-tools"])
async def test_the_streamed_answer_is_the_plain_loops_answer(tool_rounds: int) -> None:
    def script() -> list[HonchoLLMCallResponse[Any]]:
        return [_tool_round() for _ in range(tool_rounds)] + [_answer("the answer")]

    plain_calls, plain, plain_text = await _run(script(), stream_final=False)
    calls, streamed, text = await _run(script(), stream_final=True)

    assert text == plain_text == "the answer"
    # No second call to produce the answer: the one that made it is the last.
    assert len(calls) == len(plain_calls) == tool_rounds + 1
    assert not any(call.get("stream") for call in calls)
    # Token totals cover every call made, the answering one included.
    assert streamed.input_tokens == plain.input_tokens == 100 * tool_rounds + 120
    assert streamed.output_tokens == plain.output_tokens == 5 * tool_rounds + 7


@pytest.mark.asyncio
async def test_an_empty_answer_still_gets_the_nudge_when_streaming() -> None:
    calls, _, text = await _run([_answer(""), _answer("second try")], stream_final=True)

    assert text == "second try"
    assert len(calls) == 2 and not any(call.get("stream") for call in calls)
    assert calls[1]["messages"][-1]["content"].startswith(
        "Your last response was empty"
    )


@pytest.mark.asyncio
async def test_a_streamed_structured_answer_is_validated_and_sent_as_json() -> None:
    class Answer(BaseModel):
        value: str

    _, _, text = await _run(
        [_answer(Answer(value="x"))], stream_final=True, response_model=Answer
    )
    assert text == '{"value":"x"}'

    with pytest.raises(StructuredOutputError):
        await _run([_answer("not json")], stream_final=True, response_model=Answer)


@pytest.mark.asyncio
async def test_the_max_iteration_synthesis_still_streams() -> None:
    calls, streamed, text = await _run(
        [_tool_round()], stream_final=True, max_tool_iterations=1
    )

    assert text == "synthesized"
    assert calls[-1]["stream"] is True
    assert streamed.output_tokens == 5 + 4
