"""How the tool loop produces its answer, and the tools the answering call defines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel

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

TOOLS = [
    {"name": "search", "description": "Search", "input_schema": {"type": "object"}}
]


def _plan(provider: str) -> AttemptPlan:
    return AttemptPlan(
        provider=cast(Any, provider),
        model="m",
        client=cast(ProviderClient, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(model="m", transport=cast(Any, provider)),
        attempt=1,
        retry_attempts=1,
        is_fallback=False,
    )


def _tool_round() -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="",
        input_tokens=100,
        output_tokens=5,
        finish_reasons=["tool_use"],
        tool_calls_made=[{"id": "t1", "name": "search", "input": {}}],
    )


def _answer(content: Any = "done") -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content=content, input_tokens=120, output_tokens=3, finish_reasons=["stop"]
    )


async def _run(
    provider: str,
    responses: list[HonchoLLMCallResponse[Any]],
    *,
    max_tool_iterations: int,
    stream_final: bool,
) -> tuple[list[dict[str, Any]], Any, list[str]]:
    """Drive the loop over scripted responses.

    Returns every LLM call's kwargs, the loop's result, and (under
    stream_final) the streamed chunks.
    """
    calls: list[dict[str, Any]] = []
    scripted = iter(responses)

    async def fake_call(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if kwargs.get("stream"):

            async def chunks() -> AsyncIterator[HonchoLLMCallStreamChunk]:
                yield HonchoLLMCallStreamChunk(content="streamed")

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
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=lambda: _plan(provider),
            before_retry_callback=lambda _retry_state: None,
            stream_final=stream_final,
            telemetry=None,
        )
        streamed: list[str] = []
        if stream_final:
            assert isinstance(result, StreamingResponseWithMetadata)
            streamed = [c.content async for c in result]
    return calls, result, streamed


_SYNTHESIS = pytest.mark.parametrize("stream_final", [False, True])


@pytest.mark.asyncio
@_SYNTHESIS
async def test_anthropic_synthesis_defines_tools_but_disallows_calls(
    stream_final: bool,
) -> None:
    """Anthropic rejects tool_use/tool_result history unless tools are defined."""
    calls, _, _ = await _run(
        "anthropic",
        [_tool_round(), _answer()],
        max_tool_iterations=1,
        stream_final=stream_final,
    )
    assert calls[-1]["tools"] == TOOLS
    assert calls[-1]["tool_choice"] == "none"


@pytest.mark.asyncio
@_SYNTHESIS
async def test_other_providers_synthesize_without_tools(stream_final: bool) -> None:
    calls, _, _ = await _run(
        "openai",
        [_tool_round(), _answer()],
        max_tool_iterations=1,
        stream_final=stream_final,
    )
    assert calls[-1]["tools"] is None
    assert calls[-1]["tool_choice"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("tool_rounds", [0, 1], ids=["no-tools", "after-tools"])
async def test_a_streamed_answer_is_the_loops_answer_not_a_second_call(
    provider: str, tool_rounds: int
) -> None:
    """Asking again cost a full call and could come back empty (thinking only)."""
    responses = [_tool_round()] * tool_rounds + [_answer("the loop's answer")]

    calls, result, streamed = await _run(
        provider, responses, max_tool_iterations=3, stream_final=True
    )

    assert streamed == ["the loop's answer"]
    assert len(calls) == tool_rounds + 1
    assert not any(c.get("stream") for c in calls)
    # Every call made is counted, the answering one included.
    assert result.input_tokens == 100 * tool_rounds + 120
    assert result.output_tokens == 5 * tool_rounds + 3


@pytest.mark.asyncio
async def test_a_structured_answer_streams_as_json() -> None:
    class Answer(BaseModel):
        value: str

    _, _, streamed = await _run(
        "openai", [_answer(Answer(value="x"))], max_tool_iterations=3, stream_final=True
    )
    assert streamed == ['{"value":"x"}']


@pytest.mark.asyncio
async def test_an_empty_last_answer_is_asked_for_again() -> None:
    """With no empty-answer nudge left, the loop still asks once for a streamed answer."""
    calls, _, streamed = await _run(
        "anthropic",
        [_tool_round(), _answer("")],
        max_tool_iterations=2,
        stream_final=True,
    )
    assert streamed == ["streamed"]
    assert calls[-1]["stream"] is True
    assert calls[-1]["tools"] == TOOLS and calls[-1]["tool_choice"] == "none"
