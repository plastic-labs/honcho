"""The call that produces the answer after tool rounds, and the tools it defines."""

from __future__ import annotations

from collections.abc import AsyncIterator
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


async def _answer_call_kwargs(
    provider: str, *, max_tool_iterations: int = 1, stream_final: bool = False
) -> dict[str, Any]:
    """Run one tool round, then let the loop finish; return the answering call's kwargs.

    With one iteration the loop hits its limit and synthesizes; with more, the
    model answers in the loop (and under stream_final that answer is streamed
    afresh).
    """
    calls: list[dict[str, Any]] = []
    # Built per run: the loop mutates the responses it is handed.
    responses = iter(
        [
            HonchoLLMCallResponse(
                content="",
                output_tokens=5,
                finish_reasons=["tool_use"],
                tool_calls_made=[{"id": "t1", "name": "search", "input": {}}],
            ),
            HonchoLLMCallResponse(
                content="done", output_tokens=3, finish_reasons=["stop"]
            ),
        ]
    )

    async def fake_call(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if kwargs.get("stream"):

            async def chunks() -> AsyncIterator[HonchoLLMCallStreamChunk]:
                yield HonchoLLMCallStreamChunk(content="done")

            return chunks()
        return next(responses)

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
        if stream_final:
            assert isinstance(result, StreamingResponseWithMetadata)
            assert [c.content async for c in result] == ["done"]
            assert calls[-1]["stream"] is True
    return calls[-1]


_PATHS = pytest.mark.parametrize(
    "max_tool_iterations, stream_final",
    [(1, False), (1, True)],
    ids=["synthesis", "streamed-synthesis"],
)


@pytest.mark.asyncio
@_PATHS
async def test_anthropic_answer_call_defines_tools_but_disallows_calls(
    max_tool_iterations: int, stream_final: bool
) -> None:
    """Anthropic rejects tool_use/tool_result history unless tools are defined."""
    kwargs = await _answer_call_kwargs(
        "anthropic", max_tool_iterations=max_tool_iterations, stream_final=stream_final
    )
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == "none"


@pytest.mark.asyncio
@_PATHS
async def test_other_providers_answer_without_tools(
    max_tool_iterations: int, stream_final: bool
) -> None:
    kwargs = await _answer_call_kwargs(
        "openai", max_tool_iterations=max_tool_iterations, stream_final=stream_final
    )
    assert kwargs["tools"] is None
    assert kwargs["tool_choice"] is None
