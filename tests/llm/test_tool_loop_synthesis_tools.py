"""The max-iteration synthesis call and the tool definitions it sends."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest

from src.config import ModelConfig
from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse, ProviderClient

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


async def _synthesis_call_kwargs(provider: str) -> dict[str, Any]:
    """Run one tool round into the iteration limit; return the synthesis call's kwargs."""
    calls: list[dict[str, Any]] = []
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

    async def fake_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        calls.append(kwargs)
        return next(responses)

    async def execute(_name: str, _input: dict[str, Any]) -> str:
        return "result"

    with patch.object(tool_loop, "honcho_llm_call_inner", new=fake_call):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
            tool_choice="auto",
            tool_executor=execute,
            max_tool_iterations=1,
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
            stream_final=False,
            telemetry=None,
        )
    return calls[-1]


@pytest.mark.asyncio
async def test_anthropic_synthesis_defines_tools_but_disallows_calls() -> None:
    """Anthropic rejects tool_use/tool_result history unless tools are defined."""
    kwargs = await _synthesis_call_kwargs("anthropic")
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_other_providers_synthesize_without_tools() -> None:
    kwargs = await _synthesis_call_kwargs("openai")
    assert kwargs["tools"] is None
    assert kwargs["tool_choice"] is None
