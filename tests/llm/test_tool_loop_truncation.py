# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""Regression tests for `hit_input_token_cap` propagation through
`execute_tool_loop`.

The toolless path (`src/llm/api.py:325-340`) detects the cap hit up-front
by comparing input tokens against `max_input_tokens`. Before this fix,
the tool-loop path called `truncate_messages_to_fit` per iteration but
never propagated the flag — Dialectic (the main tool-loop consumer)
under-reported `hit_input_token_cap` on dialectic/representation events.

The rule is intentionally token-based, not message-count-based, so the
deriver's single-prompt path (where `truncate_messages_to_fit` keeps the
last unit even when oversized) still surfaces a real cap hit.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse


def _make_plan() -> AttemptPlan:
    # `selected_config=None` works for these tests since `_call_with_messages`
    # passes it straight through to the mocked `honcho_llm_call_inner`.
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


async def _terminating_call(*_args: Any, **_kwargs: Any) -> HonchoLLMCallResponse[Any]:
    # No tool calls — execute_tool_loop terminates after iteration 1.
    return HonchoLLMCallResponse(
        content="done",
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        finish_reasons=["stop"],
        tool_calls_made=[],
    )


@pytest.mark.asyncio
async def test_hit_input_token_cap_fires_when_input_exceeds_cap():
    """When the input message list exceeds `max_input_tokens` by token
    count, the response carries `hit_input_token_cap=True`. Regression
    check for the rule switch from message-count to token-based.
    """

    # Pretend the conversation totals 200 tokens; cap is 100.
    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_terminating_call),
        patch("src.llm.conversation.count_message_tokens", return_value=200),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=lambda msgs, _cap: msgs,
        ),
    ):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "huge"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=100,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.hit_input_token_cap is True


@pytest.mark.asyncio
async def test_hit_input_token_cap_false_when_under_cap():
    """Input tokens under cap → flag stays False, no false positive."""

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_terminating_call),
        patch("src.llm.conversation.count_message_tokens", return_value=50),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=lambda msgs, _cap: msgs,
        ),
    ):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "small"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=100,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.hit_input_token_cap is False


@pytest.mark.asyncio
async def test_hit_input_token_cap_fires_even_when_truncate_cant_shrink():
    """Critical regression: the single-message over-cap case (deriver's
    prompt-only call) used to silently return hit=False because
    `truncate_messages_to_fit` keeps the last unit even when oversized,
    making the old message-count-based check stay at False. The new
    token-based rule correctly catches this case.
    """

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_terminating_call),
        patch("src.llm.conversation.count_message_tokens", return_value=99_999),
        # Truncate is a no-op (matches real behavior for single-message inputs).
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=lambda msgs, _cap: msgs,
        ),
    ):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "x" * 1_000_000}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=1_000,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.hit_input_token_cap is True
