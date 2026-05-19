# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""Regression tests for `input_was_truncated` propagation through
`execute_tool_loop`.

The toolless path (`src/llm/api.py:325-335`) tracks truncation up-front
and stamps it on the returned response. Before this fix, the tool-loop
path called `truncate_messages_to_fit` per iteration but never propagated
the flag — Dialectic (the main tool-loop consumer) under-reported
`hit_input_token_cap` on representation/dialectic events.
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


@pytest.mark.asyncio
async def test_input_was_truncated_propagates_when_truncation_occurs():
    """When `truncate_messages_to_fit` actually clamps the message list,
    the final `HonchoLLMCallResponse` must carry `input_was_truncated=True`
    so RepresentationCompletedEvent / DialecticCompletedEvent populate
    `hit_input_token_cap` correctly.
    """

    async def _fake_llm_call(*_args: Any, **_kwargs: Any) -> HonchoLLMCallResponse[Any]:
        # No tool calls — the loop terminates immediately after iteration 1.
        return HonchoLLMCallResponse(
            content="done",
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            finish_reasons=["stop"],
            tool_calls_made=[],
        )

    # Force the truncation helper to report a shrink.
    def _shrinking_truncate(
        messages: list[dict[str, Any]], _cap: int
    ) -> list[dict[str, Any]]:
        # Drop one message to simulate cap enforcement.
        if len(messages) > 1:
            return messages[:-1]
        return messages

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_fake_llm_call),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=_shrinking_truncate,
        ),
    ):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
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
            max_input_tokens=10,  # Triggers the truncation path.
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.input_was_truncated is True


@pytest.mark.asyncio
async def test_input_was_truncated_false_when_no_clamp():
    """When the helper returns the message list unchanged, the flag
    stays False — no false positives for sub-cap inputs.
    """

    async def _fake_llm_call(*_args: Any, **_kwargs: Any) -> HonchoLLMCallResponse[Any]:
        return HonchoLLMCallResponse(
            content="done",
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            finish_reasons=["stop"],
            tool_calls_made=[],
        )

    # Truncation helper is a no-op — passes messages through unchanged.
    def _passthrough(messages: list[dict[str, Any]], _cap: int) -> list[dict[str, Any]]:
        return messages

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=_fake_llm_call),
        patch(
            "src.llm.conversation.truncate_messages_to_fit",
            side_effect=_passthrough,
        ),
    ):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "one"}],
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
            max_input_tokens=10_000,
            get_attempt_plan=_make_plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.input_was_truncated is False
