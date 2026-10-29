"""A failed attempt is recorded as a failure, with the output the model gave.

Both telemetry streams read one record per attempt from
`honcho_llm_call_inner`, so these cover the two ways an attempt fails after the
provider answered: the backend cannot parse the structured output, or the
caller's `validate_response` hook rejects the parsed response.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from src.llm import capture, executor, registry
from src.llm.backend import CompletionResult
from src.llm.capture import CapturedLLMCall
from src.llm.runtime import AttemptPlan
from src.llm.structured_output import StructuredOutputError
from src.telemetry.events import BaseEvent, LLMCallCompletedEvent


class Rejected(Exception):
    pass


def _plan() -> AttemptPlan:
    return AttemptPlan(
        provider="anthropic",
        model="claude-sonnet-4-5",
        client=cast(Any, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=cast(Any, object()),
        attempt=1,
        retry_attempts=3,
        is_fallback=False,
    )


def _reject(_response: Any) -> None:
    raise Rejected


@pytest.fixture
def recorded() -> Iterator[tuple[list[BaseEvent], list[CapturedLLMCall]]]:
    events: list[BaseEvent] = []
    calls: list[CapturedLLMCall] = []

    class _Spy:
        def export(self, call: CapturedLLMCall) -> None:
            calls.append(call)

    capture.clear_exporters()
    capture.register_exporter(_Spy())
    with (
        patch.object(registry, "CLIENTS", {"anthropic": object()}),
        patch.object(executor, "backend_for_provider", return_value=object()),
        patch("src.telemetry.events.emit", side_effect=events.append),
    ):
        yield events, calls
    capture.clear_exporters()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "validate_response", "error", "output", "output_tokens"),
    [
        pytest.param(
            AsyncMock(
                side_effect=StructuredOutputError(
                    "structured output failed", raw_content='{"explicit": [unq'
                )
            ),
            None,
            StructuredOutputError,
            '{"explicit": [unq',
            0,
            id="unparseable",
        ),
        pytest.param(
            AsyncMock(
                return_value=CompletionResult(
                    content="", output_tokens=7, finish_reason="length"
                )
            ),
            _reject,
            Rejected,
            "",
            7,
            id="rejected-by-validator",
        ),
    ],
)
async def test_failed_attempt_is_recorded_with_its_output(
    recorded: tuple[list[BaseEvent], list[CapturedLLMCall]],
    backend: AsyncMock,
    validate_response: Any,
    error: type[Exception],
    output: str,
    output_tokens: int,
) -> None:
    events, calls = recorded

    with (
        patch.object(executor, "execute_completion", new=backend),
        pytest.raises(error),
    ):
        await executor.honcho_llm_call_inner(
            "anthropic",
            "claude-sonnet-4-5",
            "hello",
            max_tokens=128,
            plan=_plan(),
            telemetry=None,
            validate_response=validate_response,
        )

    [event] = events
    assert isinstance(event, LLMCallCompletedEvent)
    assert event.outcome == "error"
    assert event.error_class == error.__name__
    assert event.provider_output_tokens == output_tokens

    [call] = calls
    assert call.outcome == "error"
    assert call.error_class == error.__name__
    assert call.output_content == output


@pytest.mark.asyncio
async def test_accepted_response_is_recorded_as_success(
    recorded: tuple[list[BaseEvent], list[CapturedLLMCall]],
) -> None:
    events, calls = recorded
    seen: list[Any] = []
    result = CompletionResult(content="ok", output_tokens=2, finish_reason="stop")

    with patch.object(
        executor, "execute_completion", new=AsyncMock(return_value=result)
    ):
        response = await executor.honcho_llm_call_inner(
            "anthropic",
            "claude-sonnet-4-5",
            "hello",
            max_tokens=128,
            plan=_plan(),
            telemetry=None,
            validate_response=seen.append,
        )

    assert seen == [response]
    [event] = events
    assert isinstance(event, LLMCallCompletedEvent)
    assert event.outcome == "success"
    [call] = calls
    assert call.outcome == "success"
    assert call.output_content == "ok"
