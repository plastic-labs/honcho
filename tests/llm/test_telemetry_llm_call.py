# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""Phase 1 tests for LLMCallCompletedEvent emission and the high-volume sampler.

Targets:
- `src/llm/executor.py::honcho_llm_call_inner` emits one event per call,
  on both success and failure (try/finally).
- `LLMTelemetryContext` round-trips workspace/run_id/iteration/call_purpose
  onto the event without mutating the caller-supplied context.
- The sampler at `src/telemetry/emitter.py::_should_sample` keeps every event
  of a run together (same run_id → same decision) and lets ground_truth
  events through unconditionally.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.llm.backend import CompletionResult as BackendCompletionResult
from src.llm.executor import _emit_llm_call_completed
from src.llm.runtime import AttemptPlan
from src.llm.types import LLMTelemetryContext
from src.telemetry.events import (
    BaseEvent,
    CallPurpose,
    DialecticCompletedEvent,
    LLMCallCompletedEvent,
)


def _make_plan(
    *, attempt: int = 1, retry_attempts: int = 3, is_fallback: bool = False
) -> AttemptPlan:
    """Minimal AttemptPlan; client/selected_config are unused by the emitter helper."""
    return AttemptPlan(
        provider="anthropic",
        model="claude-sonnet-4-5",
        client=object(),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=object(),
        attempt=attempt,
        retry_attempts=retry_attempts,
        is_fallback=is_fallback,
    )


class TestEmitLLMCallCompleted:
    def test_emits_success_event(self):
        emitted: list[BaseEvent] = []

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_llm_call_completed(
                plan=_make_plan(),
                telemetry=LLMTelemetryContext(
                    workspace_name="ws1",
                    call_purpose=CallPurpose.DIALECTIC_ANSWER.value,
                    parent_category="dialectic",
                    run_id="run-xyz",
                    iteration=2,
                ),
                provider="anthropic",
                model="claude-sonnet-4-5",
                max_tokens=2048,
                duration_ms=300.0,
                has_tools=True,
                was_stream=False,
                outcome="success",
                result=BackendCompletionResult(
                    content="hi",
                    input_tokens=10,
                    output_tokens=5,
                    cache_read_input_tokens=2,
                    cache_creation_input_tokens=1,
                    finish_reason="stop",
                ),
                error=None,
            )

        assert len(emitted) == 1
        event = emitted[0]
        assert isinstance(event, LLMCallCompletedEvent)
        assert event.outcome == "success"
        assert event.is_final_attempt is False
        assert event.workspace_name == "ws1"
        assert event.run_id == "run-xyz"
        assert event.iteration == 2
        assert event.call_purpose == CallPurpose.DIALECTIC_ANSWER
        assert event.provider_input_tokens == 10
        assert event.provider_output_tokens == 5
        assert event.cache_read_tokens == 2
        assert event.cache_creation_tokens == 1
        assert event.finish_reason == "stop"
        assert event.has_tools is True

    def test_emits_error_event_with_class_name(self):
        emitted: list[BaseEvent] = []

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_llm_call_completed(
                plan=_make_plan(attempt=3, retry_attempts=3, is_fallback=True),
                telemetry=None,
                provider="openai",
                model="gpt-4",
                max_tokens=512,
                duration_ms=15.0,
                has_tools=False,
                was_stream=False,
                outcome="error",
                result=None,
                error=RuntimeError("nope"),
            )

        assert len(emitted) == 1
        event = emitted[0]
        assert isinstance(event, LLMCallCompletedEvent)
        assert event.outcome == "error"
        # On the last attempt, is_final_attempt must be True (replaces the
        # synthetic "retry_exhausted" outcome from earlier drafts).
        assert event.is_final_attempt is True
        assert event.error_class == "RuntimeError"
        assert event.was_fallback is True
        # No result → token fields are 0.
        assert event.provider_input_tokens == 0
        assert event.provider_output_tokens == 0

    def test_unknown_call_purpose_silently_dropped(self):
        """Unknown call_purpose strings should not raise; event still emits."""
        emitted: list[BaseEvent] = []

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_llm_call_completed(
                plan=_make_plan(),
                telemetry=LLMTelemetryContext(call_purpose="not.a.real.purpose"),
                provider="anthropic",
                model="claude",
                max_tokens=1,
                duration_ms=0.0,
                has_tools=False,
                was_stream=False,
                outcome="success",
                result=BackendCompletionResult(),
                error=None,
            )

        assert len(emitted) == 1
        event = emitted[0]
        assert isinstance(event, LLMCallCompletedEvent)
        # Unknown purpose drops to None rather than raising.
        assert event.call_purpose is None

    def test_provider_label_inferred_from_openrouter_model_prefix(self):
        emitted: list[BaseEvent] = []

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_llm_call_completed(
                plan=_make_plan(),
                telemetry=None,
                provider="openai",
                model="anthropic/claude-3-5-sonnet",
                max_tokens=1,
                duration_ms=0.0,
                has_tools=False,
                was_stream=False,
                outcome="success",
                result=BackendCompletionResult(),
                error=None,
            )

        event = emitted[0]
        assert isinstance(event, LLMCallCompletedEvent)
        assert event.provider_label == "anthropic"

    def test_telemetry_failures_swallowed(self):
        """A broken emitter must NOT propagate exceptions out of the LLM path."""

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("emitter wedged")

        with patch("src.telemetry.events.emit", side_effect=explode):
            # Must not raise.
            _emit_llm_call_completed(
                plan=_make_plan(),
                telemetry=None,
                provider="anthropic",
                model="claude",
                max_tokens=1,
                duration_ms=0.0,
                has_tools=False,
                was_stream=False,
                outcome="success",
                result=BackendCompletionResult(),
                error=None,
            )


class TestSampler:
    """Tests for the deterministic high-volume sampler."""

    def test_rate_one_passes_everything(self):
        from src.telemetry.emitter import _should_sample

        event = LLMCallCompletedEvent(
            transport="anthropic",
            model="m",
            effective_max_output_tokens=1,
            outcome="success",
            is_final_attempt=True,
            attempt=1,
            retry_attempts=1,
            was_fallback=False,
            duration_ms=0.0,
            run_id="anything",
        )
        assert _should_sample(event, 1.0) is True

    def test_rate_zero_drops_everything(self):
        from src.telemetry.emitter import _should_sample

        event = LLMCallCompletedEvent(
            transport="anthropic",
            model="m",
            effective_max_output_tokens=1,
            outcome="success",
            is_final_attempt=True,
            attempt=1,
            retry_attempts=1,
            was_fallback=False,
            duration_ms=0.0,
            run_id="anything",
        )
        assert _should_sample(event, 0.0) is False

    def test_same_run_id_gets_same_decision(self):
        """Two events with the same run_id must hash to the same bucket so an
        entire trace is either kept or dropped — never half-sampled."""
        from src.telemetry.emitter import _should_sample

        def make(iteration: int) -> LLMCallCompletedEvent:
            return LLMCallCompletedEvent(
                transport="anthropic",
                model="m",
                effective_max_output_tokens=1,
                outcome="success",
                is_final_attempt=False,
                attempt=1,
                retry_attempts=1,
                was_fallback=False,
                duration_ms=0.0,
                run_id="stable-run-id",
                iteration=iteration,
            )

        rate = 0.5
        a = _should_sample(make(1), rate)
        b = _should_sample(make(2), rate)
        c = _should_sample(make(3), rate)
        assert a == b == c


class TestExecutorEndToEnd:
    """Exercise honcho_llm_call_inner's try/finally on both paths."""

    @pytest.mark.asyncio
    async def test_success_path_emits_one_event(self):
        from src.llm import executor

        emitted: list[BaseEvent] = []
        result = BackendCompletionResult(
            content="ok", input_tokens=3, output_tokens=2, finish_reason="stop"
        )

        with (
            patch.object(executor, "CLIENTS", {"anthropic": object()}),
            patch.object(
                executor,
                "backend_for_provider",
                return_value=object(),
            ),
            patch.object(
                executor,
                "execute_completion",
                new=AsyncMock(return_value=result),
            ),
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
        ):
            await executor.honcho_llm_call_inner(
                "anthropic",
                "claude-sonnet-4-5",
                "hello",
                max_tokens=128,
                plan=_make_plan(),
                telemetry=LLMTelemetryContext(
                    workspace_name="ws",
                    call_purpose=CallPurpose.DERIVER_REPRESENTATION.value,
                    parent_category="representation",
                ),
            )

        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, LLMCallCompletedEvent)
        assert ev.outcome == "success"
        assert ev.provider_output_tokens == 2

    @pytest.mark.asyncio
    async def test_error_path_still_emits_via_finally(self):
        from src.llm import executor

        emitted: list[BaseEvent] = []

        async def _boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("backend exploded")

        with (
            patch.object(executor, "CLIENTS", {"anthropic": object()}),
            patch.object(
                executor,
                "backend_for_provider",
                return_value=object(),
            ),
            patch.object(executor, "execute_completion", new=_boom),
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            pytest.raises(RuntimeError),
        ):
            await executor.honcho_llm_call_inner(
                "anthropic",
                "claude-sonnet-4-5",
                "hello",
                max_tokens=128,
                plan=_make_plan(attempt=3, retry_attempts=3),
                telemetry=None,
            )

        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, LLMCallCompletedEvent)
        assert ev.outcome == "error"
        assert ev.is_final_attempt is True
        assert ev.error_class == "RuntimeError"


def test_ground_truth_event_skips_sampler():
    """DialecticCompletedEvent declares _volume_class='ground_truth' (default
    on BaseEvent) so the sampler should never gate it, even at rate 0.0."""
    from src.telemetry.emitter import _should_sample

    # The sampler is only consulted by emit() when volume_class == 'high_volume'.
    # Verify the class declaration directly so future refactors of emit() can't
    # silently start sampling ground-truth events.
    assert DialecticCompletedEvent.volume_class() == "ground_truth"
    # And the sampler itself, while we're here, must be deterministic 1.0 → True.
    event = DialecticCompletedEvent(
        run_id="r",
        workspace_name="ws",
        peer_name="p",
        reasoning_level="medium",
        total_duration_ms=10.0,
        input_tokens=1,
        output_tokens=1,
    )
    assert _should_sample(event, 1.0) is True
