# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""tests for LLMCallCompletedEvent emission and the high-volume sampler.

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

from typing import Any, cast
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
    async def test_cancelled_path_emits_cancelled_outcome(self):
        """asyncio.CancelledError mid-call surfaces as outcome='cancelled', not
        'error' — client disconnects / shutdowns must not pollute error rates."""
        import asyncio

        from src.llm import executor

        emitted: list[BaseEvent] = []

        async def _cancel(*_args: Any, **_kwargs: Any) -> Any:
            raise asyncio.CancelledError()

        with (
            patch.object(executor, "CLIENTS", {"anthropic": object()}),
            patch.object(executor, "backend_for_provider", return_value=object()),
            patch.object(executor, "execute_completion", new=_cancel),
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await executor.honcho_llm_call_inner(
                "anthropic",
                "claude-sonnet-4-5",
                "hello",
                max_tokens=128,
                plan=_make_plan(),
                telemetry=None,
            )

        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, LLMCallCompletedEvent)
        assert ev.outcome == "cancelled"
        assert ev.error_class == "CancelledError"

    @pytest.mark.asyncio
    async def test_stream_cancelled_emits_cancelled_outcome(self):
        """Stream path: mid-iteration CancelledError surfaces as 'cancelled'."""
        import asyncio
        from collections.abc import AsyncIterator

        from src.llm import executor

        emitted: list[BaseEvent] = []

        async def _cancelling_stream() -> AsyncIterator[Any]:
            # one chunk then cancel — simulates a client disconnect mid-stream.
            yield object()  # caller's `async for` consumes this
            raise asyncio.CancelledError()

        async def _setup_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            return _cancelling_stream()

        with (
            patch.object(executor, "CLIENTS", {"anthropic": object()}),
            patch.object(executor, "backend_for_provider", return_value=object()),
            patch.object(executor, "execute_stream", new=_setup_stream),
            patch.object(
                executor,
                "stream_chunk_to_response_chunk",
                side_effect=lambda chunk: chunk,
            ),
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
        ):
            stream = await executor.honcho_llm_call_inner(
                "anthropic",
                "claude-sonnet-4-5",
                "hello",
                max_tokens=128,
                plan=_make_plan(),
                telemetry=None,
                stream=True,
            )
            with pytest.raises(asyncio.CancelledError):
                async for _ in stream:
                    pass

        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, LLMCallCompletedEvent)
        assert ev.outcome == "cancelled"
        assert ev.was_stream is True

    @pytest.mark.asyncio
    async def test_stream_setup_failure_emits_and_propagates(self):
        """Stream-setup errors must propagate out of the AWAITED
        `honcho_llm_call_inner` call (not deferred until first iteration),
        so the outer retry wrapper in tool_loop.stream_final_response sees
        them. Regression check for the bug where `_stream()` returned a
        generator without awaiting `execute_stream`, hiding setup failures
        from tenacity.
        """
        from src.llm import executor

        emitted: list[BaseEvent] = []

        async def _setup_explodes(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("rate limited")

        with (
            patch.object(executor, "CLIENTS", {"anthropic": object()}),
            patch.object(executor, "backend_for_provider", return_value=object()),
            patch.object(executor, "execute_stream", new=_setup_explodes),
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            pytest.raises(RuntimeError, match="rate limited"),
        ):
            # The await itself must raise — that's how tenacity sees it.
            await executor.honcho_llm_call_inner(
                "anthropic",
                "claude-sonnet-4-5",
                "hello",
                max_tokens=128,
                plan=_make_plan(),
                telemetry=None,
                stream=True,
            )

        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, LLMCallCompletedEvent)
        assert ev.outcome == "error"
        assert ev.was_stream is True
        assert ev.error_class == "RuntimeError"

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
    on BaseEvent) so the sampler should never gate it, even at rate 0.0.

    Drives the real emitter under a zero-rate config and asserts the
    ground-truth event still lands in the buffer — the actual bypass
    behavior in emit() — not just the helper-function semantics in
    _should_sample. Regression guard for a future refactor that pushes
    ground_truth events through the sampling decision.
    """
    from src.telemetry.emitter import TelemetryEmitter, _should_sample

    # The volume_class declaration is the gate emit() consults.
    assert DialecticCompletedEvent.volume_class() == "ground_truth"

    event = DialecticCompletedEvent(
        run_id="r",
        workspace_name="ws",
        peer_name="p",
        reasoning_level="medium",
        total_duration_ms=10.0,
        input_tokens=1,
        output_tokens=1,
    )
    # Sampler is deterministic 1.0 → True for any event (sanity).
    assert _should_sample(event, 1.0) is True

    # Drive the emitter directly under rate=0.0. enabled=True requires a
    # non-None endpoint; we use a placeholder URL — the emitter buffers but
    # we never flush, so no HTTP traffic is generated. The buffer growing
    # proves the ground_truth event bypassed the sampler.
    emitter = TelemetryEmitter(endpoint="http://test/events", enabled=True)
    with patch("src.config.settings") as mock_settings:
        mock_settings.TELEMETRY.HIGH_VOLUME_SAMPLE_RATE = 0.0
        mock_settings.TELEMETRY.NAMESPACE = "test"
        emitter.emit(event)
    assert emitter.buffer_size == 1  # ground_truth survived the sampler

    # Sanity check: a high-volume event under rate=0.0 DOES get dropped,
    # proving the test setup actually exercises the sampling code path.
    from src.telemetry.events import LLMCallCompletedEvent

    sampled_event = LLMCallCompletedEvent(
        transport="anthropic",
        model="m",
        effective_max_output_tokens=1,
        finish_reason="stop",
        outcome="success",
        is_final_attempt=True,
        attempt=1,
        retry_attempts=1,
        was_fallback=False,
        duration_ms=1.0,
        has_tools=False,
        was_stream=False,
    )
    assert sampled_event.volume_class() == "high_volume"
    with patch("src.config.settings") as mock_settings:
        mock_settings.TELEMETRY.HIGH_VOLUME_SAMPLE_RATE = 0.0
        mock_settings.TELEMETRY.NAMESPACE = "test"
        emitter.emit(sampled_event)
    # Buffer still 1 — high_volume event was sampled out.
    assert emitter.buffer_size == 1


class TestStreamFinalResponseRetryAttempt:
    """`stream_final_response` (src/llm/tool_loop.py) must bump the
    per-attempt index on the plan it passes to `honcho_llm_call_inner`.
    Previously every retried stream-setup emit reported the same `attempt`
    value because the pinned `winning_plan` was reused unchanged. Fix 13
    plumbs a per-retry plan via `dataclasses.replace`.
    """

    @pytest.mark.asyncio
    async def test_attempt_index_bumps_across_retries(self):
        from collections.abc import AsyncIterator

        from src.llm import executor, tool_loop

        emitted: list[BaseEvent] = []

        # execute_stream raises on the first two calls, succeeds on the third.
        call_count = 0

        async def _flaky_setup(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")

            async def _ok_stream() -> AsyncIterator[Any]:
                # Empty async generator — the unreachable yield is required
                # to keep this function an async generator (no `async def
                # ... -> AsyncIterator: return` shortcut exists in Python).
                return
                yield  # pyright: ignore[reportUnreachable]

            return _ok_stream()

        # selected_config=None lets effective_config_for_call synthesize a
        # minimal ModelConfig — avoids needing a real ModelConfig in this test.
        winning_plan = AttemptPlan(
            provider="anthropic",
            model="claude-sonnet-4-5",
            client=object(),
            thinking_budget_tokens=None,
            reasoning_effort=None,
            selected_config=None,
            attempt=1,
            retry_attempts=3,
            is_fallback=False,
        )

        with (
            patch.object(executor, "CLIENTS", {"anthropic": object()}),
            patch.object(executor, "backend_for_provider", return_value=object()),
            patch.object(executor, "execute_stream", new=_flaky_setup),
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
        ):
            stream = tool_loop.stream_final_response(
                winning_plan=winning_plan,
                prompt="hi",
                max_tokens=64,
                conversation_messages=[{"role": "user", "content": "x"}],
                response_model=None,
                json_mode=False,
                temperature=None,
                stop_seqs=None,
                verbosity=None,
                enable_retry=True,
                retry_attempts=3,
                before_retry_callback=lambda _r: None,
                telemetry=None,
            )
            # Drain (empty) so the wrapper's finally fires for the success attempt.
            async for _chunk in stream:
                pass

        # 3 emissions: attempts 1 & 2 errored, attempt 3 succeeded.
        llm_events = [e for e in emitted if isinstance(e, LLMCallCompletedEvent)]
        assert [e.attempt for e in llm_events] == [1, 2, 3]
        # Final attempt flag: only True on the last retry (attempt 3 of 3).
        assert [e.is_final_attempt for e in llm_events] == [False, False, True]
        # First two errored, last succeeded.
        assert [e.outcome for e in llm_events] == ["error", "error", "success"]


class TestStreamingResponseTokenWriteBack:
    """`StreamingResponseWithMetadata` must accumulate the final-stream's
    output_tokens (reported in usage chunks by OpenAI/Anthropic) into its
    `output_tokens` attribute as the stream drains, so DialecticCompletedEvent
    sees tool-loop totals + final-stream totals — not tool-loop totals alone.
    """

    @pytest.mark.asyncio
    async def test_output_tokens_folds_in_final_stream_usage(self):
        from src.llm.types import (
            HonchoLLMCallStreamChunk,
            StreamingResponseWithMetadata,
        )

        async def _fake_stream() -> Any:
            # Content-only chunks, then a final usage chunk with cumulative
            # output_tokens=137 — matches OpenAI's include_usage pattern.
            yield HonchoLLMCallStreamChunk(content="hel", output_tokens=None)
            yield HonchoLLMCallStreamChunk(content="lo", output_tokens=None)
            yield HonchoLLMCallStreamChunk(content="", is_done=True, output_tokens=137)

        wrapper = StreamingResponseWithMetadata(
            stream=_fake_stream(),
            tool_calls_made=[],
            input_tokens=200,
            output_tokens=50,  # tool-loop running output total
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        # Before drain, the wrapper holds only the tool-loop total.
        assert wrapper.output_tokens == 50

        chunks: list[HonchoLLMCallStreamChunk] = []
        async for chunk in wrapper:
            chunks.append(chunk)

        # After drain, the final-stream's 137 output tokens fold in.
        assert wrapper.output_tokens == 50 + 137
        # And we yielded every chunk to the caller — the wrapper is a
        # passthrough, not a sink.
        assert len(chunks) == 3


class TestStreamingResponseRunHandleClose:
    """When a `langfuse_run_handle` is transferred to the streaming wrapper,
    the wrapper owns it: the accumulated streamed text is stamped as the run
    span's output and the span is closed exactly once when the stream drains.
    The close lives in a `finally`, so an early-exit caller still closes the
    span rather than leaking it.
    """

    class _FakeRunHandle:
        def __init__(self) -> None:
            self.end_calls: list[Any] = []

        def end(self, *, output: Any = None) -> None:
            self.end_calls.append(output)

    @staticmethod
    async def _fake_stream() -> Any:
        from src.llm.types import HonchoLLMCallStreamChunk

        yield HonchoLLMCallStreamChunk(content="hel", output_tokens=None)
        yield HonchoLLMCallStreamChunk(content="lo", output_tokens=None)
        yield HonchoLLMCallStreamChunk(content="", is_done=True, output_tokens=7)

    @pytest.mark.asyncio
    async def test_full_drain_stamps_output_and_closes_once(self):
        from src.llm.types import StreamingResponseWithMetadata

        handle = self._FakeRunHandle()
        wrapper = StreamingResponseWithMetadata(
            stream=self._fake_stream(),
            tool_calls_made=[],
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            langfuse_run_handle=handle,
        )

        async for _ in wrapper:
            pass

        # Closed exactly once, with the concatenated streamed text as output.
        assert handle.end_calls == ["hello"]
        # Ownership released so a second drain can't double-close.
        assert wrapper._langfuse_run_handle is None

    @pytest.mark.asyncio
    async def test_abandoned_stream_still_closes_via_finally(self):
        from src.llm.types import StreamingResponseWithMetadata

        handle = self._FakeRunHandle()
        wrapper = StreamingResponseWithMetadata(
            stream=self._fake_stream(),
            tool_calls_made=[],
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            langfuse_run_handle=handle,
        )

        # Consume one chunk, then abandon the stream. `aclose()` is what the
        # runtime/GC drives when a caller stops iterating early; it throws
        # GeneratorExit at the suspended `yield`, firing the `finally`.
        from collections.abc import AsyncGenerator

        from src.llm.types import HonchoLLMCallStreamChunk

        agen = cast(
            "AsyncGenerator[HonchoLLMCallStreamChunk, None]", wrapper.__aiter__()
        )
        first = await agen.__anext__()
        assert first.content == "hel"
        await agen.aclose()

        # Span closed once with only the text accumulated before abandonment —
        # the span is closed, not leaked.
        assert handle.end_calls == ["hel"]
        assert wrapper._langfuse_run_handle is None
