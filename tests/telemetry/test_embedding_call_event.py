# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""tests for EmbeddingCallCompletedEvent emission.

Targets:
- New `EmbeddingCallCompletedEvent` (embedding.call.completed) at schema v1
  with high_volume sampling.
- `EmbeddingCallPurpose` closed enum.
- The `embedding_call_purpose` context manager round-trips the slug onto
  the event via the ContextVar, without changing call signatures.
- `_emit_embedding_call` wrapper emits on success AND on exception, and
  propagates the underlying error unchanged.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.embedding_client import _emit_embedding_call
from src.telemetry.events import (
    BaseEvent,
    EmbeddingCallCompletedEvent,
    EmbeddingCallPurpose,
)
from src.utils.types import (
    embedding_call_purpose,
    get_embedding_call_purpose,
)


class TestEventShape:
    def test_event_type_and_version(self):
        assert EmbeddingCallCompletedEvent.event_type() == "embedding.call.completed"
        assert EmbeddingCallCompletedEvent.schema_version() == 1
        assert EmbeddingCallCompletedEvent.category() == "llm"

    def test_volume_class_is_high_volume(self):
        """event participates in HIGH_VOLUME_SAMPLE_RATE alongside
        llm.call.completed — search-heavy paths can flood the buffer."""
        assert EmbeddingCallCompletedEvent.volume_class() == "high_volume"

    def test_resource_id_disambiguates(self):
        ev_a = EmbeddingCallCompletedEvent(
            provider="openai",
            model="text-embedding-3-small",
            input_count=5,
            duration_ms=10.0,
            outcome="success",
            call_purpose=EmbeddingCallPurpose.SEARCH_MEMORY,
        )
        ev_b = EmbeddingCallCompletedEvent(
            provider="openai",
            model="text-embedding-3-small",
            input_count=5,
            duration_ms=10.0,
            outcome="success",
            call_purpose=EmbeddingCallPurpose.SEARCH_MESSAGES,
        )
        # Different purpose → different resource id.
        assert ev_a.get_resource_id() != ev_b.get_resource_id()


class TestEmbeddingCallPurposeEnum:
    def test_known_values(self):
        # The closed taxonomy. Adding a value here requires a coordinated
        # update with downstream analytics that filter on call_purpose.
        assert EmbeddingCallPurpose.SEARCH_MEMORY.value == "search_memory"
        assert EmbeddingCallPurpose.SEARCH_MESSAGES.value == "search_messages"
        assert EmbeddingCallPurpose.CREATE_OBSERVATIONS.value == "create_observations"
        assert EmbeddingCallPurpose.VECTOR_SYNC.value == "vector_sync"
        assert EmbeddingCallPurpose.SUMMARY.value == "summary"
        assert EmbeddingCallPurpose.MESSAGE_CREATE.value == "message_create"


class TestContextManager:
    def test_sets_and_clears(self):
        assert get_embedding_call_purpose() is None
        with embedding_call_purpose("search_memory"):
            assert get_embedding_call_purpose() == "search_memory"
        # ContextVar must reset on exit.
        assert get_embedding_call_purpose() is None

    def test_nested_context_managers_restore_outer(self):
        """Nested usage shouldn't lose the outer purpose on inner-exit."""
        with embedding_call_purpose("search_memory"):
            assert get_embedding_call_purpose() == "search_memory"
            with embedding_call_purpose("create_observations"):
                assert get_embedding_call_purpose() == "create_observations"
            # After inner exits, outer purpose must be restored — not None.
            assert get_embedding_call_purpose() == "search_memory"

    def test_exception_in_block_still_resets(self):
        try:
            with embedding_call_purpose("search_memory"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert get_embedding_call_purpose() is None


class TestEmitEmbeddingCallWrapper:
    @pytest.mark.asyncio
    async def test_success_emits_event_with_purpose(self):
        emitted: list[BaseEvent] = []

        async def _fake_call() -> list[float]:
            return [0.1, 0.2, 0.3]

        with (
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            embedding_call_purpose("search_memory"),
        ):
            result = await _emit_embedding_call(
                provider="openai",
                model="text-embedding-3-small",
                texts=["query"],
                input_tokens_estimate=3,
                fn=_fake_call,
            )

        assert result == [0.1, 0.2, 0.3]
        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, EmbeddingCallCompletedEvent)
        assert ev.outcome == "success"
        assert ev.provider == "openai"
        assert ev.model == "text-embedding-3-small"
        assert ev.input_count == 1
        assert ev.input_tokens_estimate == 3
        assert ev.call_purpose == EmbeddingCallPurpose.SEARCH_MEMORY
        assert ev.error_class is None

    @pytest.mark.asyncio
    async def test_exception_emits_error_event_and_propagates(self):
        emitted: list[BaseEvent] = []

        async def _boom() -> list[float]:
            raise RuntimeError("provider down")

        with (
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            pytest.raises(RuntimeError, match="provider down"),
        ):
            await _emit_embedding_call(
                provider="gemini",
                model="text-embedding-005",
                texts=["a", "b"],
                input_tokens_estimate=10,
                fn=_boom,
            )

        # Event emitted on the error path too (try/finally).
        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, EmbeddingCallCompletedEvent)
        assert ev.outcome == "error"
        assert ev.error_class == "RuntimeError"
        assert ev.input_count == 2

    @pytest.mark.asyncio
    async def test_cancellation_emits_cancelled_outcome(self):
        """asyncio.CancelledError surfaces as outcome='cancelled', not 'error'
        — client disconnects / shutdowns must not pollute error rates."""
        import asyncio

        emitted: list[BaseEvent] = []

        async def _cancel() -> list[float]:
            raise asyncio.CancelledError()

        with (
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await _emit_embedding_call(
                provider="openai",
                model="text-embedding-3-small",
                texts=["a"],
                input_tokens_estimate=4,
                fn=_cancel,
            )

        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, EmbeddingCallCompletedEvent)
        assert ev.outcome == "cancelled"
        assert ev.error_class == "CancelledError"

    @pytest.mark.asyncio
    async def test_unknown_purpose_drops_to_none(self):
        """Unknown call_purpose ContextVar strings shouldn't crash the
        emitter — they fall through to call_purpose=None on the event."""
        emitted: list[BaseEvent] = []

        async def _fake_call() -> list[float]:
            return [0.1]

        with (
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            embedding_call_purpose("not.a.real.purpose"),
        ):
            await _emit_embedding_call(
                provider="openai",
                model="x",
                texts=["q"],
                input_tokens_estimate=1,
                fn=_fake_call,
            )

        ev = emitted[0]
        assert isinstance(ev, EmbeddingCallCompletedEvent)
        assert ev.call_purpose is None

    @pytest.mark.asyncio
    async def test_telemetry_failure_swallowed(self):
        """Telemetry path must not propagate exceptions into the caller."""

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("emitter wedged")

        async def _fake_call() -> str:
            return "ok"

        with patch("src.telemetry.events.emit", side_effect=explode):
            # Must NOT raise from the wrapper even though emit() throws.
            result = await _emit_embedding_call(
                provider="openai",
                model="x",
                texts=["q"],
                input_tokens_estimate=1,
                fn=_fake_call,
            )
        assert result == "ok"


class TestIsFinalAttempt:
    """`is_final_attempt` must reflect real retry state — one-shot callers
    report True (no further attempt), retry-loop callers thread the real
    index. Previously hardcoded to False; that conflated one-shot success,
    mid-retry failure, and exhausted retry on dashboards.
    """

    @pytest.mark.asyncio
    async def test_oneshot_default_is_true(self):
        emitted: list[BaseEvent] = []

        async def _fake_call() -> list[float]:
            return [0.1]

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            await _emit_embedding_call(
                provider="openai",
                model="x",
                texts=["q"],
                input_tokens_estimate=1,
                fn=_fake_call,
            )

        ev = emitted[0]
        assert isinstance(ev, EmbeddingCallCompletedEvent)
        assert ev.is_final_attempt is True

    @pytest.mark.asyncio
    async def test_mid_retry_is_false(self):
        emitted: list[BaseEvent] = []

        async def _boom() -> list[float]:
            raise RuntimeError("transient")

        with (
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            pytest.raises(RuntimeError),
        ):
            await _emit_embedding_call(
                provider="openai",
                model="x",
                texts=["q"],
                input_tokens_estimate=1,
                fn=_boom,
                is_final_attempt=False,
            )

        ev = emitted[0]
        assert isinstance(ev, EmbeddingCallCompletedEvent)
        assert ev.is_final_attempt is False

    @pytest.mark.asyncio
    async def test_exhausted_retry_is_true(self):
        emitted: list[BaseEvent] = []

        async def _boom() -> list[float]:
            raise RuntimeError("permanent")

        with (
            patch(
                "src.telemetry.events.emit",
                side_effect=lambda event: emitted.append(event),
            ),
            pytest.raises(RuntimeError),
        ):
            await _emit_embedding_call(
                provider="openai",
                model="x",
                texts=["q"],
                input_tokens_estimate=1,
                fn=_boom,
                is_final_attempt=True,
            )

        ev = emitted[0]
        assert isinstance(ev, EmbeddingCallCompletedEvent)
        assert ev.is_final_attempt is True
        assert ev.outcome == "error"
