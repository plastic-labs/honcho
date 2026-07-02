# pyright: reportPrivateUsage=false, reportUnannotatedClassAttribute=false, reportUnusedFunction=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""Tests for embedding calls joining the trace stream.

`_publish_embedding_event` emits an `EmbeddingCallTracedEvent` (gated on
TRACE_PAYLOADS_ENABLED) in addition to the metrics-grade completed event, carrying the
span-tree correlation from the embedding ContextVars so an embedding made inside
an agent run nests under that run's trace.
"""

from __future__ import annotations

import pytest

import src.telemetry.events as events_mod
from src.config import settings
from src.embedding_client import _publish_embedding_event
from src.telemetry.events.trace import EmbeddingCallTracedEvent
from src.utils.types import embedding_call_purpose


@pytest.fixture
def capture_emits(monkeypatch: pytest.MonkeyPatch):
    """Capture emit()/emit_trace() without a live emitter."""
    traced: list[object] = []
    monkeypatch.setattr(events_mod, "emit", lambda _e: None)
    monkeypatch.setattr(events_mod, "emit_trace", lambda e: traced.append(e))
    return traced


def test_embedding_traced_event_carries_correlation(
    capture_emits: list[object], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PAYLOADS_ENABLED", True)
    with embedding_call_purpose(
        "dialectic.prefetch",
        workspace_name="ws",
        run_id="run-1",
        parent_category="dialectic",
        session_id="sess-1",
    ):
        _publish_embedding_event(
            provider="openai",
            model="text-embedding-3",
            input_count=1,
            input_tokens_estimate=7,
            duration_ms=1.0,
            outcome="success",
            error=None,
            is_final_attempt=True,
        )

    assert len(capture_emits) == 1
    ev = capture_emits[0]
    assert isinstance(ev, EmbeddingCallTracedEvent)
    # The embedding gets its own span under the run: trace_id/parent are the
    # run_id, span_id is a fresh id so sibling embeddings don't collide.
    assert ev.trace_id == "run-1"
    assert ev.parent_span_id == "run-1"
    assert ev.span_id and ev.span_id != "run-1"
    assert ev.session_id == "sess-1"
    assert ev.call_purpose == "dialectic.prefetch"
    assert ev.parent_category == "dialectic"
    assert ev.provider == "openai" and ev.model == "text-embedding-3"
    assert ev.provider_input_tokens == 7
    assert ev.provider_output_tokens == 0
    assert ev.input_count == 1


def test_no_trace_event_when_payloads_off(
    capture_emits: list[object], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PAYLOADS_ENABLED", False)
    with embedding_call_purpose("dialectic.prefetch", run_id="run-1"):
        _publish_embedding_event(
            provider="openai",
            model="m",
            input_count=2,
            input_tokens_estimate=3,
            duration_ms=1.0,
            outcome="success",
            error=None,
            is_final_attempt=True,
        )
    assert capture_emits == []


def test_sessionless_embedding_has_no_session(
    capture_emits: list[object], monkeypatch: pytest.MonkeyPatch
):
    # A deriver/reconciler embedding (no session scope) traces with session None.
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PAYLOADS_ENABLED", True)
    with embedding_call_purpose(
        "deriver", workspace_name="ws", parent_category="deriver"
    ):
        _publish_embedding_event(
            provider="gemini",
            model="emb",
            input_count=5,
            input_tokens_estimate=20,
            duration_ms=2.0,
            outcome="success",
            error=None,
            is_final_attempt=True,
        )
    assert len(capture_emits) == 1
    ev = capture_emits[0]
    assert isinstance(ev, EmbeddingCallTracedEvent)
    assert ev.session_id is None
    # No run_id → the span self-roots (trace_id == span_id) with no parent.
    assert ev.parent_span_id is None
    assert ev.span_id and ev.trace_id == ev.span_id
