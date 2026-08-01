"""Tests for full-fidelity trace events, dedup, and the CloudEvents exporter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from src.llm.backend import CompletionResult
from src.llm.capture import CapturedLLMCall, build_captured_call
from src.telemetry import trace_session
from src.telemetry.events.trace import LLMCallTracedEvent, TraceContentEvent


class TestLLMCallTracedEvent:
    def test_metadata(self):
        assert LLMCallTracedEvent.event_type() == "llm.call.traced"
        assert LLMCallTracedEvent.category() == "trace"
        # Ground-truth — never sampled (the system of record).
        assert LLMCallTracedEvent.volume_class() == "ground_truth"

    def test_resource_id_format(self):
        event = LLMCallTracedEvent(
            span_id="s1",
            iteration=2,
            attempt=1,
            step_seq=3,
            transport="anthropic",
            model="m",
        )
        # {span_id}:{iteration}:{attempt}:{step_seq} — tool_call_seq dropped.
        assert event.get_resource_id() == "s1:2:1:3"

    def test_keeps_default_evt_id(self):
        event = LLMCallTracedEvent(
            span_id="s1",
            iteration=1,
            attempt=1,
            step_seq=1,
            transport="anthropic",
            model="m",
        )
        event_id = event.generate_id()
        assert event_id.startswith("evt_")
        assert len(event_id) == 26


class TestTraceContentEvent:
    def test_metadata(self):
        assert TraceContentEvent.event_type() == "trace.content"
        assert TraceContentEvent.category() == "trace"
        assert TraceContentEvent.volume_class() == "ground_truth"

    def test_resource_id_is_content_hash(self):
        event = TraceContentEvent(content_hash="sha256:abc", role="user", content="hi")
        assert event.get_resource_id() == "sha256:abc"

    def test_generate_id_is_content_addressed_and_timestamp_free(self):
        # Two instances with the same hash but DIFFERENT timestamps must collide
        # on id, so cross-process/retry re-sends dedupe at the transport layer.
        import datetime

        a = TraceContentEvent(
            content_hash="sha256:deadbeefdeadbeefdeadbeef",
            role="user",
            content="hi",
            timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        )
        b = TraceContentEvent(
            content_hash="sha256:deadbeefdeadbeefdeadbeef",
            role="user",
            content="hi",
            timestamp=datetime.datetime(2026, 6, 22, tzinfo=datetime.UTC),
        )
        assert a.generate_id() == b.generate_id()
        assert a.generate_id().startswith("content_")
        # Different content → different id.
        c = TraceContentEvent(content_hash="sha256:other", role="user", content="hi")
        assert c.generate_id() != a.generate_id()


class TestTraceSessionDedup:
    def setup_method(self):
        trace_session.reset()

    def teardown_method(self):
        trace_session.reset()

    def test_first_emit_true_repeat_false(self):
        assert trace_session.mark_emitted("run-1", "h1") is True
        assert trace_session.mark_emitted("run-1", "h1") is False  # already shipped
        assert trace_session.mark_emitted("run-1", "h2") is True  # new hash

    def test_runs_are_independent(self):
        assert trace_session.mark_emitted("run-1", "h1") is True
        assert trace_session.mark_emitted("run-2", "h1") is True  # different run

    def test_lru_evicts_least_recently_used_run(self, monkeypatch: pytest.MonkeyPatch):
        # Shrink the window so eviction is testable without _MAX_RUNS runs.
        monkeypatch.setattr(trace_session, "_MAX_RUNS", 2)
        trace_session.mark_emitted("run-1", "h1")
        trace_session.mark_emitted("run-2", "h1")
        # Touch run-1 so run-2 becomes the least-recently-used run.
        trace_session.mark_emitted("run-1", "h2")
        # A third run evicts the LRU run (run-2), keeping run-1.
        trace_session.mark_emitted("run-3", "h1")
        # run-1 is still tracked → its already-shipped hash stays deduped.
        assert trace_session.mark_emitted("run-1", "h1") is False
        # run-2 was evicted → its hash ships again as if a fresh run.
        assert trace_session.mark_emitted("run-2", "h1") is True


class _FakeTraceEmitter:
    """Stand-in for the trace emitter that records emitted events."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def emit(self, event: object) -> None:
        self.events.append(event)


@pytest.fixture
def trace_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeTraceEmitter]:
    """Enable payload tracing and route emit_trace at a fake emitter."""
    from src.config import settings
    from src.telemetry import emitter as emitter_mod

    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PAYLOADS_ENABLED", True)
    fake = _FakeTraceEmitter()
    monkeypatch.setattr(emitter_mod, "_trace_emitter", fake)
    trace_session.reset()
    yield fake
    trace_session.reset()


def _captured(
    messages: list[dict[str, Any]], *, content: str = "answer", run: str = "r1"
) -> CapturedLLMCall:
    from src.llm.types import LLMTelemetryContext

    return build_captured_call(
        telemetry=LLMTelemetryContext(
            workspace_name="ws",
            call_purpose="dialectic.answer",
            parent_category="dialectic",
            run_id=run,
            trace_id=run,
            span_id=run,
            iteration=1,
            step_seq=1,
        ),
        transport="anthropic",
        provider_label=None,
        model="claude-x",
        messages=messages,
        tools=None,
        tool_choice=None,
        result=CompletionResult(content=content, finish_reason="stop"),
        attempt=1,
        was_fallback=False,
        was_stream=False,
        finish_reason="stop",
    )


class TestTraceExporter:
    def test_refs_match_emitted_content(self, trace_on: _FakeTraceEmitter):
        from src.telemetry.trace_exporter import TraceExporter

        call = _captured([{"role": "user", "content": "q"}])
        TraceExporter().export(call)

        traced = [e for e in trace_on.events if isinstance(e, LLMCallTracedEvent)]
        contents = [e for e in trace_on.events if isinstance(e, TraceContentEvent)]
        assert len(traced) == 1
        # input message + output content → two content events.
        emitted_hashes = {c.content_hash for c in contents}
        # Every input ref points at an emitted trace.content.
        for ref in traced[0].input_message_refs:
            assert ref in emitted_hashes
        assert traced[0].output_content_ref in emitted_hashes

    def test_dedup_across_iterations(self, trace_on: _FakeTraceEmitter):
        from src.telemetry.trace_exporter import TraceExporter

        exporter = TraceExporter()
        shared = {"role": "user", "content": "system context"}
        # Iteration 1: messages [shared]; iteration 2: [shared, follow-up].
        exporter.export(_captured([shared]))
        before = sum(isinstance(e, TraceContentEvent) for e in trace_on.events)
        exporter.export(_captured([shared, {"role": "user", "content": "more"}]))
        after = sum(isinstance(e, TraceContentEvent) for e in trace_on.events)
        # `shared` already shipped this run → only the new message (+ output if
        # not already seen) emit again; `shared` is NOT re-emitted.
        shared_hash = None
        for e in trace_on.events:
            if isinstance(e, TraceContentEvent) and e.content == "system context":
                shared_hash = e.content_hash
        emitted_shared = [
            e
            for e in trace_on.events
            if isinstance(e, TraceContentEvent) and e.content_hash == shared_hash
        ]
        assert len(emitted_shared) == 1  # shipped once across both iterations
        assert after > before  # the new message did ship

    def test_purpose_allowlist_filters(
        self, trace_on: _FakeTraceEmitter, monkeypatch: pytest.MonkeyPatch
    ):
        from src.config import settings
        from src.telemetry.trace_exporter import TraceExporter

        monkeypatch.setattr(settings.TELEMETRY, "TRACE_PURPOSES", ["summary.short"])
        TraceExporter().export(_captured([{"role": "user", "content": "q"}]))
        # call_purpose is dialectic.answer, not in the allowlist → nothing emits.
        assert trace_on.events == []
