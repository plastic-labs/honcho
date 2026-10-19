# pyright: reportPrivateUsage=false
"""`LangfuseExporter` against the real Langfuse + OTEL SDKs.

The unit tests use a fake span, so they can't catch an SDK upgrade that renames
the private span fields the exporter writes (`_start_time`, `_parent`). These
assert on what the real SDK actually exports.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from src.config import settings
from src.llm.backend import CompletionResult, ToolCallResult
from src.llm.capture import CapturedToolCall, build_captured_call
from src.llm.types import LLMTelemetryContext
from src.telemetry import langfuse_session
from src.telemetry.langfuse_exporter import LangfuseExporter


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    # The SDK caches its span pipeline per public key, so each test needs its own.
    public_key = f"pk-lf-{uuid.uuid4().hex}"
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", public_key)
    span_exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=public_key,
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        tracer_provider=TracerProvider(),
        span_exporter=span_exporter,
    )
    import langfuse

    monkeypatch.setattr(langfuse, "get_client", lambda: client)
    langfuse_session.reset()
    yield span_exporter
    langfuse_session.reset()
    client.shutdown()


def _export_dialectic_run(duration_ms: float) -> None:
    telemetry = LLMTelemetryContext(
        workspace_name="ws",
        call_purpose="dialectic.answer",
        parent_category="dialectic",
        agent_type="dialectic",
        run_id="r1",
        trace_id="r1",
        span_id="r1",
        track_name="Dialectic Agent",
        iteration=1,
    )
    call = build_captured_call(
        telemetry=telemetry,
        transport="anthropic",
        provider_label=None,
        model="claude-x",
        messages=[{"role": "user", "content": "q"}],
        tools=None,
        tool_choice=None,
        result=CompletionResult(
            content="",
            input_tokens=10,
            output_tokens=5,
            finish_reason="tool_use",
            tool_calls=[ToolCallResult(id="tc-0", name="search_memory", input={})],
        ),
        attempt=1,
        was_fallback=False,
        was_stream=False,
        finish_reason="tool_use",
        duration_ms=duration_ms,
    )
    exporter = LangfuseExporter()
    exporter.export(call)
    exporter.export_tool_call(
        CapturedToolCall(
            run_id="r1",
            agent_type="dialectic",
            workspace_name="ws",
            iteration=1,
            tool_call_seq=0,
            tool_call_id="tc-0",
            name="search_memory",
            input={},
            output="result",
            is_error=False,
            duration_ms=duration_ms,
        )
    )


def _finished(span_exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    from langfuse import get_client

    get_client().flush()
    return {s.name: s for s in span_exporter.get_finished_spans()}


def _duration_ms(span: ReadableSpan) -> float:
    assert span.start_time is not None and span.end_time is not None
    return (span.end_time - span.start_time) / 1_000_000


def test_generation_and_tool_starts_are_backdated(exported: InMemorySpanExporter):
    _export_dialectic_run(duration_ms=2_000)
    spans = _finished(exported)

    assert abs(_duration_ms(spans["Dialectic Agent generation"]) - 2_000) < 50
    assert abs(_duration_ms(spans["search_memory"]) - 2_000) < 50


def test_trace_has_exactly_one_parentless_root(exported: InMemorySpanExporter):
    _export_dialectic_run(duration_ms=10)
    spans = _finished(exported)

    roots = [name for name, s in spans.items() if s.parent is None]
    assert roots == ["Dialectic Agent"]
    ids = {s.context.span_id for s in spans.values() if s.context is not None}
    for s in spans.values():
        if s.parent is not None:
            assert s.parent.span_id in ids
    assert spans["Dialectic Agent"].attributes is not None
    assert spans["Dialectic Agent"].attributes["langfuse.observation.type"] == "agent"
