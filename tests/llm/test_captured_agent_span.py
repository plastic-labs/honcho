# pyright: reportUnusedParameter=false
"""Tests for the run/step lifecycle records in `src/llm/runtime.py`."""

from __future__ import annotations

from typing import Any

import pytest

from src.llm import runtime
from src.llm.types import LLMTelemetryContext


class _RecordingSpanExporter:
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, call: Any) -> None:
        pass

    def export_span(self, span: Any) -> None:
        self.spans.append(span)

    def export_tool_call(self, tool_call: Any) -> None:
        pass


class _CallOnlyExporter:
    def export(self, call: Any) -> None:
        pass


class TestCapturedHandles:
    """Run/step handles report their lifecycle to span-tree exporters."""

    @pytest.fixture
    def exporter(self, monkeypatch: pytest.MonkeyPatch) -> _RecordingSpanExporter:
        from src.llm import capture

        recording = _RecordingSpanExporter()
        monkeypatch.setattr(capture, "_EXPORTERS", [recording])
        return recording

    def test_run_reports_start_and_end_once(
        self, exporter: _RecordingSpanExporter
    ) -> None:
        tele = LLMTelemetryContext(run_id="r1", trace_id="r1", track_name="Agent")
        handle = runtime.start_captured_span("run", tele, input="q")
        assert handle is not None
        handle.end(output="a", is_error=True)
        handle.end(output="ignored")

        start, end = exporter.spans
        assert (start.kind, start.phase, start.input) == ("run", "start", "q")
        assert (end.kind, end.phase, end.output, end.is_error) == (
            "run",
            "end",
            "a",
            True,
        )
        assert end.time_ns >= start.time_ns

    def test_step_reports_its_iteration(self, exporter: _RecordingSpanExporter) -> None:
        tele = LLMTelemetryContext(run_id="r1", iteration=2)
        step = runtime.start_captured_span("step", tele)
        assert step is not None
        step.end()

        assert [(s.kind, s.phase, s.iteration) for s in exporter.spans] == [
            ("step", "start", 2),
            ("step", "end", 2),
        ]

    def test_noop_without_span_identity(self, exporter: _RecordingSpanExporter) -> None:
        assert runtime.start_captured_span("run", LLMTelemetryContext()) is None
        assert exporter.spans == []

    def test_noop_without_a_span_tree_exporter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm import capture

        monkeypatch.setattr(capture, "_EXPORTERS", [_CallOnlyExporter()])
        tele = LLMTelemetryContext(run_id="r1")
        assert runtime.start_captured_span("run", tele) is None
        assert runtime.start_captured_span("step", tele) is None
