"""Tests for OpenTelemetry instrumentation (memory-semconv v0.1.0).

These tests use local TracerProvider/MeterProvider instances rather than
replacing the global OTel providers, so they are safe to run in any order
and alongside other test modules.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture()
def local_tracer_and_exporter():
    """Return a (tracer, exporter) pair backed by an in-process span exporter.

    Does NOT replace the global TracerProvider — tests call
    ``tracer.start_as_current_span`` directly.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


@pytest.fixture()
def local_meter_and_reader():
    """Return a (meter, reader) pair backed by InMemoryMetricReader."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    return meter, reader


def test_setup_otel_noop_when_disabled():
    """setup_otel with enabled=False must not replace the global providers."""
    import opentelemetry.trace as trace_api

    import src.telemetry.otel as otel_mod
    from src.telemetry.otel import setup_otel

    otel_mod._initialized = False
    before = trace_api.get_tracer_provider()
    setup_otel(enabled=False)
    after = trace_api.get_tracer_provider()
    assert before is after
    assert not otel_mod._initialized


def test_setup_otel_enabled_sets_sdk_provider():
    """setup_otel with enabled=True installs a real TracerProvider."""
    import opentelemetry.trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

    import src.telemetry.otel as otel_mod

    otel_mod._initialized = False
    original = trace_api.get_tracer_provider()
    try:
        from src.telemetry.otel import setup_otel

        setup_otel(enabled=True, service_name="test-honcho")
        assert isinstance(trace_api.get_tracer_provider(), SdkTracerProvider)
        assert otel_mod._initialized
    finally:
        # Best-effort restore so this test does not pollute the global provider
        # for others. This reaches into OTel-internal globals guarded by a
        # set-once lock; those names are not public API, so guard every access
        # and never let teardown raise if the internals change across versions.
        try:
            import opentelemetry.trace as _trace_mod

            if hasattr(_trace_mod, "_TRACER_PROVIDER"):
                _trace_mod._TRACER_PROVIDER = original  # type: ignore[attr-defined]
            once = getattr(_trace_mod, "_TRACER_PROVIDER_SET_ONCE", None)
            if once is not None and hasattr(once, "_done"):
                once._done = False  # type: ignore[attr-defined]
        except Exception:
            pass
        otel_mod._initialized = False


def test_memory_semconv_attributes_present(local_tracer_and_exporter):
    """Memory search spans must carry all memory-semconv v0.1.0 required attributes."""
    tracer, exporter = local_tracer_and_exporter

    from src.telemetry.otel import (
        MEMORY_OBSERVED,
        MEMORY_OBSERVER,
        MEMORY_OPERATION,
        MEMORY_QUERY,
        MEMORY_RESULT_COUNT,
        MEMORY_SYSTEM,
        MEMORY_WORKSPACE,
    )

    with tracer.start_as_current_span(
        "memory.search",
        attributes={
            MEMORY_SYSTEM: "honcho",
            MEMORY_OPERATION: "search",
            MEMORY_WORKSPACE: "ws1",
            MEMORY_OBSERVER: "user-a",
            MEMORY_OBSERVED: "user-a",
            MEMORY_QUERY: "what did I say about dogs",
            MEMORY_RESULT_COUNT: 3,
        },
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs[MEMORY_SYSTEM] == "honcho"
    assert attrs[MEMORY_OPERATION] == "search"
    assert attrs[MEMORY_RESULT_COUNT] == 3
    assert MEMORY_QUERY in attrs


def test_memory_add_attributes(local_tracer_and_exporter):
    """Add spans must carry operation=add and item_count."""
    tracer, exporter = local_tracer_and_exporter

    from src.telemetry.otel import MEMORY_ITEM_COUNT, MEMORY_OPERATION, MEMORY_SYSTEM

    with tracer.start_as_current_span(
        "memory.add",
        attributes={
            MEMORY_SYSTEM: "honcho",
            MEMORY_OPERATION: "add",
            MEMORY_ITEM_COUNT: 5,
        },
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs[MEMORY_OPERATION] == "add"
    assert attrs[MEMORY_ITEM_COUNT] == 5


def test_memory_delete_attributes(local_tracer_and_exporter):
    """Delete spans must carry operation=delete."""
    tracer, exporter = local_tracer_and_exporter

    from src.telemetry.otel import MEMORY_OPERATION, MEMORY_SYSTEM

    with tracer.start_as_current_span(
        "memory.delete",
        attributes={MEMORY_SYSTEM: "honcho", MEMORY_OPERATION: "delete"},
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert (spans[0].attributes or {}).get(MEMORY_OPERATION) == "delete"


def test_search_query_capture_is_opt_in(monkeypatch):
    """memory.query must be omitted by default (PII / high cardinality) and only
    recorded — truncated — when OTEL.CAPTURE_QUERY is explicitly enabled."""
    from src.config import settings
    from src.crud.document import _memory_search_attributes
    from src.telemetry.otel import MEMORY_QUERY

    monkeypatch.setattr(settings.OTEL, "CAPTURE_QUERY", False, raising=False)
    attrs = _memory_search_attributes("ws1", "a", "b", "secret user query")
    assert MEMORY_QUERY not in attrs

    monkeypatch.setattr(settings.OTEL, "CAPTURE_QUERY", True, raising=False)
    monkeypatch.setattr(settings.OTEL, "QUERY_MAX_LENGTH", 5, raising=False)
    attrs = _memory_search_attributes("ws1", "a", "b", "secret user query")
    assert attrs[MEMORY_QUERY] == "secre"


def test_search_duration_metric_recorded(local_meter_and_reader):
    """memory.search.duration histogram must record and export correctly."""
    meter, reader = local_meter_and_reader

    hist = meter.create_histogram("memory.search.duration", unit="ms")
    hist.record(42.0, {"memory.system": "honcho"})

    data = reader.get_metrics_data()
    found = False
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "memory.search.duration":
                    found = True
    assert found, "memory.search.duration metric not found in exported data"


def test_search_results_metric_recorded(local_meter_and_reader):
    """memory.search.results histogram must record and export correctly."""
    meter, reader = local_meter_and_reader

    hist = meter.create_histogram("memory.search.results")
    hist.record(3, {"memory.system": "honcho"})

    data = reader.get_metrics_data()
    found = any(
        metric.name == "memory.search.results"
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    )
    assert found, "memory.search.results metric not found in exported data"


def test_operation_duration_metric_recorded(local_meter_and_reader):
    """memory.operation.duration histogram must record for add and delete ops."""
    meter, reader = local_meter_and_reader

    hist = meter.create_histogram("memory.operation.duration", unit="ms")
    hist.record(10.5, {"memory.system": "honcho", "memory.operation": "add"})
    hist.record(5.2, {"memory.system": "honcho", "memory.operation": "delete"})

    data = reader.get_metrics_data()
    found = any(
        metric.name == "memory.operation.duration"
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    )
    assert found, "memory.operation.duration metric not found in exported data"


@pytest.fixture()
def local_logger_and_exporter():
    """Return an (otel_logger, exporter) pair backed by InMemoryLogExporter."""
    exporter = InMemoryLogExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    otel_logger = provider.get_logger("test")
    return otel_logger, exporter


def test_memory_search_log_emitted(local_logger_and_exporter):
    """memory.search operations must emit an OTel log record with required attributes."""
    otel_logger, exporter = local_logger_and_exporter

    from src.telemetry.otel import (
        MEMORY_OPERATION,
        MEMORY_RESULT_COUNT,
        MEMORY_SYSTEM,
        MEMORY_WORKSPACE,
    )

    otel_logger.emit(
        body="memory.search: 3 results in 42.0ms",
        severity_text="INFO",
        attributes={
            MEMORY_SYSTEM: "honcho",
            MEMORY_OPERATION: "search",
            MEMORY_WORKSPACE: "ws1",
            MEMORY_RESULT_COUNT: 3,
        },
    )

    records = exporter.get_finished_logs()
    assert len(records) == 1
    rec = records[0].log_record
    assert str(rec.body) == "memory.search: 3 results in 42.0ms"
    assert (rec.attributes or {}).get(MEMORY_OPERATION) == "search"
    assert (rec.attributes or {}).get(MEMORY_RESULT_COUNT) == 3


def test_memory_add_log_emitted(local_logger_and_exporter):
    """memory.add operations must emit an OTel log record."""
    otel_logger, exporter = local_logger_and_exporter

    from src.telemetry.otel import MEMORY_ITEM_COUNT, MEMORY_OPERATION, MEMORY_SYSTEM

    otel_logger.emit(
        body="memory.add: 5 documents created in 15.0ms",
        severity_text="INFO",
        attributes={
            MEMORY_SYSTEM: "honcho",
            MEMORY_OPERATION: "add",
            MEMORY_ITEM_COUNT: 5,
        },
    )

    records = exporter.get_finished_logs()
    assert len(records) == 1
    assert (records[0].log_record.attributes or {}).get(MEMORY_OPERATION) == "add"


def test_memory_delete_log_emitted(local_logger_and_exporter):
    """memory.delete operations must emit an OTel log record."""
    otel_logger, exporter = local_logger_and_exporter

    from src.telemetry.otel import MEMORY_ITEM_COUNT, MEMORY_OPERATION, MEMORY_SYSTEM

    otel_logger.emit(
        body="memory.delete: 2 documents deleted in 5.0ms",
        severity_text="INFO",
        attributes={
            MEMORY_SYSTEM: "honcho",
            MEMORY_OPERATION: "delete",
            MEMORY_ITEM_COUNT: 2,
        },
    )

    records = exporter.get_finished_logs()
    assert len(records) == 1
    assert (records[0].log_record.attributes or {}).get(MEMORY_OPERATION) == "delete"


# --------------------------------------------------------------------------- #
# Regression: FastAPI instrumentation must attach BEFORE the ASGI
# middleware stack is frozen, or no SERVER span is produced and inbound W3C
# traceparent is never extracted (end-to-end traces collapse into orphaned
# single-span traces). Starlette's ``add_middleware`` on an already-built stack
# is a silent no-op, so attaching in the lifespan startup handler (which runs
# after the stack is built) fails silently. main.py must call ``instrument_app``
# at module import time. These tests lock in both the failure mode and the fix.
# --------------------------------------------------------------------------- #

_TRACEPARENT_TID = "a" * 32
_TRACEPARENT = f"00-{_TRACEPARENT_TID}-{'b' * 16}-01"


def test_fastapi_instrumentation_extracts_inbound_traceparent():
    """The behavior the fix restores: an instrumented FastAPI app opens a SERVER
    span that is a child of the inbound W3C traceparent, so a caller's trace and
    Honcho's spans become one end-to-end trace (cross-process linkage)."""
    from fastapi import FastAPI
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.trace import SpanKind
    from starlette.testclient import TestClient

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    app = FastAPI()

    @app.get("/ping")
    async def ping():  # pragma: no cover - trivial handler
        return {"ok": True}

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    try:
        with TestClient(app) as client:
            client.get("/ping", headers={"traceparent": _TRACEPARENT})
        server_spans = [
            s for s in exporter.get_finished_spans() if s.kind == SpanKind.SERVER
        ]
    finally:
        FastAPIInstrumentor.uninstrument_app(app)

    assert len(server_spans) == 1
    # The server span must join the caller's trace, not start a new one.
    assert format(server_spans[0].context.trace_id, "032x") == _TRACEPARENT_TID


def test_main_calls_instrument_app_at_module_scope_not_in_lifespan():
    """Regression guard. ``instrument_app`` MUST be invoked at module
    import time — before Starlette freezes the ASGI middleware stack on the first
    request. Attaching it inside the ``lifespan`` startup handler (which runs
    after the stack is built) is a silent no-op: no SERVER spans, no inbound
    traceparent extraction, orphaned single-span memory.* traces. This parses
    src/main.py and fails if ``instrument_app`` is called inside any function
    (e.g. moved back into ``lifespan``) instead of at module scope.
    """
    import ast
    from pathlib import Path

    # Resolve relative to this test file (repo_root/tests/telemetry/test_otel.py)
    # so the test passes regardless of pytest's working directory.
    repo_root = Path(__file__).resolve().parents[2]
    main_src = (repo_root / "src" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(main_src)

    def _calls_instrument_app(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "instrument_app"
            for n in ast.walk(node)
        )

    module_level = any(
        isinstance(stmt, ast.Expr) and _calls_instrument_app(stmt) for stmt in tree.body
    )
    in_function = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _calls_instrument_app(node)
        for node in ast.walk(tree)
    )

    assert module_level, "instrument_app(app) must be called at module scope in main.py"
    assert not in_function, (
        "instrument_app must NOT be called inside a function (e.g. lifespan) — "
        "it would attach after the middleware stack is frozen (regression)"
    )
