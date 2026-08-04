"""OpenTelemetry setup for Honcho — memory-semconv v0.1.0.

Opt-in via OTEL_ENABLED=true. When disabled, all tracer/meter/logger calls
become no-ops through the OTel API's default no-op providers.

Three signals are emitted:
- **Traces**: memory.search / memory.add / memory.delete spans
- **Metrics**: memory.search.duration, memory.search.results, memory.operation.duration
- **Logs**: structured OTel log records on each memory operation
"""

from __future__ import annotations

import logging

from opentelemetry import _logs as otel_logs
from opentelemetry import metrics, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

# memory-semconv v0.1.0 attribute names
MEMORY_SYSTEM = "memory.system"
MEMORY_OPERATION = "memory.operation"
MEMORY_QUERY = "memory.query"
MEMORY_RESULT_COUNT = "memory.result_count"
MEMORY_ITEM_COUNT = "memory.item_count"
MEMORY_VECTOR_COUNT = "memory.vector_count"
MEMORY_WORKSPACE = "memory.workspace"
MEMORY_OBSERVER = "memory.observer"
MEMORY_OBSERVED = "memory.observed"

_initialized = False

# Providers created by setup_otel, retained so shutdown_otel() can flush and
# close them on process exit (BatchSpan/BatchLogRecord processors buffer).
_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None
_logger_provider: LoggerProvider | None = None


def setup_otel(
    *,
    enabled: bool = False,
    service_name: str = "honcho",
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize OTel TracerProvider, MeterProvider, and LoggerProvider.

    When *enabled* is False this is a no-op — the OTel API's default no-op
    providers remain active so all instrumentation calls are safe zero-cost stubs.

    Args:
        enabled: Master toggle; False leaves OTel in no-op mode.
        service_name: Value for the ``service.name`` resource attribute.
        otlp_endpoint: OTLP gRPC endpoint (e.g. ``http://localhost:4317``).
            When None, falls back to the ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.
            If neither is set and enabled is True, console exporters are used.
    """
    global _initialized, _tracer_provider, _meter_provider, _logger_provider
    if not enabled or _initialized:
        return

    resource = Resource.create({"service.name": service_name})

    # --- Tracer ---
    tracer_provider = TracerProvider(resource=resource)
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
            logger.info("OTel traces → OTLP gRPC at %s", otlp_endpoint)
        except Exception:
            logger.warning("OTLP span exporter unavailable; falling back to console")
            tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
    else:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OTel traces → console (no OTLP endpoint configured)")

    trace.set_tracer_provider(tracer_provider)
    _tracer_provider = tracer_provider

    # --- Meter ---
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otlp_endpoint)
            )
            logger.info("OTel metrics → OTLP gRPC at %s", otlp_endpoint)
        except Exception:
            logger.warning("OTLP metric exporter unavailable; falling back to console")
            reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
    else:
        reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        logger.info("OTel metrics → console (no OTLP endpoint configured)")

    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)
    _meter_provider = meter_provider

    # --- Logger ---
    log_provider = LoggerProvider(resource=resource)
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter,
            )

            log_provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=otlp_endpoint))
            )
            logger.info("OTel logs → OTLP gRPC at %s", otlp_endpoint)
        except Exception:
            logger.warning("OTLP log exporter unavailable; falling back to console")
            log_provider.add_log_record_processor(
                BatchLogRecordProcessor(ConsoleLogExporter())
            )
    else:
        log_provider.add_log_record_processor(
            BatchLogRecordProcessor(ConsoleLogExporter())
        )
        logger.info("OTel logs → console (no OTLP endpoint configured)")

    otel_logs.set_logger_provider(log_provider)
    _logger_provider = log_provider

    # --- Auto-instrumentation for outbound HTTP ---
    # httpx underpins the Anthropic/OpenAI/embedding SDKs, so instrumenting it
    # turns each LLM/embedding call into a CLIENT span. Combined with the FastAPI
    # server span (see instrument_app), the memory.* spans nest into a single
    # end-to-end trace per request instead of appearing as orphaned single-span
    # traces.
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.info("OTel httpx client instrumentation enabled")
    except Exception:
        logger.warning("OTel httpx instrumentation unavailable; skipping")

    _initialized = True
    logger.info("OpenTelemetry initialized (service=%s)", service_name)


def shutdown_otel() -> None:
    """Flush and shut down the OTel providers created by ``setup_otel``.

    The tracer/logger providers use ``Batch*`` processors that buffer telemetry;
    without an explicit shutdown, records still in the buffer are lost when the
    process exits. Call this from the app/worker shutdown path. Safe to call when
    OTel was never initialized (no-op) and never raises.
    """
    global _initialized, _tracer_provider, _meter_provider, _logger_provider
    for provider in (_tracer_provider, _meter_provider, _logger_provider):
        if provider is None:
            continue
        try:
            provider.shutdown()  # flushes then shuts down
        except Exception:
            logger.warning("Error shutting down OTel provider", exc_info=True)
    _tracer_provider = _meter_provider = _logger_provider = None
    _initialized = False


def instrument_app(app: object) -> None:
    """Instrument a FastAPI app so each request opens a root SERVER span.

    Without this, ``memory.*`` spans have no active parent context and each is
    exported as its own single-span trace. Attaching the FastAPI instrumentor
    makes the incoming HTTP request the root span; memory operations and the
    downstream httpx (LLM/embedding) CLIENT spans then nest under it, yielding a
    real end-to-end agent trace.

    Safe to call unconditionally: no-ops when OTel was never initialized (the
    global no-op TracerProvider produces no spans) and swallows import errors so
    a missing optional dependency never breaks startup.
    """
    if not _initialized:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("OTel FastAPI instrumentation enabled (request root spans)")
    except Exception:
        logger.warning("OTel FastAPI instrumentation unavailable; skipping")


def get_tracer(name: str = "honcho.memory") -> trace.Tracer:
    """Return a tracer scoped to *name*."""
    return trace.get_tracer(name)


def get_meter(name: str = "honcho.memory") -> metrics.Meter:
    """Return a meter scoped to *name*."""
    return metrics.get_meter(name)


def get_otel_logger(name: str = "honcho.memory") -> otel_logs.Logger:
    """Return an OTel Logger scoped to *name* for structured log emission."""
    return otel_logs.get_logger(name)
