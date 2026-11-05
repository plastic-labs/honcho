"""OpenTelemetry distributed tracing for the API and the deriver.

Off unless `TRACING_ENABLED`. When on, this module owns the process's global
`TracerProvider` and exports spans over OTLP/HTTP, either to `TRACING_OTLP_ENDPOINT`
(a local OTel Desktop Viewer, a collector) or to the Sentry project in `SENTRY_DSN`.

What a trace holds:

- automatic spans for API requests (FastAPI), SQL (SQLAlchemy), outbound HTTP
  (HTTPX, which covers the LLM and embedding SDKs) and Redis;
- semantic spans at Honcho's own boundaries (`start_span` callers): dialectic chat
  and SSE, agent tool calls, LLM completion/stream, embeddings, enqueue and the
  deriver's batch/item processing.

Structure, ids and timings only: no prompts, message bodies or tool results.
Content lives in Langfuse, which carries the OTel trace id in its metadata, and
spans carry `honcho.run_id` for the reverse lookup.

The queue is Postgres, so trace context crosses it inside the queue payload
(`QUEUE_TRACEPARENT_KEY`). Worker spans start their own trace and *link* to the
producing spans rather than parenting under them: one batch merges items from
many API requests.

Sampling (`HonchoSampler`): children follow their parent; a trace that starts here
is kept if its workspace is in `FORCE_SAMPLE_WORKSPACES`, if it links to a sampled
trace (so a sampled request's queue work is sampled too), or by `SAMPLE_RATE`.
Root spans that are not an entry point (SQL from a background poller, say) are
dropped so they cannot flood the backend.

Every helper here is a no-op while tracing is off and never raises into callers.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import re
import socket
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace import Span as SDKSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import (
    Decision,
    Sampler,
    SamplingResult,
    TraceIdRatioBased,
)
from opentelemetry.trace import Link, Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import Attributes

from src._version import HONCHO_VERSION
from src.config import settings
from src.telemetry.tenant import current_tenant_id
from src.utils.sanitization import escape_unstorable

logger = logging.getLogger(__name__)

T = TypeVar("T")

TRACER_NAME = "honcho"

# Queue payload key holding the producer's W3C traceparent. Popped before payload
# validation (like `_retry_attempts`); see `src/deriver/consumer.py`.
QUEUE_TRACEPARENT_KEY = "_traceparent"

# Worker spans link to at most this many producers (one per queue item).
MAX_QUEUE_LINKS = 128

_EXCLUDED_URLS = "/health,/metrics,/openapi.json,/docs,/redoc"
_WORKSPACE_PATH = re.compile(r"/v3/workspaces/([^/?#]+)")
_TRACEPARENT = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")
_propagator = TraceContextTextMapPropagator()

_provider: TracerProvider | None = None
_instrumented_apps: set[int] = set()
_engines_instrumented = False


class HonchoSampler(Sampler):
    """Parent-based sampling with workspace overrides and link inheritance."""

    def __init__(
        self,
        rate: float,
        force_workspaces: Sequence[str] = (),
        trust_remote_parent: bool = False,
    ) -> None:
        self._ratio: TraceIdRatioBased = TraceIdRatioBased(rate)
        self._force: frozenset[str] = frozenset(force_workspaces)
        self._trust_remote: bool = trust_remote_parent

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: Any = None,
    ) -> SamplingResult:
        parent = trace.get_current_span(parent_context).get_span_context()
        if parent.is_valid and (not parent.is_remote or self._trust_remote):
            decision = (
                Decision.RECORD_AND_SAMPLE
                if parent.trace_flags.sampled
                else Decision.DROP
            )
            return SamplingResult(decision, attributes, parent.trace_state)

        attrs: Mapping[str, Any] = attributes or {}
        if not _is_entry_span(name, kind):
            return SamplingResult(Decision.DROP)
        workspace = _workspace_from_attributes(attrs)
        if workspace is not None and workspace in self._force:
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)
        if any(link.context.trace_flags.sampled for link in links or ()):
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)
        return self._ratio.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"HonchoSampler({self._ratio.get_description()})"


def _is_entry_span(name: str, kind: SpanKind | None) -> bool:
    # API requests and Honcho's own semantic roots (worker batches/items) start
    # traces; an orphan SQL/HTTP/Redis span from a background loop does not.
    return kind in (SpanKind.SERVER, SpanKind.CONSUMER) or name.startswith("honcho.")


def _workspace_from_attributes(attrs: Mapping[str, Any]) -> str | None:
    if isinstance(workspace := attrs.get("honcho.workspace_id"), str):
        return workspace
    for key in ("url.path", "http.target"):
        if isinstance(path := attrs.get(key), str) and (
            match := _WORKSPACE_PATH.search(path)
        ):
            return match.group(1)
    return None


class _TenantSpanProcessor(SpanProcessor):
    """Stamp the ambient tenant on every span, mirroring the other telemetry."""

    def on_start(self, span: SDKSpan, parent_context: Context | None = None) -> None:
        if (tenant_id := current_tenant_id()) is not None:
            span.set_attribute("honcho.tenant_id", tenant_id)

    def on_end(self, span: ReadableSpan) -> None:
        pass


def _encodable(value: Any) -> Any:
    if isinstance(value, str):
        escaped = escape_unstorable(value)
        return value if escaped == value else escaped
    if isinstance(value, (tuple, list)):
        items = cast(Sequence[Any], value)
        cleaned: tuple[Any, ...] = tuple(_encodable(item) for item in items)
        unchanged = all(a is b for a, b in zip(cleaned, items, strict=True))
        return items if unchanged else cleaned
    return value


def _encodable_attributes(attributes: Attributes) -> Attributes:
    if not attributes:
        return attributes
    cleaned = {key: _encodable(value) for key, value in attributes.items()}
    return (
        attributes
        if all(cleaned[key] is value for key, value in attributes.items())
        else cleaned
    )


def _encodable_span(span: ReadableSpan) -> ReadableSpan:
    attributes = _encodable_attributes(span.attributes)
    events = [
        Event(e.name, a, e.timestamp)
        if (a := _encodable_attributes(e.attributes)) is not e.attributes
        else e
        for e in span.events
    ]
    status = span.status
    if status.description and (d := _encodable(status.description)) is not (
        status.description
    ):
        status = Status(status.status_code, d)
    if (
        attributes is span.attributes
        and status is span.status
        and all(a is b for a, b in zip(events, span.events, strict=True))
    ):
        return span
    return ReadableSpan(
        name=_encodable(span.name),
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=events,
        links=span.links,
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class EncodableSpanExporter(SpanExporter):
    """Escape strings OTLP cannot encode before handing spans to `inner`.

    Protobuf rejects lone surrogates. The OTLP encoder then drops the whole
    attribute and logs an error for each one, so an exception that echoes bad
    input (#1492) loses its message and stack trace. Escaping them (and NUL),
    as `\\ud83d`, keeps that evidence. Spans without such strings pass through.
    """

    def __init__(self, inner: SpanExporter) -> None:
        self._inner: SpanExporter = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([_encodable_span(span) for span in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def _exporter_target() -> tuple[str, dict[str, str] | None]:
    if settings.TRACING.EXPORTER == "sentry":
        from sentry_sdk.consts import VERSION, EndpointType
        from sentry_sdk.utils import Dsn

        if not settings.SENTRY.DSN:
            raise ValueError("TRACING_EXPORTER=sentry requires SENTRY_DSN")
        auth = Dsn(settings.SENTRY.DSN).to_auth(f"sentry.python/{VERSION}")
        return (
            auth.get_api_url(EndpointType.OTLP_TRACES),
            {"X-Sentry-Auth": auth.to_header()},
        )
    endpoint = settings.TRACING.OTLP_ENDPOINT.rstrip("/")
    return f"{endpoint}/v1/traces", settings.TRACING.OTLP_HEADERS


def tracing_enabled() -> bool:
    return _provider is not None


def initialize_tracing(
    *, instance_type: str, app: Any = None, engines: Sequence[Any] = ()
) -> None:
    """Set up tracing for this process; safe to call more than once.

    Args:
        instance_type: "api" or "deriver"; becomes `service.name` honcho-<type>.
        app: FastAPI app to instrument (API only).
        engines: SQLAlchemy async engines to instrument.
    """
    global _provider, _engines_instrumented
    if not settings.TRACING.ENABLED:
        return
    try:
        if _provider is None:
            endpoint, headers = _exporter_target()
            provider = TracerProvider(
                resource=Resource.create(
                    {
                        "service.name": f"honcho-{instance_type}",
                        "service.namespace": settings.NAMESPACE,
                        "service.version": settings.SENTRY.RELEASE or HONCHO_VERSION,
                        "service.instance.id": f"{socket.gethostname()}:{os.getpid()}",
                        "deployment.environment.name": settings.SENTRY.ENVIRONMENT,
                    }
                ),
                sampler=HonchoSampler(
                    settings.TRACING.SAMPLE_RATE,
                    settings.TRACING.FORCE_SAMPLE_WORKSPACES,
                    settings.TRACING.TRUST_INCOMING_TRACEPARENT,
                ),
            )
            provider.add_span_processor(_TenantSpanProcessor())
            provider.add_span_processor(
                BatchSpanProcessor(
                    EncodableSpanExporter(
                        OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=5)
                    )
                )
            )
            trace.set_tracer_provider(provider)
            _provider = provider

            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            HTTPXClientInstrumentor().instrument()
            RedisInstrumentor().instrument()  # pyright: ignore[reportUnknownMemberType]
            logger.info(
                "Tracing enabled: exporter=%s sample_rate=%s",
                settings.TRACING.EXPORTER,
                settings.TRACING.SAMPLE_RATE,
            )

        if app is not None and id(app) not in _instrumented_apps:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=_provider,
                excluded_urls=_EXCLUDED_URLS,
                exclude_spans=["receive", "send"],
            )
            _instrumented_apps.add(id(app))

        sync_engines = list(
            {
                id(sync): sync
                for sync in (getattr(e, "sync_engine", e) for e in engines)
            }.values()
        )
        if sync_engines and not _engines_instrumented:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            # One call: the instrumentor is a process-wide singleton and
            # ignores a second instrument().
            SQLAlchemyInstrumentor().instrument(
                engines=sync_engines,
                tracer_provider=_provider,
                enable_commenter=False,
            )
            _engines_instrumented = True
    except Exception:
        logger.warning("Tracing setup failed; continuing without it", exc_info=True)


def shutdown_tracing() -> None:
    """Flush buffered spans (bounded) and stop exporting."""
    global _provider
    provider, _provider = _provider, None
    if provider is None:
        return
    try:
        provider.force_flush(timeout_millis=3000)
        provider.shutdown()
    except Exception:
        logger.warning("Tracing shutdown failed", exc_info=True)


def _tracer() -> trace.Tracer:
    return (_provider or trace.get_tracer_provider()).get_tracer(TRACER_NAME)


def _clean(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (attributes or {}).items() if v is not None}


@contextmanager
def start_span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    links: Sequence[Link] | None = None,
) -> Generator[Span]:
    """Open a semantic span as the current span.

    An exception marks the span as an error and is recorded on it; cancellation
    (`CancelledError`, `GeneratorExit`) is tagged `honcho.outcome=cancelled`
    without an error status. Exceptions always propagate unchanged.
    """
    if _provider is None:
        yield trace.INVALID_SPAN
        return
    with _tracer().start_as_current_span(
        name,
        kind=kind,
        attributes=_clean(attributes),
        links=links,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as exc:
            _record_failure(span, exc)
            raise


def _record_failure(span: Span, exc: BaseException) -> None:
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        span.set_attribute("honcho.outcome", "cancelled")
        return
    span.set_attribute("honcho.outcome", "error")
    span.set_attribute("error.type", type(exc).__name__)
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR))


def start_detached_span(
    name: str, *, attributes: Mapping[str, Any] | None = None
) -> Span:
    """Start a span that is not made current; end it with `finish_span`.

    For work that spans `yield`s (streams): attaching a context across a yield
    leaks it into the consumer, so callers activate the span with `use_span`
    only around each awaited step instead.
    """
    if _provider is None:
        return trace.INVALID_SPAN
    return _tracer().start_span(
        name, attributes=_clean(attributes), record_exception=False
    )


@contextmanager
def use_span(span: Span) -> Generator[None]:
    """Make `span` current for a block that does not yield."""
    if not span.is_recording():
        yield
        return
    with trace.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    ):
        yield


def finish_span(span: Span, error: BaseException | None = None) -> None:
    """End a detached span, recording `error` if the work failed."""
    if not span.is_recording():
        return
    if error is not None:
        _record_failure(span, error)
    span.end()


async def traced_stream(
    name: str,
    stream: AsyncIterator[T],
    *,
    attributes: Mapping[str, Any] | None = None,
) -> AsyncGenerator[T]:
    """Yield from `stream` inside a span that is current only while it produces.

    The span covers the stream's whole life (until exhausted, failed or closed),
    and work the stream does between items nests under it; the consumer's code
    between items does not.
    """
    span = start_detached_span(name, attributes=attributes)
    error: BaseException | None = None
    items = aiter(stream)
    try:
        while True:
            with use_span(span):
                try:
                    item = await anext(items)
                except StopAsyncIteration:
                    break
            yield item
    except BaseException as exc:
        error = exc
        raise
    finally:
        if isinstance(items, AsyncGenerator):
            with use_span(span):
                await items.aclose()
        finish_span(span, error)


def set_span_attributes(attributes: Mapping[str, Any]) -> None:
    """Set attributes on the current span, skipping None values."""
    if _provider is not None:
        trace.get_current_span().set_attributes(_clean(attributes))


def add_span_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
    if _provider is not None:
        trace.get_current_span().add_event(name, _clean(attributes))


def mark_span_error(error: BaseException | str) -> None:
    """Mark the current span failed when an error is handled rather than raised.

    Pass the exception itself where there is one: it is recorded as an
    exception event, as for an error that propagates out of `start_span`, so a
    caught error keeps its message and stack. A string is just the error type.
    """
    if _provider is None:
        return
    span = trace.get_current_span()
    if isinstance(error, BaseException):
        span.set_attribute("error.type", type(error).__name__)
        span.record_exception(error)
    else:
        span.set_attribute("error.type", error)
    span.set_status(Status(StatusCode.ERROR))


def record_span_exception(error: BaseException) -> None:
    """Record a secondary exception on the current span without changing its status."""
    if _provider is not None:
        trace.get_current_span().record_exception(error)


def current_trace_id(*, sampled_only: bool = False) -> str | None:
    """Hex trace id of the current span, or None.

    Unsampled ids are still useful: Sentry errors carry them. Pass
    `sampled_only` when the id is a pointer to the exported trace itself.
    """
    if _provider is None:
        return None
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid or (sampled_only and not ctx.trace_flags.sampled):
        return None
    return trace.format_trace_id(ctx.trace_id)


def current_traceparent() -> str | None:
    """W3C traceparent for the current span, for carrying across the queue."""
    if _provider is None:
        return None
    carrier: dict[str, str] = {}
    try:
        _propagator.inject(carrier)
    except Exception:
        return None
    return carrier.get("traceparent")


def link_from_traceparent(
    traceparent: object, attributes: Mapping[str, Any] | None = None
) -> Link | None:
    """A span link to the producer named by a queued traceparent, if valid."""
    if (
        _provider is None
        or not isinstance(traceparent, str)
        or not _TRACEPARENT.match(traceparent)
    ):
        return None
    ctx = trace.get_current_span(
        _propagator.extract({"traceparent": traceparent}, context=Context())
    ).get_span_context()
    return Link(ctx, _clean(attributes)) if ctx.is_valid else None


def queue_links(payloads: Sequence[Mapping[str, Any] | None]) -> list[Link]:
    """Links from a worker span to the spans that enqueued its items.

    One link per distinct producer: items from one API request share it.
    """
    links: dict[int, Link] = {}
    for payload in payloads:
        if len(links) >= MAX_QUEUE_LINKS:
            break
        if payload and (
            link := link_from_traceparent(payload.get(QUEUE_TRACEPARENT_KEY))
        ):
            links.setdefault(link.context.span_id, link)
    return list(links.values())


def queue_wait_ms(
    created_at: Sequence[datetime.datetime | None],
) -> float | None:
    """Milliseconds the oldest of these queue items waited before processing."""
    stamps = [stamp for stamp in created_at if stamp is not None]
    if _provider is None or not stamps:
        return None
    oldest = min(stamps)
    if oldest.tzinfo is None:
        oldest = oldest.replace(tzinfo=datetime.UTC)
    return max(
        0.0, (datetime.datetime.now(datetime.UTC) - oldest).total_seconds() * 1000
    )


def stamp_queue_records(
    records: list[dict[str, Any]], *, traceparent: str | None = None
) -> list[dict[str, Any]]:
    """Add a traceparent to each queue record's payload.

    Uses `traceparent` when given, else the current span's. Does nothing while
    tracing is off. Rollout: a deriver older than this change rejects payloads
    carrying the key (`extra="forbid"`), so enable tracing only once every
    deriver runs it.
    """
    if _provider is None:
        return records
    if (traceparent := traceparent or current_traceparent()) is not None:
        for record in records:
            payload = record.get("payload")
            if isinstance(payload, dict):
                payload[QUEUE_TRACEPARENT_KEY] = traceparent
    return records
