"""OpenTelemetry tracing: sampling, queue context, span outcomes, stream context.

Spans are recorded into an in-memory exporter by swapping `otel._provider`; no
global tracer provider is installed and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import Decision
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
)

from src import models
from src.deriver import consumer
from src.deriver.queue_manager import QueueManager
from src.llm import executor, registry
from src.llm.backend import StreamChunk
from src.llm.runtime import AttemptPlan
from src.telemetry import otel
from src.telemetry.otel import HonchoSampler
from src.utils.queue_payload import create_deletion_payload
from src.webhooks import events as webhook_events

TRACE_ID = 0x0AF7651916CD43DD8448EB211C80319C


def _decision(sampler: HonchoSampler, **kwargs: Any) -> Decision:
    kwargs.setdefault("name", "honcho.test")
    return sampler.should_sample(
        kwargs.pop("parent_context", None), TRACE_ID, **kwargs
    ).decision


def _parent(*, sampled: bool, remote: bool) -> Context:
    ctx = SpanContext(
        trace_id=TRACE_ID,
        span_id=0x00F067AA0BA902B7,
        is_remote=remote,
        trace_flags=TraceFlags(TraceFlags.SAMPLED if sampled else 0),
    )
    return trace.set_span_in_context(NonRecordingSpan(ctx))


def _link(*, sampled: bool) -> Link:
    return Link(
        SpanContext(
            trace_id=TRACE_ID + 1,
            span_id=0x1,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED if sampled else 0),
        )
    )


class TestSampler:
    def test_child_follows_local_parent(self) -> None:
        sampler = HonchoSampler(0.0)
        assert (
            _decision(sampler, parent_context=_parent(sampled=True, remote=False))
            == Decision.RECORD_AND_SAMPLE
        )
        assert (
            _decision(
                HonchoSampler(1.0),
                parent_context=_parent(sampled=False, remote=False),
            )
            == Decision.DROP
        )

    def test_incoming_traceparent_ignored_unless_trusted(self) -> None:
        remote = _parent(sampled=True, remote=True)
        assert (
            _decision(HonchoSampler(0.0), parent_context=remote, kind=SpanKind.SERVER)
            == Decision.DROP
        )
        assert (
            _decision(
                HonchoSampler(0.0, trust_remote_parent=True), parent_context=remote
            )
            == Decision.RECORD_AND_SAMPLE
        )

    def test_orphan_non_entry_root_is_dropped(self) -> None:
        # SQL from a background poller has no parent and is not an entry point.
        assert (
            _decision(HonchoSampler(1.0), name="SELECT honcho", kind=SpanKind.CLIENT)
            == Decision.DROP
        )

    def test_entry_roots_use_rate(self) -> None:
        assert (
            _decision(HonchoSampler(1.0), name="POST /v3/x", kind=SpanKind.SERVER)
            == Decision.RECORD_AND_SAMPLE
        )
        assert _decision(HonchoSampler(0.0)) == Decision.DROP

    @pytest.mark.parametrize(
        "attributes",
        [
            {"honcho.workspace_id": "acme"},
            {"url.path": "/v3/workspaces/acme/peers/alice/chat"},
            {"http.target": "/v3/workspaces/acme/sessions/list"},
        ],
    )
    def test_forced_workspace_always_sampled(self, attributes: dict[str, str]) -> None:
        sampler = HonchoSampler(0.0, force_workspaces=["acme"])
        assert (
            _decision(sampler, kind=SpanKind.SERVER, attributes=attributes)
            == Decision.RECORD_AND_SAMPLE
        )
        assert (
            _decision(
                sampler,
                kind=SpanKind.SERVER,
                attributes={"url.path": "/v3/workspaces/other/peers"},
            )
            == Decision.DROP
        )

    def test_root_linked_to_sampled_producer_is_sampled(self) -> None:
        sampler = HonchoSampler(0.0)
        assert (
            _decision(sampler, links=[_link(sampled=False), _link(sampled=True)])
            == Decision.RECORD_AND_SAMPLE
        )
        assert _decision(sampler, links=[_link(sampled=False)]) == Decision.DROP


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=HonchoSampler(1.0))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel, "_provider", provider)
    yield exporter
    provider.shutdown()


def _sid(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.span_id


def _by_name(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    [span] = [s for s in exporter.get_finished_spans() if s.name == name]
    return span


class TestDisabled:
    def test_helpers_are_noops(self) -> None:
        assert otel._provider is None  # pyright: ignore[reportPrivateUsage]
        with otel.start_span("honcho.x", attributes={"a": 1}) as span:
            assert not span.is_recording()
            otel.set_span_attributes({"b": 2})
        assert otel.current_trace_id() is None
        assert otel.current_traceparent() is None
        assert otel.stamp_queue_records([{"payload": {}}]) == [{"payload": {}}]
        assert otel.queue_links([{otel.QUEUE_TRACEPARENT_KEY: "00-" + "a" * 32}]) == []


class TestSpans:
    def test_error_is_recorded_and_propagates(
        self, spans: InMemorySpanExporter
    ) -> None:
        with pytest.raises(ValueError), otel.start_span("honcho.fails"):
            raise ValueError("boom")
        span = _by_name(spans, "honcho.fails")
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "ValueError"
        assert span.attributes["honcho.outcome"] == "error"

    def test_cancellation_is_not_an_error(self, spans: InMemorySpanExporter) -> None:
        with pytest.raises(asyncio.CancelledError), otel.start_span("honcho.cancel"):
            raise asyncio.CancelledError
        span = _by_name(spans, "honcho.cancel")
        assert span.status.status_code != StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["honcho.outcome"] == "cancelled"

    def test_handled_error_keeps_its_exception(
        self, spans: InMemorySpanExporter
    ) -> None:
        with otel.start_span("honcho.handled"):
            try:
                raise ValueError("boom")
            except ValueError as e:
                otel.mark_span_error(e)
        span = _by_name(spans, "honcho.handled")
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "ValueError"
        [event] = span.events
        assert event.name == "exception"
        assert event.attributes is not None
        assert event.attributes["exception.message"] == "boom"

    def test_none_attributes_are_dropped(self, spans: InMemorySpanExporter) -> None:
        with otel.start_span("honcho.attrs", attributes={"a": None, "b": "x"}):
            otel.set_span_attributes({"c": None, "d": 1})
        assert dict(_by_name(spans, "honcho.attrs").attributes or {}) == {
            "b": "x",
            "d": 1,
        }


class TestQueueContext:
    def test_round_trip_links_worker_to_producer(
        self, spans: InMemorySpanExporter
    ) -> None:
        with otel.start_span("honcho.queue.enqueue", kind=SpanKind.PRODUCER):
            records = otel.stamp_queue_records(
                [{"payload": {"a": 1}}, {"payload": {"b": 2}}]
            )
        producer = _by_name(spans, "honcho.queue.enqueue")
        payloads = [r["payload"] for r in records]
        links = otel.queue_links([*payloads, None, {"_traceparent": "garbage"}])
        # Items from one request share their producer: one link, not two.
        assert [link.context.span_id for link in links] == [_sid(producer)]

    @pytest.mark.usefixtures("spans")
    def test_link_count_is_capped(self) -> None:
        payloads: list[dict[str, Any]] = []
        for _ in range(otel.MAX_QUEUE_LINKS + 5):
            with otel.start_span("honcho.queue.enqueue"):
                payloads += [
                    r["payload"] for r in otel.stamp_queue_records([{"payload": {}}])
                ]
        assert len(otel.queue_links(payloads)) == otel.MAX_QUEUE_LINKS

    @pytest.mark.usefixtures("spans")
    def test_wait_ms_uses_oldest_item(self) -> None:
        now = datetime.datetime.now(datetime.UTC)
        wait = otel.queue_wait_ms([now, None, now - datetime.timedelta(seconds=2)])
        assert wait is not None and wait >= 2000
        # Unsaved items have no created_at yet.
        assert otel.queue_wait_ms([None]) is None

    @pytest.mark.asyncio
    async def test_consumer_strips_traceparent_before_validation(
        self, spans: InMemorySpanExporter
    ) -> None:
        """Payload models forbid extra keys; the traceparent must not reach them."""
        with otel.start_span("honcho.queue.enqueue", kind=SpanKind.PRODUCER):
            [record] = otel.stamp_queue_records(
                [{"payload": create_deletion_payload("session", "sess_1")}]
            )
        assert otel.QUEUE_TRACEPARENT_KEY in record["payload"]
        item = models.QueueItem(
            id=7,
            task_type="deletion",
            work_unit_key="deletion:ws:sess_1",
            workspace_name="ws",
            payload=record["payload"],
            created_at=datetime.datetime.now(datetime.UTC),
        )
        with (
            patch.object(consumer, "process_deletion", new=AsyncMock()) as deleted,
            consumer.queue_item_span(item),
        ):
            await consumer.process_item(item)
        deleted.assert_awaited_once()

        worker = _by_name(spans, "honcho.queue.process_item")
        producer = _by_name(spans, "honcho.queue.enqueue")
        assert worker.kind == SpanKind.CONSUMER
        assert worker.parent is None  # its own trace, linked to the producer
        assert [link.context.span_id for link in worker.links] == [_sid(producer)]
        assert worker.attributes is not None
        assert worker.attributes["honcho.workspace_id"] == "ws"
        assert worker.attributes["honcho.queue.attempt"] == 1

    @pytest.mark.asyncio
    async def test_webhook_links_to_given_traceparent(
        self, spans: InMemorySpanExporter
    ) -> None:
        """queue.empty publishes after the work unit's spans end, so it passes the
        drained item's traceparent explicitly; the webhook's worker links to it."""
        with otel.start_span("honcho.queue.enqueue", kind=SpanKind.PRODUCER):
            traceparent = otel.current_traceparent()
        added: list[models.QueueItem] = []
        db = AsyncMock()
        db.add = added.append

        @contextlib.asynccontextmanager
        async def fake_db(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Any]:
            yield db

        with patch.object(webhook_events, "tracked_db", fake_db):
            await webhook_events.publish_webhook_event(
                webhook_events.TestEvent(workspace_id="ws"), traceparent=traceparent
            )
        [item] = added
        assert item.payload[otel.QUEUE_TRACEPARENT_KEY] == traceparent

        item.id = 9
        item.created_at = datetime.datetime.now(datetime.UTC)
        with (
            patch(
                "src.webhooks.webhook_delivery.deliver_webhook", new=AsyncMock()
            ) as delivered,
            consumer.queue_item_span(item),
        ):
            await consumer.process_item(item)
        delivered.assert_awaited_once()
        worker = _by_name(spans, "honcho.queue.process_item")
        producer = _by_name(spans, "honcho.queue.enqueue")
        assert [link.context.span_id for link in worker.links] == [_sid(producer)]


def _plan() -> AttemptPlan:
    return AttemptPlan(
        provider="anthropic",
        model="claude-sonnet-4-5",
        client=cast(Any, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=cast(Any, object()),
        attempt=2,
        retry_attempts=3,
        is_fallback=True,
    )


class TestLLMStreamSpan:
    @pytest.mark.asyncio
    async def test_provider_io_nests_under_span_but_consumer_does_not(
        self, spans: InMemorySpanExporter
    ) -> None:
        tracer = otel._tracer()  # pyright: ignore[reportPrivateUsage]

        async def provider_stream() -> AsyncIterator[StreamChunk]:
            # Stands in for the SDK's HTTP request, opened on first pull.
            with tracer.start_as_current_span("POST provider"):
                pass
            yield StreamChunk(content="")
            yield StreamChunk(content="hi")
            yield StreamChunk(is_done=True, finish_reason="stop", output_tokens=3)

        seen_between_chunks: list[Any] = []
        with (
            patch.object(registry, "CLIENTS", {"anthropic": object()}),
            patch.object(executor, "backend_for_provider", return_value=object()),
            patch.object(
                executor,
                "execute_stream",
                new=AsyncMock(return_value=provider_stream()),
            ),
            patch("src.telemetry.events.emit"),
            otel.start_span("honcho.dialectic.chat_stream"),
        ):
            outer = trace.get_current_span()
            stream = await executor.honcho_llm_call_inner(
                "anthropic",
                "claude-sonnet-4-5",
                "hello",
                max_tokens=64,
                stream=True,
                plan=_plan(),
                telemetry=None,
            )
            async for _chunk in stream:
                seen_between_chunks.append(trace.get_current_span())

        assert all(span is outer for span in seen_between_chunks)
        llm = _by_name(spans, "honcho.llm.stream")
        http = _by_name(spans, "POST provider")
        chat = _by_name(spans, "honcho.dialectic.chat_stream")
        assert http.parent is not None and http.parent.span_id == _sid(llm)
        assert llm.parent is not None and llm.parent.span_id == _sid(chat)
        assert [event.name for event in llm.events] == ["honcho.llm.first_token"]
        assert llm.attributes is not None
        assert llm.attributes["gen_ai.usage.output_tokens"] == 3
        assert llm.attributes["honcho.llm.attempt"] == 2
        assert llm.attributes["honcho.llm.is_fallback"] is True
        assert "gen_ai.usage.input_tokens" not in llm.attributes


class TestCorrelation:
    @pytest.mark.usefixtures("spans")
    def test_langfuse_metadata_carries_sampled_trace_id(self) -> None:
        from types import SimpleNamespace

        from src.telemetry.langfuse_exporter import LangfuseExporter

        call = cast(Any, SimpleNamespace(workspace_name="ws", trace_id="run_1"))
        assert "otel_trace_id" not in LangfuseExporter._root_metadata(call)  # pyright: ignore[reportPrivateUsage]
        with otel.start_span("honcho.dialectic.chat") as span:
            md = LangfuseExporter._root_metadata(call)  # pyright: ignore[reportPrivateUsage]
        assert md["otel_trace_id"] == trace.format_trace_id(
            span.get_span_context().trace_id
        )

    @pytest.mark.parametrize("tracing", [True, False])
    def test_sentry_native_tracing_off_when_otel_owns_it(
        self, monkeypatch: pytest.MonkeyPatch, tracing: bool
    ) -> None:
        from sentry_sdk.integrations.otlp import OTLPIntegration

        from src.config import settings
        from src.telemetry import sentry

        monkeypatch.setattr(settings.TRACING, "ENABLED", tracing)
        with patch("sentry_sdk.init") as init:
            sentry.initialize_sentry(integrations=[])
        kwargs = init.call_args.kwargs
        otlp = [i for i in kwargs["integrations"] if isinstance(i, OTLPIntegration)]
        if tracing:
            assert kwargs["traces_sample_rate"] is None
            assert "traces_sampler" not in kwargs
            [integration] = otlp
            # otel.py owns the exporter and propagator; Sentry only links errors.
            assert integration.setup_otlp_traces_exporter is False
            assert integration.setup_propagator is False
        else:
            assert kwargs["traces_sampler"] is sentry.traces_sampler
            assert otlp == []


class TestTracedStream:
    @staticmethod
    async def _source(fail: bool = False) -> AsyncIterator[int]:
        tracer = otel._tracer()  # pyright: ignore[reportPrivateUsage]
        for i in range(3):
            with tracer.start_as_current_span(f"work {i}"):
                pass
            yield i
        if fail:
            raise RuntimeError("upstream")

    @pytest.mark.asyncio
    async def test_work_nests_and_consumer_context_is_untouched(
        self, spans: InMemorySpanExporter
    ) -> None:
        with otel.start_span("honcho.request"):
            outer = trace.get_current_span()
            seen: list[int] = []
            async for item in otel.traced_stream("honcho.stream", self._source()):
                assert trace.get_current_span() is outer
                seen.append(item)
        assert seen == [0, 1, 2]
        stream = _by_name(spans, "honcho.stream")
        for i in range(3):
            work = _by_name(spans, f"work {i}")
            assert work.parent is not None
            assert work.parent.span_id == _sid(stream)

    @pytest.mark.asyncio
    async def test_failure_and_early_close_end_the_span(
        self, spans: InMemorySpanExporter
    ) -> None:
        with pytest.raises(RuntimeError):
            async for _ in otel.traced_stream("honcho.fails", self._source(fail=True)):
                pass
        assert _by_name(spans, "honcho.fails").status.status_code == StatusCode.ERROR

        stream = otel.traced_stream("honcho.closed", self._source())
        assert await anext(stream) == 0
        await stream.aclose()
        closed = _by_name(spans, "honcho.closed")
        assert closed.attributes is not None
        assert closed.attributes["honcho.outcome"] == "cancelled"


class TestQueueOutcome:
    """The worker span says what happened to the item, after error handling."""

    @staticmethod
    def _item() -> models.QueueItem:
        return models.QueueItem(
            id=3,
            task_type="summary",
            work_unit_key="summary:ws:s1",
            workspace_name="ws",
            payload={"session_name": "s1"},
            created_at=datetime.datetime.now(datetime.UTC),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mark_error", "outcome", "released"),
        [
            (None, "failed", False),
            (RuntimeError("UPDATE rejected"), "mark_failed", True),
        ],
    )
    async def test_terminal_error_outcome(
        self,
        spans: InMemorySpanExporter,
        mark_error: Exception | None,
        outcome: str,
        released: bool,
    ) -> None:
        """#1492: a terminal error whose errored-mark fails leaves the item
        unprocessed; the span must not claim it failed for good."""
        qm = QueueManager.__new__(QueueManager)
        item = self._item()
        with (
            patch.object(
                qm, "mark_queue_item_as_errored", new=AsyncMock(side_effect=mark_error)
            ),
            patch.object(qm, "_clear_work_unit_retry_attempts", new=AsyncMock()),
            consumer.queue_item_span(item),
        ):
            release = await qm._handle_processing_error(  # pyright: ignore[reportPrivateUsage]
                ValueError("bad value"), [item], item.work_unit_key, "processing item"
            )
        assert release is released
        span = _by_name(spans, "honcho.queue.process_item")
        assert span.attributes is not None
        assert span.attributes["honcho.queue.outcome"] == outcome
        assert span.attributes["error.type"] == "ValueError"
        messages = [(e.attributes or {}).get("exception.message") for e in span.events]
        assert messages == ["bad value", *([str(mark_error)] if mark_error else [])]


class TestEncodableExport:
    def test_unencodable_strings_are_escaped_not_dropped(self) -> None:
        """#1492: an exception echoing a lone surrogate must keep its message;
        the OTLP encoder would otherwise drop the attribute and log an error."""
        sink = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(
            SimpleSpanProcessor(otel.EncodableSpanExporter(sink))
        )
        tracer = provider.get_tracer("test")
        bad = "bad value 'a\x00b' and '\ud83d'"
        with tracer.start_as_current_span("dirty") as span:
            span.record_exception(ValueError(bad))
            span.set_attribute("honcho.note", bad)
            span.set_status(Status(StatusCode.ERROR, bad))
        with tracer.start_as_current_span("clean"):
            pass

        dirty, clean = sink.get_finished_spans()
        escaped = "bad value 'a\\x00b' and '\\ud83d'"
        assert dirty.attributes is not None
        assert dirty.attributes["honcho.note"] == escaped
        assert dirty.status.description == escaped
        [event] = dirty.events
        assert event.attributes is not None
        assert event.attributes["exception.message"] == escaped
        assert "\\ud83d" in str(event.attributes["exception.stacktrace"])
        # Encodes in full: no attribute is dropped.
        encoded = encode_spans([dirty])
        [span_proto] = encoded.resource_spans[0].scope_spans[0].spans
        assert {a.key for a in span_proto.events[0].attributes} >= {
            "exception.message",
            "exception.stacktrace",
        }
        assert clean.name == "clean" and clean.events == ()
