"""Prometheus metrics for Honcho."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from enum import Enum
from typing import cast, final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from starlette.requests import Request
from starlette.responses import Response

from src.config import REASONING_LEVELS, settings
from src.utils.types import walk_subclasses

disable_created_metrics()

logger = logging.getLogger(__name__)


class NamespacedCounter(Counter):
    def labels(self, **kwargs: str) -> NamespacedCounter:
        kwargs["namespace"] = cast(str, settings.METRICS.NAMESPACE)
        return super().labels(**kwargs)  # type: ignore[return-value]


class NamespacedGauge(Gauge):
    def labels(self, **kwargs: str) -> NamespacedGauge:
        kwargs["namespace"] = cast(str, settings.METRICS.NAMESPACE)
        return super().labels(**kwargs)  # type: ignore[return-value]


class NamespacedHistogram(Histogram):
    def labels(self, **kwargs: str) -> NamespacedHistogram:
        kwargs["namespace"] = cast(str, settings.METRICS.NAMESPACE)
        return super().labels(**kwargs)  # type: ignore[return-value]


class TokenTypes(Enum):
    INPUT = "input"
    OUTPUT = "output"


class DeriverTaskTypes(Enum):
    INGESTION = "ingestion"
    SUMMARY = "summary"


class DeriverComponents(Enum):
    PROMPT = "prompt"
    MESSAGES = "messages"
    PREVIOUS_SUMMARY = "previous_summary"
    OUTPUT_TOTAL = "output_total"


class DialecticComponents(Enum):
    TOTAL = "total"


# Valid (token_type, component) pairs for deriver_tokens_processed, per task_type,
# used to zero-initialize counter children (see initialize_bounded_metrics).
# region ai
# NOT the cartesian product: input tokens only pair with input components, output
# only with OUTPUT_TOTAL, and PREVIOUS_SUMMARY occurs only for summary tasks
# (ingestion has no previous summary). Enumerating anything broader would fabricate
# impossible always-0 series (e.g. output/prompt, or ingestion/previous_summary).
# Explicit literal, drift-guarded by tests/telemetry/test_metric_zero_init.py.
# Sources: track_deriver_input_tokens (src/utils/tokens.py) + the OUTPUT_TOTAL sites
# in src/deriver/deriver.py and src/utils/summarizer.py.
# endregion
_DERIVER_TOKEN_COMBOS_BY_TASK: dict[str, tuple[tuple[str, str], ...]] = {
    DeriverTaskTypes.INGESTION.value: (
        (TokenTypes.INPUT.value, DeriverComponents.PROMPT.value),
        (TokenTypes.INPUT.value, DeriverComponents.MESSAGES.value),
        (TokenTypes.OUTPUT.value, DeriverComponents.OUTPUT_TOTAL.value),
    ),
    DeriverTaskTypes.SUMMARY.value: (
        (TokenTypes.INPUT.value, DeriverComponents.PROMPT.value),
        (TokenTypes.INPUT.value, DeriverComponents.MESSAGES.value),
        (TokenTypes.INPUT.value, DeriverComponents.PREVIOUS_SUMMARY.value),
        (TokenTypes.OUTPUT.value, DeriverComponents.OUTPUT_TOTAL.value),
    ),
}


api_requests_counter = NamespacedCounter(
    "api_requests",
    "Total API requests",
    ["namespace", "method", "endpoint", "status_code"],
)

# Per-route latency. Buckets are a geometric ladder spanning
# the full range of API classes
api_request_duration_seconds = NamespacedHistogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["namespace", "method", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20, 30, 60, 120),
)

messages_created_counter = NamespacedCounter(
    "messages_created",
    "Total messages created",
    ["namespace", "workspace_name"],
)

embed_now_tasks_shed_counter = NamespacedCounter(
    "embed_now_tasks_shed",
    "Immediate-embed background tasks skipped because MAX_PENDING_EMBED_TASKS was reached",
    ["namespace"],
)

embed_now_tasks_in_flight_gauge = NamespacedGauge(
    "embed_now_tasks_in_flight",
    "Immediate-embed background tasks currently in flight for this process",
    ["namespace"],
)

dialectic_calls_counter = NamespacedCounter(
    "dialectic_calls",
    "Total dialectic calls",
    ["namespace", "workspace_name", "reasoning_level"],
)

deriver_queue_items_processed_counter = NamespacedCounter(
    "deriver_queue_items_processed",
    "Total deriver queue items processed",
    ["namespace", "workspace_name", "task_type"],
)

deriver_tokens_processed_counter = NamespacedCounter(
    "deriver_tokens_processed",
    "Total tokens processed by the deriver",
    ["namespace", "task_type", "token_type", "component"],
)

dialectic_tokens_processed_counter = NamespacedCounter(
    "dialectic_tokens_processed",
    "Total tokens processed by the dialectic",
    ["namespace", "token_type", "component", "reasoning_level"],
)

dreamer_tokens_processed_counter = NamespacedCounter(
    "dreamer_tokens_processed",
    "Total tokens processed by the dreamer",
    ["namespace", "specialist_name", "token_type"],
)

# CloudEvents emitter health metrics. Split intentional (sampled out) vs unintentional
# (dropped) so the dropped counter remains a real alert signal.
telemetry_events_emitted_counter = NamespacedCounter(
    "telemetry_events_emitted",
    "CloudEvents successfully placed on the emitter buffer",
    ["namespace", "type"],
)

telemetry_events_sampled_out_counter = NamespacedCounter(
    "telemetry_events_sampled_out",
    "CloudEvents intentionally dropped by HIGH_VOLUME_SAMPLE_RATE",
    ["namespace", "type"],
)

telemetry_events_dropped_counter = NamespacedCounter(
    "telemetry_events_dropped",
    "CloudEvents lost unintentionally (buffer_full or send_failed)",
    ["namespace", "reason"],
)

telemetry_buffer_size_gauge = NamespacedGauge(
    "telemetry_buffer_size",
    "Current size of the CloudEvents emitter buffer",
    ["namespace"],
)

# Embedding backlog: MessageEmbedding rows still awaiting a vector
# (sync_state='pending').
# region ai
# Distinct from embed_now_tasks_in_flight (in-flight fast-path work in the API
# process) — this is the durable, DB-wide backlog the reconciler drains. Every
# deriver replica refreshes it on its own timer from
# ReconcilerScheduler._scheduler_loop, so replicas disagree by at most one interval.
# Service-wide, not per-process — hence the help string's "never sum()".
# endregion
message_embeddings_pending_gauge = NamespacedGauge(
    "message_embeddings_pending",
    "MessageEmbedding rows awaiting embedding (sync_state='pending'). "
    + "Service-wide DB count, reported independently by every replica — "
    + "aggregate with max() or avg(), never sum()",
    ["namespace"],
)

# DB connection-pool health. The in-flight gauge counts statements actually
# executing on the wire, so checked_out minus in_flight reveals connections held
# but parked (the "idle in transaction during an external call" antipattern).
db_queries_in_flight_gauge = NamespacedGauge(
    "db_queries_in_flight",
    "DB statements currently executing on a connection for this instance",
    ["namespace", "instance_type"],
)


@final
class PrometheusMetrics:
    _instance: PrometheusMetrics | None = None

    def __new__(cls) -> PrometheusMetrics:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _handle_metric_error(self, method_name: str, error: Exception) -> None:
        import sentry_sdk

        sentry_sdk.capture_exception(error)
        logger.warning(
            "Failed to record Prometheus metric in %s: %s", method_name, str(error)
        )

    def record_api_request(
        self,
        *,
        method: str,
        endpoint: str,
        status_code: str,
        duration_seconds: float,
    ) -> None:
        try:
            api_requests_counter.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()
            api_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration_seconds)
        except Exception as e:
            self._handle_metric_error("record_api_request", e)

    def record_messages_created(
        self,
        *,
        count: int,
        workspace_name: str,
    ) -> None:
        try:
            messages_created_counter.labels(
                workspace_name=workspace_name,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_messages_created", e)

    def record_embed_now_task_shed(self) -> None:
        try:
            embed_now_tasks_shed_counter.labels().inc()
        except Exception as e:
            self._handle_metric_error("record_embed_now_task_shed", e)

    def set_embed_now_tasks_in_flight(self, count: int) -> None:
        try:
            embed_now_tasks_in_flight_gauge.labels().set(count)
        except Exception as e:
            self._handle_metric_error("set_embed_now_tasks_in_flight", e)

    def record_dialectic_call(
        self,
        *,
        workspace_name: str,
        reasoning_level: str,
    ) -> None:
        try:
            dialectic_calls_counter.labels(
                workspace_name=workspace_name,
                reasoning_level=reasoning_level,
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_dialectic_call", e)

    def record_deriver_queue_item(
        self,
        *,
        count: int,
        workspace_name: str,
        task_type: str,
    ) -> None:
        try:
            deriver_queue_items_processed_counter.labels(
                workspace_name=workspace_name,
                task_type=task_type,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_deriver_queue_item", e)

    def record_deriver_tokens(
        self,
        *,
        count: int,
        task_type: str,
        token_type: str,
        component: str,
    ) -> None:
        try:
            deriver_tokens_processed_counter.labels(
                task_type=task_type,
                token_type=token_type,
                component=component,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_deriver_tokens", e)

    def record_dialectic_tokens(
        self,
        *,
        count: int,
        token_type: str,
        component: str,
        reasoning_level: str,
    ) -> None:
        try:
            dialectic_tokens_processed_counter.labels(
                token_type=token_type,
                component=component,
                reasoning_level=reasoning_level,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_dialectic_tokens", e)

    def record_dreamer_tokens(
        self,
        *,
        count: int,
        specialist_name: str,
        token_type: str,
    ) -> None:
        try:
            dreamer_tokens_processed_counter.labels(
                specialist_name=specialist_name,
                token_type=token_type,
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_dreamer_tokens", e)

    def record_telemetry_event_emitted(self, *, event_type: str) -> None:
        try:
            telemetry_events_emitted_counter.labels(type=event_type).inc()
        except Exception as e:
            self._handle_metric_error("record_telemetry_event_emitted", e)

    def record_telemetry_event_sampled_out(self, *, event_type: str) -> None:
        try:
            telemetry_events_sampled_out_counter.labels(type=event_type).inc()
        except Exception as e:
            self._handle_metric_error("record_telemetry_event_sampled_out", e)

    def record_telemetry_event_dropped(self, *, reason: str) -> None:
        # Reason is one of "buffer_full" | "send_failed".
        try:
            telemetry_events_dropped_counter.labels(reason=reason).inc()
        except Exception as e:
            self._handle_metric_error("record_telemetry_event_dropped", e)

    def _touch(self, counter: NamespacedCounter, **labels: str) -> None:
        """Pre-create a counter child series at 0 without incrementing it."""
        # region ai
        # A labeled Prometheus counter exports no time series until its first
        # ``labels(...)`` call, so pre-touching a child keeps it present at 0 — a
        # missing series then signals a broken scrape rather than "no events".
        # Fail-soft (like the recorders): a bad init must never crash startup.
        # endregion
        try:
            counter.labels(**labels)
        except Exception as e:
            self._handle_metric_error("_touch", e)

    def initialize_telemetry_dropped_metrics(self, *, reasons: list[str]) -> None:
        """Pre-create telemetry_events_dropped ``(namespace, reason)`` children at 0.

        Args:
            reasons: The reason label values the calling emitter can produce.
        """
        # region ai
        # The metric stays invisible in Prometheus/Grafana until an event is actually
        # dropped, so materializing the children at startup keeps it present at 0 — a
        # missing series then means a broken scrape, not "no drops" (see _touch).
        #
        # Called per-emitter from ``TelemetryEmitter.start()`` rather than hoisted into
        # the process-level ``initialize_bounded_metrics``: the trace emitter (whose
        # reasons carry a ``trace_`` prefix) only exists when ``TRACE_PAYLOADS_ENABLED``
        # is set, so hoisting would fabricate ``trace_*`` series on deployments that run
        # with tracing off.
        # endregion
        if not settings.METRICS.ENABLED:
            return

        for reason in reasons:
            self._touch(telemetry_events_dropped_counter, reason=reason)

    def initialize_bounded_metrics(self, *, instance_type: str) -> None:
        """Pre-create bounded-label counter children at 0 for this process, so an
        absent series means a broken scrape rather than "nothing happened".

        Args:
            instance_type: "api" or "deriver" — selects the process-specific
                counters. Event-type and buffer metrics are initialized in both.
        """
        # region ai
        # A Prometheus counter does not exist until its first increment, so a
        # never-yet-incremented metric is indistinguishable from a broken scrape:
        # you cannot graph or alert on a series that is absent. Materializing the
        # children at 0 inverts that — a missing series now means something is wrong,
        # and "no events" reads as a flat 0 instead of a gap.
        #
        # That only holds for label sets we can enumerate honestly, so a metric is
        # initialized here only when its full label domain is bounded, enumerable at
        # startup, and actually emitted by THIS process. High-cardinality labels
        # (endpoint, workspace_name) and impossible label tuples are deliberately left
        # absent — fabricating a permanently-0 series that no code path can ever
        # increment is the same lie in the other direction.
        #
        # Multi-instance safety splits the metrics here into three buckets:
        #
        # 1. instance-scoped (``telemetry_buffer_size``, ``embed_now_tasks_in_flight``)
        #    — per-process by nature, so any aggregation is meaningful and zero-init is
        #    unambiguously right.
        # 2. service-scoped additive (the token counters, ``telemetry_events_emitted``)
        #    — each instance holds a partial count and ``sum()`` reconstructs the whole,
        #    so multi-instance safe.
        # 3. service-scoped non-additive — every instance reports the whole service's
        #    value, so the instances are N witnesses to one fact rather than N parts of
        #    one whole. ``sum()`` is therefore never correct here: it scales with the
        #    replica count. Scale-preserving aggregations (``max()``, ``avg()``,
        #    quantiles) ARE correct, but only while the witnesses disagree by a bounded
        #    amount — which requires every instance to refresh on its own timer (see
        #    ``message_embeddings_pending``, refreshed per replica from
        #    ``ReconcilerScheduler._scheduler_loop``). A bucket-3 metric that cannot
        #    meet that bar does not belong in the app at all — it belongs in an exporter
        #    that yields exactly one series.
        #
        # Prometheus stamps ``instance``/``job`` at scrape time, which is why buckets 1
        # and 2 need no special handling. ``telemetry_events_dropped`` is handled
        # separately, per-emitter, in ``TelemetryEmitter.start()`` (prefix-dependent).
        # endregion
        if not settings.METRICS.ENABLED:
            return

        # ai: lazy import avoids an import-time cycle (metrics is imported widely)
        from src.telemetry.events import ALL_EVENT_TYPES, HIGH_VOLUME_EVENT_TYPES

        # region ai
        # Common: both processes run a TelemetryEmitter, so both emit their own subset
        # of event types. The domain is bounded/low-cardinality (~21 types), so init
        # the full set in each process rather than maintain a fragile
        # per-event-type -> process map.
        # endregion
        for event_type in ALL_EVENT_TYPES:
            self._touch(telemetry_events_emitted_counter, type=event_type)
        for event_type in HIGH_VOLUME_EVENT_TYPES:
            self._touch(telemetry_events_sampled_out_counter, type=event_type)
        self.set_telemetry_buffer_size(size=0)

        if instance_type == "api":
            # dialectic tokens: token_type x component(total) x reasoning_level
            for token_type in TokenTypes:
                for level in REASONING_LEVELS:
                    self._touch(
                        dialectic_tokens_processed_counter,
                        token_type=token_type.value,
                        component=DialecticComponents.TOTAL.value,
                        reasoning_level=level,
                    )
            # ai: embed_now fast path runs as an API-process background task
            self._touch(embed_now_tasks_shed_counter)
            self.set_embed_now_tasks_in_flight(0)

        elif instance_type == "deriver":
            # deriver tokens: only the valid (token_type, component) tuples per
            # task_type (see _DERIVER_TOKEN_COMBOS_BY_TASK).
            for task_type_value, combos in _DERIVER_TOKEN_COMBOS_BY_TASK.items():
                for token_type_value, component_value in combos:
                    self._touch(
                        deriver_tokens_processed_counter,
                        task_type=task_type_value,
                        token_type=token_type_value,
                        component=component_value,
                    )
            # dreamer tokens: specialist_name x token_type.
            # region ai
            # Names come from the concrete BaseSpecialist subclasses (walked recursively
            # via walk_subclasses) so a new specialist can't silently miss init.
            # endregion
            from src.dreamer.specialists import BaseSpecialist

            for specialist in walk_subclasses(BaseSpecialist):
                for token_type in TokenTypes:
                    self._touch(
                        dreamer_tokens_processed_counter,
                        specialist_name=specialist.name,
                        token_type=token_type.value,
                    )
            # ai: init at 0 so the gauge is visible before its first per-replica refresh
            self.set_message_embeddings_pending(count=0)

    def set_telemetry_buffer_size(self, *, size: int) -> None:
        try:
            telemetry_buffer_size_gauge.labels().set(size)
        except Exception as e:
            self._handle_metric_error("set_telemetry_buffer_size", e)

    def set_message_embeddings_pending(self, *, count: int) -> None:
        try:
            message_embeddings_pending_gauge.labels().set(count)
        except Exception as e:
            self._handle_metric_error("set_message_embeddings_pending", e)


prometheus_metrics = PrometheusMetrics()


class DBPoolCollector:
    """Scrape-time collector for SQLAlchemy connection-pool stats.

    Computed live on each /metrics scrape from the async engine's pool, so it
    is always current with no background task or sampling lag. One instance is
    registered per process (the API server or a deriver worker).
    """

    def __init__(self, instance_type: str) -> None:
        # instance_type: "api" | "deriver"
        self.instance_type: str = instance_type

    def collect(self) -> Iterator[GaugeMetricFamily]:
        namespace = settings.METRICS.NAMESPACE or ""
        gauge = GaugeMetricFamily(
            "db_pool_connections",
            "DB connections held by this instance, by pool state",
            labels=["namespace", "instance_type", "state"],
        )
        # Fail soft: Prometheus aborts the entire scrape (dropping ALL metrics)
        # if any collector raises, so never let a pool/import hiccup here sink
        # the whole /metrics response.
        try:
            # Lazy import to avoid an import cycle at module load (db imports
            # config, telemetry is imported widely). Reads engine.pool directly.
            from src.db import get_pool_stats

            stats = get_pool_stats()
        except Exception:
            logger.warning("Failed to collect DB pool stats", exc_info=True)
            stats = {}
        for state, value in stats.items():
            gauge.add_metric([namespace, self.instance_type, state], value)
        yield gauge


_db_pool_collector_registered = False


def register_db_pool_collector(instance_type: str) -> None:
    """Register the DB pool collector once per process (no-op if metrics off)."""
    global _db_pool_collector_registered
    if _db_pool_collector_registered or not settings.METRICS.ENABLED:
        return
    REGISTRY.register(DBPoolCollector(instance_type))
    _db_pool_collector_registered = True


async def metrics_endpoint(_request: Request) -> Response:
    if not settings.METRICS.ENABLED:
        return Response("Metrics are disabled", status_code=404)
    try:
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}", exc_info=True)
        return Response("Failed to generate metrics", status_code=500)
