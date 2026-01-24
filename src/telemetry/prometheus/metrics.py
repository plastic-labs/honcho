"""Prometheus metrics for Honcho."""

from __future__ import annotations

import logging
from enum import Enum
from typing import cast, final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    disable_created_metrics,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

from src.config import settings

disable_created_metrics()

logger = logging.getLogger(__name__)


class NamespacedCounter(Counter):
    def labels(self, **kwargs: str) -> NamespacedCounter:
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


api_requests_counter = NamespacedCounter(
    "api_requests",
    "Total API requests",
    ["namespace", "method", "endpoint", "status_code"],
)

messages_created_counter = NamespacedCounter(
    "messages_created",
    "Total messages created",
    ["namespace", "workspace_name"],
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
    ) -> None:
        try:
            api_requests_counter.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()
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


prometheus_metrics = PrometheusMetrics()


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
