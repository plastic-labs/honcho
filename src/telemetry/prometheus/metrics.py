"""
Prometheus metrics implementation
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, final

from prometheus_client import Counter, generate_latest
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =============================================================================
# Metric label enums
# =============================================================================


class TokenTypes(Enum):
    INPUT = "input"
    OUTPUT = "output"


class DeriverTaskTypes(Enum):
    INGESTION = "ingestion"
    SUMMARY = "summary"


class DeriverComponents(Enum):
    PROMPT = "prompt"  # used in ingestion and summary
    MESSAGES = "messages"  # used in ingestion and summary
    PREVIOUS_SUMMARY = "previous_summary"  # only used for summary
    OUTPUT_TOTAL = "output_total"


class DialecticComponents(Enum):
    TOTAL = "total"


# =============================================================================
# Prometheus Counter Definitions
# =============================================================================

# API requests counter
api_requests_counter = Counter(
    "api_requests",
    "Total API requests",
    ["method", "endpoint", "status_code", "namespace"],
)

# Messages created counter
messages_created_counter = Counter(
    "messages_created",
    "Total messages created",
    ["workspace_name", "namespace"],
)

# Dialectic calls counter
dialectic_calls_counter = Counter(
    "dialectic_calls",
    "Total dialectic calls",
    ["workspace_name", "reasoning_level", "namespace"],
)

# Deriver queue items processed counter
deriver_queue_items_processed_counter = Counter(
    "deriver_queue_items_processed",
    "Total deriver queue items processed",
    ["workspace_name", "task_type", "namespace"],
)

# Token counters
deriver_tokens_processed_counter = Counter(
    "deriver_tokens_processed",
    "Total tokens processed by the deriver",
    ["task_type", "token_type", "component", "namespace"],
)

dialectic_tokens_processed_counter = Counter(
    "dialectic_tokens_processed",
    "Total tokens processed by the dialectic",
    ["token_type", "component", "reasoning_level", "namespace"],
)

dreamer_tokens_processed_counter = Counter(
    "dreamer_tokens_processed",
    "Total tokens processed by the dreamer",
    ["specialist_name", "token_type", "namespace"],
)


# =============================================================================
# Prometheus Metrics Class
# =============================================================================


@final
class PrometheusMetrics:
    """
    Container for Prometheus metrics.

    Namespace is managed at the instance level (from settings).
    """

    _instance: PrometheusMetrics | None = None
    _namespace: str = "honcho"

    def __new__(cls) -> PrometheusMetrics:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_namespace(self) -> str:
        """Get the namespace from settings lazily."""
        from src.config import settings

        return settings.PROMETHEUS.NAMESPACE or "honcho"

    def _handle_metric_error(self, method_name: str, error: Exception) -> None:
        """Handle errors from metric recording by logging to Sentry."""
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
        """Record an API request metric."""
        try:
            api_requests_counter.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                namespace=self._get_namespace(),
            ).inc()
        except Exception as e:
            self._handle_metric_error("record_api_request", e)

    def record_messages_created(
        self,
        *,
        count: int,
        workspace_name: str,
    ) -> None:
        """Record messages created metric."""
        try:
            messages_created_counter.labels(
                workspace_name=workspace_name,
                namespace=self._get_namespace(),
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_messages_created", e)

    def record_dialectic_call(
        self,
        *,
        workspace_name: str,
        reasoning_level: str,
    ) -> None:
        """Record a dialectic call metric."""
        try:
            dialectic_calls_counter.labels(
                workspace_name=workspace_name,
                reasoning_level=reasoning_level,
                namespace=self._get_namespace(),
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
        """Record deriver queue items processed metric."""
        try:
            deriver_queue_items_processed_counter.labels(
                workspace_name=workspace_name,
                task_type=task_type,
                namespace=self._get_namespace(),
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
        """Record deriver token usage metric."""
        try:
            deriver_tokens_processed_counter.labels(
                task_type=task_type,
                token_type=token_type,
                component=component,
                namespace=self._get_namespace(),
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
        """Record dialectic token usage metric."""
        try:
            dialectic_tokens_processed_counter.labels(
                token_type=token_type,
                component=component,
                reasoning_level=reasoning_level,
                namespace=self._get_namespace(),
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
        """Record dreamer token usage metric."""
        try:
            dreamer_tokens_processed_counter.labels(
                specialist_name=specialist_name,
                token_type=token_type,
                namespace=self._get_namespace(),
            ).inc(count)
        except Exception as e:
            self._handle_metric_error("record_dreamer_tokens", e)


# Singleton instance
prometheus_metrics = PrometheusMetrics()


# =============================================================================
# Metrics Endpoint
# =============================================================================


async def metrics_endpoint(_request: Request) -> Response:
    """
    Async endpoint that returns Prometheus metrics in text format.

    This is designed to be mounted as a route in FastAPI/Starlette.
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
