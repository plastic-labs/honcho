"""
OpenTelemetry metrics implementation with OTLP export.

This module provides push-based metrics that replace the pull-based Prometheus
/metrics endpoint. Metrics are pushed via OTLP to any compatible backend
(Mimir, Grafana Cloud, etc.) at configurable intervals.

Usage:
    from src.telemetry.otel import get_meter, initialize_otel_metrics

    # Initialize once at startup
    initialize_otel_metrics()

    # Get a meter for your component
    meter = get_meter("honcho.deriver")

    # Create instruments
    counter = meter.create_counter("tokens_processed", unit="tokens")
    counter.add(100, {"task_type": "ingestion"})
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING, final

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Meter

logger = logging.getLogger(__name__)

# Global state
_meter_provider: MeterProvider | None = None
_initialized: bool = False


def initialize_otel_metrics(
    *,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    export_interval_millis: int = 60000,
    service_name: str = "honcho",
    service_namespace: str | None = None,
    enabled: bool = True,
) -> None:
    """
    Initialize OpenTelemetry metrics with OTLP export.

    This should be called once at application startup. If already initialized,
    subsequent calls are no-ops.

    Args:
        endpoint: OTLP HTTP endpoint URL (e.g., "https://mimir.example.com/otlp/v1/metrics").
                  If None, metrics are collected but not exported (useful for testing).
        headers: Optional headers to include in requests (e.g., {"X-Scope-OrgID": "tenant"}).
        export_interval_millis: How often to export metrics (default: 60 seconds).
        service_name: Service name for resource attributes (default: "honcho").
        service_namespace: Optional namespace for the service.
        enabled: If False, metrics are no-ops (default: True).
    """
    global _meter_provider, _initialized

    if _initialized:
        logger.debug("OTel metrics already initialized, skipping")
        return

    if not enabled:
        logger.info("OTel metrics disabled")
        _initialized = True
        return

    # Build resource attributes
    resource_attributes = {
        "service.name": service_name,
    }
    if service_namespace:
        resource_attributes["service.namespace"] = service_namespace

    resource = Resource.create(resource_attributes)

    # Create metric reader
    readers: list[PeriodicExportingMetricReader] = []

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )

            exporter = OTLPMetricExporter(
                endpoint=endpoint,
                headers=headers or {},
            )
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=export_interval_millis,
            )
            readers.append(reader)
            logger.info(f"OTel metrics configured to push via OTLP to {endpoint}")
        except Exception as e:
            logger.error(f"Failed to configure OTLP metrics exporter: {e}")
            # Continue without exporter - metrics still work locally
    else:
        logger.info(
            "OTel metrics initialized without remote export (no endpoint configured)"
        )

    # Create and set the meter provider
    # Note: empty list is valid for metric_readers (metrics still work, just not exported)
    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=readers,
    )
    metrics.set_meter_provider(_meter_provider)

    # Register shutdown handler
    atexit.register(shutdown_otel_metrics)

    _initialized = True
    logger.info("OTel metrics initialized successfully")


def shutdown_otel_metrics() -> None:
    """
    Shutdown the OTel metrics provider, flushing any pending metrics.

    This is automatically called at process exit via atexit, but can be
    called manually for graceful shutdown.
    """
    global _meter_provider, _initialized

    if _meter_provider is not None:
        try:
            _meter_provider.shutdown()
            logger.info("OTel metrics shutdown complete")
        except Exception as e:
            logger.error(f"Error during OTel metrics shutdown: {e}")
        finally:
            _meter_provider = None
            _initialized = False


def get_meter(name: str, version: str = "") -> Meter:
    """
    Get an OTel Meter for creating instruments.

    Args:
        name: The name of the instrumentation scope (e.g., "honcho.deriver").
        version: Optional version of the instrumentation scope.

    Returns:
        An OTel Meter instance for creating counters, histograms, etc.

    Example:
        meter = get_meter("honcho.deriver")
        counter = meter.create_counter("tokens_processed", unit="tokens")
        counter.add(100, {"task_type": "ingestion"})
    """
    return metrics.get_meter(name, version)


# =============================================================================
# Pre-defined metrics that mirror existing Prometheus counters
# =============================================================================

# These are created lazily on first use to avoid issues with initialization order


@final
class OTelMetrics:
    """
    Container for OTel metrics that mirror existing Prometheus counters.

    This class provides a bridge during migration - the same metrics are
    available via both Prometheus (pull) and OTel (push).
    """

    _instance: OTelMetrics | None = None
    _is_initialized: bool = False

    # Meters (lazily initialized)
    _api_meter: Meter | None = None
    _deriver_meter: Meter | None = None
    _dialectic_meter: Meter | None = None
    _dreamer_meter: Meter | None = None

    # Counters (lazily initialized)
    _api_requests: Counter | None = None
    _messages_created: Counter | None = None
    _dialectic_calls: Counter | None = None
    _deriver_queue_items: Counter | None = None
    _deriver_tokens: Counter | None = None
    _dialectic_tokens: Counter | None = None
    _dreamer_tokens: Counter | None = None

    def __new__(cls) -> OTelMetrics:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_initialized(self) -> None:
        """Lazily initialize meters and instruments."""
        if self._is_initialized:
            return

        # Get meters for different components
        self._api_meter = get_meter("honcho.api")
        self._deriver_meter = get_meter("honcho.deriver")
        self._dialectic_meter = get_meter("honcho.dialectic")
        self._dreamer_meter = get_meter("honcho.dreamer")

        # Create counters that mirror Prometheus metrics
        # API requests
        self._api_requests = self._api_meter.create_counter(
            name="api_requests",
            unit="requests",
            description="Total API requests",
        )

        # Messages created
        self._messages_created = self._api_meter.create_counter(
            name="messages_created",
            unit="messages",
            description="Total messages created",
        )

        # Dialectic calls
        self._dialectic_calls = self._dialectic_meter.create_counter(
            name="dialectic_calls",
            unit="calls",
            description="Total dialectic calls",
        )

        # Deriver queue items processed
        self._deriver_queue_items = self._deriver_meter.create_counter(
            name="deriver_queue_items_processed",
            unit="items",
            description="Total deriver queue items processed",
        )

        # Token counters
        self._deriver_tokens = self._deriver_meter.create_counter(
            name="deriver_tokens_processed",
            unit="tokens",
            description="Total tokens processed by the deriver",
        )

        self._dialectic_tokens = self._dialectic_meter.create_counter(
            name="dialectic_tokens_processed",
            unit="tokens",
            description="Total tokens processed by the dialectic",
        )

        self._dreamer_tokens = self._dreamer_meter.create_counter(
            name="dreamer_tokens_processed",
            unit="tokens",
            description="Total tokens processed by the dreamer",
        )

        self._is_initialized = True

    def record_api_request(
        self,
        *,
        method: str,
        endpoint: str,
        status_code: str,
        namespace: str,
    ) -> None:
        """Record an API request metric."""
        self._ensure_initialized()
        if self._api_requests is None:
            return  # Not initialized, skip silently
        self._api_requests.add(
            1,
            {
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "namespace": namespace,
            },
        )

    def record_messages_created(
        self,
        *,
        count: int,
        workspace_name: str,
        namespace: str,
    ) -> None:
        """Record messages created metric."""
        self._ensure_initialized()
        if self._messages_created is None:
            return  # Not initialized, skip silently
        self._messages_created.add(
            count,
            {
                "workspace_name": workspace_name,
                "namespace": namespace,
            },
        )

    def record_dialectic_call(
        self,
        *,
        workspace_name: str,
        reasoning_level: str,
        namespace: str,
    ) -> None:
        """Record a dialectic call metric."""
        self._ensure_initialized()
        if self._dialectic_calls is None:
            return  # Not initialized, skip silently
        self._dialectic_calls.add(
            1,
            {
                "workspace_name": workspace_name,
                "reasoning_level": reasoning_level,
                "namespace": namespace,
            },
        )

    def record_deriver_queue_item(
        self,
        *,
        workspace_name: str,
        task_type: str,
        namespace: str,
    ) -> None:
        """Record a deriver queue item processed metric."""
        self._ensure_initialized()
        if self._deriver_queue_items is None:
            return  # Not initialized, skip silently
        self._deriver_queue_items.add(
            1,
            {
                "workspace_name": workspace_name,
                "task_type": task_type,
                "namespace": namespace,
            },
        )

    def record_deriver_tokens(
        self,
        *,
        count: int,
        task_type: str,
        token_type: str,
        component: str,
        namespace: str,
    ) -> None:
        """Record deriver token usage metric."""
        self._ensure_initialized()
        if self._deriver_tokens is None:
            return  # Not initialized, skip silently
        self._deriver_tokens.add(
            count,
            {
                "task_type": task_type,
                "token_type": token_type,
                "component": component,
                "namespace": namespace,
            },
        )

    def record_dialectic_tokens(
        self,
        *,
        count: int,
        token_type: str,
        component: str,
        reasoning_level: str,
        namespace: str,
    ) -> None:
        """Record dialectic token usage metric."""
        self._ensure_initialized()
        if self._dialectic_tokens is None:
            return  # Not initialized, skip silently
        self._dialectic_tokens.add(
            count,
            {
                "token_type": token_type,
                "component": component,
                "reasoning_level": reasoning_level,
                "namespace": namespace,
            },
        )

    def record_dreamer_tokens(
        self,
        *,
        count: int,
        specialist_name: str,
        token_type: str,
        namespace: str,
    ) -> None:
        """Record dreamer token usage metric."""
        self._ensure_initialized()
        if self._dreamer_tokens is None:
            return  # Not initialized, skip silently
        self._dreamer_tokens.add(
            count,
            {
                "specialist_name": specialist_name,
                "token_type": token_type,
                "namespace": namespace,
            },
        )


# Singleton instance
otel_metrics = OTelMetrics()
