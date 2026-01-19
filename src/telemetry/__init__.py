"""
Telemetry module for Honcho.

This module consolidates all telemetry, metrics, and observability functionality:
- Sentry: Error tracking and performance tracing
- Prometheus: Pull-based metrics (legacy, being replaced by OTel)
- OTel: Push-based metrics via Prometheus Remote Write
- Logging: Langfuse integration, Rich console output, metric accumulation
- Tracing: Sentry transaction decorators
- Metrics Collector: JSON file-based benchmark aggregation
- Reasoning Traces: JSONL logging of LLM inputs/outputs
"""

from src.telemetry.otel import get_meter, initialize_otel_metrics, shutdown_otel_metrics
from src.telemetry.otel.metrics import otel_metrics

__all__ = [
    "get_meter",
    "initialize_otel_metrics",
    "shutdown_otel_metrics",
    "otel_metrics",
    "initialize_telemetry",
]


def initialize_telemetry() -> None:
    """
    Initialize all telemetry systems based on configuration.

    This should be called once at application startup (in main.py lifespan).
    It reads configuration from settings and initializes:
    - OTel metrics (if OTEL_ENABLED=true)

    Sentry is initialized separately in sentry.py as it has its own lifecycle.
    """
    from src.config import settings

    if settings.OTEL.ENABLED:
        initialize_otel_metrics(
            endpoint=settings.OTEL.ENDPOINT,
            headers=settings.OTEL.HEADERS,
            export_interval_millis=settings.OTEL.EXPORT_INTERVAL_MILLIS,
            service_name=settings.OTEL.SERVICE_NAME,
            service_namespace=settings.METRICS.NAMESPACE,
            enabled=True,
        )
