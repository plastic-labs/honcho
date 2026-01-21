"""
Telemetry module for Honcho.

This module consolidates all telemetry, metrics, and observability functionality:
- Sentry: Error tracking and performance tracing
- OTel: Push-based metrics via OTLP to any compatible backend (e.g., Mimir)
- CloudEvents: Structured events for analytics (push-based)
- Logging: Langfuse integration, Rich console output, metric accumulation
- Tracing: Sentry transaction decorators
- Metrics Collector: JSON file-based benchmark aggregation
- Reasoning Traces: JSONL logging of LLM inputs/outputs
"""

from src.telemetry.events import emit
from src.telemetry.otel import get_meter, initialize_otel_metrics, shutdown_otel_metrics
from src.telemetry.otel.metrics import otel_metrics

__all__ = [
    "emit",
    "get_meter",
    "initialize_otel_metrics",
    "initialize_telemetry",
    "initialize_telemetry_async",
    "otel_metrics",
    "shutdown_otel_metrics",
    "shutdown_telemetry",
]


def initialize_telemetry() -> None:
    """
    Initialize all telemetry systems based on configuration.

    This should be called once at application startup (in main.py lifespan).
    It reads configuration from settings and initializes:
    - OTel metrics (if OTEL_ENABLED=true)

    Note: CloudEvents telemetry requires async initialization and should be
    initialized separately using initialize_telemetry_async().

    Sentry is initialized separately in sentry.py as it has its own lifecycle.
    """
    from src.config import settings

    if settings.OTEL.ENABLED:
        initialize_otel_metrics(
            endpoint=settings.OTEL.ENDPOINT,
            headers=settings.OTEL.HEADERS,
            export_interval_millis=settings.OTEL.EXPORT_INTERVAL_MILLIS,
            service_name=settings.OTEL.SERVICE_NAME,
            service_namespace=settings.OTEL.SERVICE_NAMESPACE,
            enabled=True,
        )


async def initialize_telemetry_async() -> None:
    """
    Initialize async telemetry systems based on configuration.

    This should be called once at application startup (in main.py lifespan),
    after initialize_telemetry(). It initializes:
    - CloudEvents emitter (if TELEMETRY_ENABLED=true)
    """
    from src.config import settings
    from src.telemetry.events import initialize_telemetry_events

    if settings.TELEMETRY.ENABLED:
        await initialize_telemetry_events()


async def shutdown_telemetry() -> None:
    """
    Shutdown all telemetry systems gracefully.

    This should be called during application shutdown to ensure:
    - OTel metrics are flushed
    - CloudEvents buffer is flushed
    """
    from src.telemetry.events import shutdown_telemetry_events

    # Shutdown CloudEvents emitter (flushes buffer)
    await shutdown_telemetry_events()

    # Shutdown OTel metrics
    shutdown_otel_metrics()
