"""
Telemetry module for Honcho.

This module consolidates all telemetry, metrics, and observability functionality:
- Sentry: Error tracking and performance tracing
- Prometheus: Pull-based metrics scraped by Fly.io
- CloudEvents: Structured events for analytics (push-based)
- Logging: Langfuse integration, Rich console output, metric accumulation
- Tracing: Sentry transaction decorators
- Metrics Collector: JSON file-based benchmark aggregation
- Reasoning Traces: JSONL logging of LLM inputs/outputs
"""

from src.telemetry.events import emit
from src.telemetry.prometheus import metrics_endpoint, prometheus_metrics

__all__ = [
    "emit",
    "initialize_telemetry_async",
    "metrics_endpoint",
    "prometheus_metrics",
    "shutdown_telemetry",
]


async def initialize_telemetry_async() -> None:
    """
    Initialize async telemetry systems based on configuration.

    This should be called once at application startup (in main.py lifespan).
    It initializes:
    - CloudEvents emitter (if TELEMETRY_ENABLED=true)

    Note: Prometheus metrics are pull-based and require no initialization.
    Sentry is initialized separately in sentry.py as it has its own lifecycle.
    """
    from src.config import settings
    from src.telemetry.events import initialize_telemetry_events

    if settings.TELEMETRY.ENABLED:
        await initialize_telemetry_events()


async def shutdown_telemetry() -> None:
    """
    Shutdown all telemetry systems gracefully.

    This should be called during application shutdown to ensure:
    - CloudEvents buffer is flushed
    """
    from src.telemetry.events import shutdown_telemetry_events

    # Shutdown CloudEvents emitter (flushes buffer)
    await shutdown_telemetry_events()
