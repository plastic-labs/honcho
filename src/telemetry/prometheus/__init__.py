"""
Prometheus telemetry module.

Exports:
- prometheus_metrics: Singleton for recording metrics
- metrics_endpoint: Async endpoint for /metrics route
- Label enums for metric values
"""

from src.telemetry.prometheus.metrics import (
    DeriverComponents,
    DeriverTaskTypes,
    DialecticComponents,
    TokenTypes,
    metrics_endpoint,
    prometheus_metrics,
)

__all__ = [
    "DeriverComponents",
    "DeriverTaskTypes",
    "DialecticComponents",
    "TokenTypes",
    "metrics_endpoint",
    "prometheus_metrics",
]
