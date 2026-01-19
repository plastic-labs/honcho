"""
OpenTelemetry metrics module for Honcho.

This module provides push-based metrics using OpenTelemetry SDK
with Prometheus Remote Write export to Mimir.
"""

from src.telemetry.otel.metrics import (
    get_meter,
    initialize_otel_metrics,
    shutdown_otel_metrics,
)

__all__ = [
    "get_meter",
    "initialize_otel_metrics",
    "shutdown_otel_metrics",
]
