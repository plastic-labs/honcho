"""
Metrics enums used by OpenTelemetry metrics.

This module contains enum definitions for metric labels. These enums are shared
across the telemetry system and used by both the deriver and dialectic metrics.

Note: The prometheus_client Counters and /metrics endpoint have been removed.
Metrics are now pushed via OpenTelemetry to an OTLP-compatible backend (e.g., Mimir).
"""

from enum import Enum


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
