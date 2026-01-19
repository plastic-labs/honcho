"""
Telemetry module for Honcho.

This module consolidates all telemetry, metrics, and observability functionality:
- Sentry: Error tracking and performance tracing
- Prometheus: Pull-based metrics
- Logging: Langfuse integration, Rich console output, metric accumulation
- Tracing: Sentry transaction decorators
- Metrics Collector: JSON file-based benchmark aggregation
- Reasoning Traces: JSONL logging of LLM inputs/outputs
"""
