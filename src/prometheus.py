from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    disable_created_metrics,
    generate_latest,
)
from starlette.responses import Response as StarletteResponse

from src.config import settings

METRICS_ENABLED = settings.METRICS.ENABLED
disable_created_metrics()  # Disables _created metrics on counters, histograms, and summaries


class NamespacedCounter(Counter):
    """Counter that automatically includes namespace label"""

    def labels(self, **kwargs: str) -> "NamespacedCounter":
        """Override labels to automatically appends namespace label"""
        kwargs["namespace"] = settings.METRICS.NAMESPACE
        return super().labels(**kwargs)


API_REQUESTS = NamespacedCounter(
    "api_requests_total",
    "Total API requests",
    [
        "namespace",
        "method",
        "endpoint",
        "status_code",
    ],
)

MESSAGES_CREATED = NamespacedCounter(
    "messages_created_total",
    "Total messages created",
    [
        "namespace",
        "workspace_name",
    ],
)

MESSAGE_INPUT_TOKENS = NamespacedCounter(
    "message_input_tokens_total",
    "Total message input tokens",
    ["namespace", "workspace_name"],
)

DIALECTIC_CALLS = NamespacedCounter(
    "dialectic_calls_total",
    "Total dialectic calls",
    [
        "namespace",
        "workspace_name",
    ],
)

DERIVER_TASKS_COMPLETED = NamespacedCounter(
    "deriver_tasks_completed_total",
    "Total deriver tasks completed",
    ["namespace", "workspace_name", "task_type"],
)

DERIVER_TOKENS_PROCESSED = NamespacedCounter(
    "tokens_processed_total",
    "Total tokens processed",
    [
        "namespace",
        "task_type",
    ],
)


async def metrics() -> StarletteResponse:
    """Prometheus metrics endpoint"""
    return StarletteResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
