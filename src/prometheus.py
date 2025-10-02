from dotenv import load_dotenv
from starlette.responses import Response as StarletteResponse

from src.config import settings

load_dotenv()

METRICS_ENABLED = settings.METRICS.ENABLED


from prometheus_client import (  # noqa: E402
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    generate_latest,
)


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
    ["namespace", "workspace_name", "session_name"],
)

QUEUE_ITEM_PROCESSED = NamespacedCounter(
    "queue_items_processed_total",
    "Total queue items processed",
    ["namespace", "workspace_name", "session_name", "task_type"],
)

DERIVER_LLM_TOKENS = NamespacedCounter(
    "deriver_llm_tokens_total",
    "Total LLM tokens consumed",
    ["namespace", "task_type", "model", "direction"],
)


def get_namespace() -> str:
    """Get the configured namespace for metrics labeling"""
    return settings.METRICS.NAMESPACE


async def metrics() -> StarletteResponse:
    """Prometheus metrics endpoint"""
    return StarletteResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
