import os
import tempfile

from dotenv import load_dotenv
from starlette.responses import Response as StarletteResponse

from src.config import settings

load_dotenv()

METRICS_ENABLED = settings.METRICS.ENABLED

PROMETHEUS_MULTIPROC_DIR = os.environ.get(
    "PROMETHEUS_MULTIPROC_DIR",
    os.path.join(tempfile.gettempdir(), "prometheus_multiproc_dir_honcho"),
)
os.environ["PROMETHEUS_MULTIPROC_DIR"] = PROMETHEUS_MULTIPROC_DIR
os.makedirs(PROMETHEUS_MULTIPROC_DIR, exist_ok=True)

from prometheus_client import (  # noqa: E402
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    generate_latest,
    multiprocess,
)

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry, path=PROMETHEUS_MULTIPROC_DIR)


class NamespacedCounter(Counter):
    """Counter that automatically includes namespace label"""

    def labels(self, **kwargs: str) -> "NamespacedCounter":
        """Override labels to automatically include namespace"""
        kwargs["namespace"] = settings.METRICS.NAMESPACE
        return super().labels(**kwargs)


# Metrics (register with the default registry; do NOT attach to `registry`)
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

QUEUE_ITEMS_CREATED = NamespacedCounter(
    "queue_items_created_total",
    "Total queue items created",
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
    return StarletteResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
