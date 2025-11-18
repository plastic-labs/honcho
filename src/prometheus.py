"""
Prometheus metrics for Honcho.

This module defines all Prometheus metrics for all Honcho processes and exposes them via the /metrics endpoint.
"""

import logging
from typing import cast

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

logger = logging.getLogger(__name__)


class NamespacedCounter(Counter):
    """Counter that automatically includes namespace label"""

    def labels(self, **kwargs: str) -> "NamespacedCounter":
        """Override labels to automatically appends namespace label"""
        # METRICS.NAMESPACE is guaranteed to be non-None by AppSettings.propagate_namespace validator
        kwargs["namespace"] = cast(str, settings.METRICS.NAMESPACE)
        return super().labels(**kwargs)


# Tracks all requests to the Honcho API.
#
# Incremented in: src/main.py middleware for every request
# Labels:
#   - method: HTTP method (GET, POST, PUT, DELETE, etc.)
#   - endpoint: FastAPI route template (e.g., "/v2/workspaces/{workspace_id}/sessions/{session_id}/messages")
#   - status_code: HTTP response status code (200, 404, 500, etc.)
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

# Tracks the total number of honcho messages created.
#
# Incremented in: src/routers/messages.py when messages are successfully created
# Labels:
#   - workspace_name: The workspace where messages were created
MESSAGES_CREATED = NamespacedCounter(
    "messages_created_total",
    "Total messages created",
    [
        "namespace",
        "workspace_name",
    ],
)


# Tracks the total number of dialectic calls made.
#
# Incremented in: src/routers/peers.py when successful dialectic calls are made
# Labels:
#   - workspace_name: The workspace where the dialectic call was made
DIALECTIC_CALLS = NamespacedCounter(
    "dialectic_calls_total",
    "Total dialectic calls",
    [
        "namespace",
        "workspace_name",
    ],
)

# Tracks the total number of queue items successfully processed by the deriver.
#
# Incremented in: src/deriver/queue_manager.py when queue items are processed
# Labels:
#   - workspace_name: The workspace where items were processed
#   - task_type: The type of task processed (e.g., "representation", "summary")
DERIVER_QUEUE_ITEMS_PROCESSED = NamespacedCounter(
    "deriver_queue_items_processed_total",
    "Total deriver queue items processed",
    ["namespace", "workspace_name", "task_type"],
)

# Tracks the total number of input and output tokens processed by the deriver.
#
# Incremented in: src/deriver/deriver.py after the critical analysis call is made
# Labels:
#   - task_type: The type of task that processed the tokens (e.g., "representation")
#   - token_type: The type of tokens ("input" or "output")
DERIVER_TOKENS_PROCESSED = NamespacedCounter(
    "deriver_tokens_processed_total",
    "Total tokens processed by the deriver",
    [
        "namespace",
        "task_type",
        "token_type",
    ],
)

# Tracks the total number of input and output tokens processed by the dialectic.
#
# Incremented in: src/dialectic/chat.py after the dialectic call is made
# Labels:
#   - token_type: The type of tokens ("input" or "output")
DIALECTIC_TOKENS_PROCESSED = NamespacedCounter(
    "dialectic_tokens_processed_total",
    "Total tokens processed by the dialectic",
    [
        "namespace",
        "token_type",
    ],
)


async def metrics() -> StarletteResponse:
    """Prometheus metrics endpoint"""
    if not settings.METRICS.ENABLED:
        return StarletteResponse("Metrics are disabled", status_code=404)
    try:
        return StarletteResponse(
            generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}", exc_info=True)
        return StarletteResponse("Failed to generate metrics", status_code=500)
