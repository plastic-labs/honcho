"""SSE (Server-Sent Events) parsing utilities."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator

logger = logging.getLogger(__name__)


def parse_sse_chunk(chunk: bytes) -> Generator[str, None, None]:
    """
    Parse an SSE chunk and yield content strings.

    Handles the SSE data format: `data: {...json...}`
    Stops yielding when a `done: true` message is received.

    Args:
        chunk: Raw bytes from the SSE stream

    Yields:
        Content strings extracted from delta objects
    """
    for line in chunk.decode("utf-8").split("\n"):
        if line.startswith("data: "):
            json_str = line[6:]  # Remove "data: " prefix
            try:
                chunk_data = json.loads(json_str)
                if chunk_data.get("done"):
                    return
                delta_obj = chunk_data.get("delta", {})
                content = delta_obj.get("content")
                if content:
                    yield content
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to decode streaming chunk: %s (data: %s)",
                    e,
                    json_str[:100],
                )
                continue
