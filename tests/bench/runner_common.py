"""
Shared utilities for Honcho benchmark test runners.

Contains common functionality for queue management, dream triggering,
Honcho client creation, and CLI argument parsing used across longmem,
beam, and locomo runners.
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import redis.asyncio as aioredis
from anthropic import AsyncAnthropic
from honcho import Honcho
from openai import AsyncOpenAI
from redis.asyncio.client import Redis

from src.telemetry.metrics_collector import MetricsCollector

# Valid reasoning levels for dialectic chat
ReasoningLevel = Literal["minimal", "low", "medium", "high", "max"]
REASONING_LEVELS: list[str] = ["minimal", "low", "medium", "high", "max"]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common command line arguments shared across all benchmark runners.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--base-api-port",
        type=int,
        default=8000,
        help="Base port for Honcho API instances (default: 8000)",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=1,
        help="Number of Honcho instances in the pool (default: 1)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for deriver queue to empty in seconds (default: 10 minutes)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of items to run concurrently in each batch (default: 10)",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to write JSON summary results for analytics (optional)",
    )

    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after executing each test (default: False)",
    )

    parser.add_argument(
        "--use-get-context",
        action="store_true",
        help="Use get_context + judge LLM instead of dialectic .chat endpoint (default: False)",
    )

    parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://localhost:6379/0",
        help="Redis URL for flush mode signaling (default: redis://localhost:6379/0)",
    )

    parser.add_argument(
        "--reasoning-level",
        type=str,
        choices=REASONING_LEVELS,
        default=None,
        help="Reasoning level for dialectic chat: minimal, low, medium, high, max (default: None)",
    )


def validate_common_arguments(args: argparse.Namespace) -> str | None:
    """
    Validate common command line arguments.

    Args:
        args: Parsed arguments

    Returns:
        Error message if validation fails, None otherwise
    """
    if args.batch_size <= 0:
        return f"Error: Batch size must be positive, got {args.batch_size}"

    if args.pool_size <= 0:
        return f"Error: Pool size must be positive, got {args.pool_size}"

    return None


def configure_logging() -> logging.Logger:
    """
    Configure logging for benchmark runners.

    Sets up logging with WARNING level and suppresses HTTP request logs.

    Returns:
        Logger instance for the calling module
    """
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Suppress HTTP request logs from the Honcho SDK
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    return logging.getLogger(__name__)


def create_anthropic_client(api_key: str | None = None) -> AsyncAnthropic:
    """
    Create an AsyncAnthropic client.

    Args:
        api_key: Optional API key. If not provided, uses LLM_ANTHROPIC_API_KEY env var.

    Returns:
        AsyncAnthropic client instance

    Raises:
        ValueError: If no API key is available
    """
    if api_key:
        return AsyncAnthropic(api_key=api_key)

    env_key = os.getenv("LLM_ANTHROPIC_API_KEY")
    if not env_key:
        raise ValueError("LLM_ANTHROPIC_API_KEY is not set")
    return AsyncAnthropic(api_key=env_key)


def create_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    env_key_name: str = "OPENAI_API_KEY",
) -> AsyncOpenAI:
    """
    Create an AsyncOpenAI client.

    Args:
        api_key: Optional API key. If not provided, uses env_key_name env var.
        base_url: Optional base URL for OpenAI-compatible APIs (e.g., OpenRouter).
        env_key_name: Name of the environment variable for the API key.

    Returns:
        AsyncOpenAI client instance

    Raises:
        ValueError: If no API key is available
    """
    key = api_key or os.getenv(env_key_name)
    if not key:
        raise ValueError(f"{env_key_name} is not set")

    if base_url:
        return AsyncOpenAI(api_key=key, base_url=base_url)
    return AsyncOpenAI(api_key=key)


def create_metrics_collector(prefix: str) -> MetricsCollector:
    """
    Create and start a MetricsCollector.

    Args:
        prefix: Prefix for the collection name (e.g., "longmem", "beam", "locomo")

    Returns:
        Started MetricsCollector instance
    """
    collector = MetricsCollector()
    collector.start_collection(f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return collector


def export_metrics(
    collector: MetricsCollector,
    prefix: str,
    output_dir: str = "tests/bench/perf_metrics",
) -> Path:
    """
    Export metrics to a JSON file and cleanup the collector.

    Args:
        collector: MetricsCollector instance
        prefix: Prefix for the output filename
        output_dir: Directory for output files

    Returns:
        Path to the exported metrics file
    """
    metrics_output = Path(
        f"{output_dir}/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    collector.export_to_json(metrics_output)
    collector.cleanup_collection()
    return metrics_output


class RunnerMixin:
    """
    Mixin class providing common functionality for benchmark runners.

    Requires the following attributes on the class:
    - redis_url: str
    - timeout_seconds: int
    - base_api_port: int
    - pool_size: int
    - reasoning_level: ReasoningLevel | None (optional)
    """

    # These are expected to be set by the inheriting class's __init__
    redis_url: str = ""
    timeout_seconds: int = 0
    base_api_port: int = 0
    pool_size: int = 0
    reasoning_level: ReasoningLevel | None = None
    # These are initialized by _init_common() - use Any to satisfy type checker
    # since the actual type is set at runtime
    metrics_collector: Any = None
    logger: Any = None

    def _init_common(self, metrics_prefix: str) -> None:
        """
        Initialize common runner components.

        Call this at the end of your __init__ after setting instance attributes.

        Args:
            metrics_prefix: Prefix for metrics collection (e.g., "longmem", "beam")
        """
        self.metrics_collector = create_metrics_collector(metrics_prefix)
        self.logger = configure_logging()

    def get_honcho_url_for_index(self, index: int) -> str:
        """Get the Honcho URL for a given index using round-robin distribution."""
        instance_id = index % self.pool_size
        port = self.base_api_port + instance_id
        return f"http://localhost:{port}"

    def create_honcho_client(self, workspace_id: str, honcho_url: str) -> Honcho:
        """Create a Honcho client for a specific workspace."""
        return Honcho(
            environment="local",
            workspace_id=workspace_id,
            base_url=honcho_url,
        )

    async def flush_deriver_queue(self) -> None:
        """Enable deriver flush mode to bypass batch token threshold."""
        redis_client: Redis = aioredis.from_url(self.redis_url)  # pyright: ignore[reportUnknownMemberType]
        try:
            await redis_client.set("honcho:deriver:flush_mode", "1", ex=60)
            print("Enabled deriver flush mode")
        finally:
            await redis_client.aclose()

    async def wait_for_deriver_queue_empty(
        self, honcho_client: Honcho, session_id: str | None = None
    ) -> bool:
        """Wait for the deriver queue to be empty."""
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.aio.queue_status(session=session_id)
            except Exception:
                await asyncio.sleep(1)
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.timeout_seconds:
                    return False
                continue

            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return True

            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout_seconds:
                return False
            await asyncio.sleep(1)

    async def trigger_dream_and_wait(
        self,
        honcho_client: Honcho,
        workspace_id: str,
        observer: str,
        observed: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """
        Trigger a dream task and wait for it to complete.

        Args:
            honcho_client: Honcho client instance
            workspace_id: Workspace identifier
            observer: Observer peer name
            observed: Observed peer name (defaults to observer)
            session_id: Session ID to scope the dream to

        Returns:
            True if dream completed successfully, False on timeout
        """
        observed = observed or observer
        honcho_url = self.get_honcho_url_for_index(0)

        url = f"{honcho_url}/v3/workspaces/{workspace_id}/schedule_dream"
        payload: dict[str, Any] = {
            "observer": observer,
            "observed": observed,
            "dream_type": "omni",
            "session_id": session_id or f"{workspace_id}_session",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=30.0,
                )
                if response.status_code != 204:
                    print(
                        f"[{workspace_id}] ERROR: Dream trigger failed with status {response.status_code}"
                    )
                    print(f"[{workspace_id}] Response body: {response.text}")
                    return False
        except Exception as e:
            print(f"[{workspace_id}] ERROR: Dream trigger exception: {e}")
            return False

        print(
            f"[{workspace_id}] Dream triggered successfully for {observer}/{observed}"
        )

        # Wait for dream queue to empty
        print(f"[{workspace_id}] Waiting for dream to complete...")
        await asyncio.sleep(2)  # Give time for dream to be enqueued
        await self.flush_deriver_queue()
        success = await self.wait_for_deriver_queue_empty(honcho_client)
        if success:
            print(f"[{workspace_id}] Dream queue empty")
        else:
            print(f"[{workspace_id}] Dream queue timeout")
        return success
