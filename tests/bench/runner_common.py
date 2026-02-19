"""
Shared utilities for Honcho benchmark test runners.

Contains the BaseRunner abstract class and RunnerConfig dataclass that provide
a common framework for all benchmark runners (longmem, beam, locomo).
"""

import argparse
import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from anthropic import AsyncAnthropic
from honcho import Honcho
from honcho.api_types import SessionConfiguration, SummaryConfiguration
from openai import AsyncOpenAI

from src.telemetry.metrics_collector import MetricsCollector

# Valid reasoning levels for dialectic chat
ReasoningLevel = Literal["minimal", "low", "medium", "high", "max"]
REASONING_LEVELS: list[str] = ["minimal", "low", "medium", "high", "max"]

# Type variable for result types
ResultT = TypeVar("ResultT")


@dataclass
class RunnerConfig:
    """Configuration shared across all benchmark runners."""

    base_api_port: int = 8000
    pool_size: int = 1
    timeout_seconds: int = 600
    batch_size: int = 10
    cleanup_workspace: bool = False
    use_get_context: bool = False
    reasoning_level: ReasoningLevel | None = None
    base_url: str | None = None
    api_key: str | None = None
    skip_dream: bool = False
    json_output: Path | None = None
    max_concurrent: int | None = None  # None means no limit (use batch_size)

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, default_timeout: int = 600
    ) -> "RunnerConfig":
        """Create config from parsed CLI arguments."""
        return cls(
            base_api_port=args.base_api_port,
            pool_size=args.pool_size,
            timeout_seconds=args.timeout
            if args.timeout is not None
            else default_timeout,
            batch_size=args.batch_size,
            cleanup_workspace=args.cleanup_workspace,
            use_get_context=args.use_get_context,
            reasoning_level=args.reasoning_level,
            base_url=args.base_url,
            api_key=args.api_key,
            skip_dream=args.skip_dream,
            json_output=args.json_output,
            max_concurrent=args.max_concurrent,
        )


@dataclass
class ItemContext:
    """Context for executing a single benchmark item."""

    workspace_id: str
    honcho_client: Honcho
    honcho_url: str
    session_id: str
    peers: dict[str, Any] = field(default_factory=dict)
    session: Any = None


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common command line arguments shared across all benchmark runners.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for remote Honcho instance (e.g., https://groudon.fly.dev). Overrides --base-api-port.",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for remote Honcho instance authentication",
    )

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
        "--reasoning-level",
        type=str,
        choices=REASONING_LEVELS,
        default=None,
        help="Reasoning level for dialectic chat: minimal, low, medium, high, max (default: None)",
    )

    parser.add_argument(
        "--skip-dream",
        action="store_true",
        help="Skip the dream consolidation step (default: False)",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent items executing at once (default: unlimited, use for rate-limited remote instances)",
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

    if args.max_concurrent is not None and args.max_concurrent <= 0:
        return f"Error: Max concurrent must be positive, got {args.max_concurrent}"

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


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class BaseRunner(ABC, Generic[ResultT]):
    """
    Abstract base class for benchmark runners.

    Provides a template method pattern for executing benchmarks with common
    infrastructure for Honcho client management, queue waiting, and dream triggering.

    Subclasses must implement:
    - get_metrics_prefix(): Return the metrics prefix (e.g., "longmem")
    - load_items(): Load and return the items to process
    - get_workspace_id(item): Return workspace ID for an item
    - get_session_id(item): Return session ID for an item
    - setup_peers(ctx, item): Create and configure peers
    - setup_session(ctx, item): Create and configure session with peers
    - ingest_messages(ctx, item): Ingest messages into the session
    - get_dream_observers(item): Return list of peer IDs to trigger dreams for
    - execute_questions(ctx, item): Execute questions and return result
    - print_summary(results, duration): Print summary of results
    - generate_output(results, duration): Generate JSON output
    """

    def __init__(self, config: RunnerConfig):
        """
        Initialize the runner with configuration.

        Args:
            config: Runner configuration
        """
        self.config: RunnerConfig = config
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.metrics_collector.start_collection(
            f"{self.get_metrics_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.logger: Logger = configure_logging()
        # Semaphore for rate limiting concurrent item execution
        self._concurrency_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
        )

    # -------------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_metrics_prefix(self) -> str:
        """Return the metrics prefix for this runner (e.g., 'longmem', 'beam')."""
        ...

    @abstractmethod
    def load_items(self) -> list[Any]:
        """Load and return the list of items to process."""
        ...

    @abstractmethod
    def get_workspace_id(self, item: Any) -> str:
        """Return the workspace ID for a given item."""
        ...

    @abstractmethod
    def get_session_id(self, item: Any, workspace_id: str) -> str:
        """Return the session ID for a given item."""
        ...

    @abstractmethod
    async def setup_peers(self, ctx: ItemContext, item: Any) -> None:
        """
        Create and configure peers for the item.

        Should populate ctx.peers with peer objects.
        """
        ...

    @abstractmethod
    async def setup_session(self, ctx: ItemContext, item: Any) -> None:
        """
        Create and configure the session with peers.

        Should set ctx.session and add peers to the session.
        """
        ...

    @abstractmethod
    async def ingest_messages(self, ctx: ItemContext, item: Any) -> int:
        """
        Ingest messages into the session.

        Returns:
            Number of messages ingested
        """
        ...

    @abstractmethod
    def get_dream_observers(self, item: Any) -> list[str]:
        """Return list of peer IDs to trigger dreams for."""
        ...

    def get_dream_session_ids(self, ctx: ItemContext, _item: Any) -> list[str]:
        """Return session IDs to use for dream scheduling.

        Subclasses can override this when ingestion stores messages across
        multiple sessions and each session should be included in dream
        scheduling.
        """
        return [ctx.session_id]

    @abstractmethod
    async def execute_questions(self, ctx: ItemContext, item: Any) -> ResultT:
        """
        Execute questions/queries for the item.

        Returns:
            Result object for this item
        """
        ...

    @abstractmethod
    def print_summary(self, results: list[ResultT], total_duration: float) -> None:
        """Print a summary of all results."""
        ...

    @abstractmethod
    def generate_output(self, results: list[ResultT], total_duration: float) -> None:
        """Generate JSON output file."""
        ...

    # -------------------------------------------------------------------------
    # Template method - the main execution flow
    # -------------------------------------------------------------------------

    async def run(self) -> tuple[list[ResultT], float]:
        """
        Run the benchmark.

        This is the main template method that orchestrates the execution flow.

        Returns:
            Tuple of (list of results, total duration in seconds)
        """
        items = self.load_items()
        if not items:
            return [], 0.0

        print(f"Found {len(items)} items to process")
        if self.config.pool_size > 1:
            print(
                f"Distributing across {self.config.pool_size} Honcho instances "
                + f"(ports {self.config.base_api_port}-{self.config.base_api_port + self.config.pool_size - 1})"
            )
        if self.config.max_concurrent:
            print(f"Limiting to {self.config.max_concurrent} concurrent item(s)")

        overall_start = time.time()
        all_results: list[ResultT] = []

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(items) + batch_size - 1) // batch_size

            print(f"\n{'=' * 60}")
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            print(f"{'=' * 60}")

            # Run items in batch concurrently (with optional rate limiting)
            batch_results = await asyncio.gather(
                *[
                    self._execute_item_with_limit(item, self._get_honcho_url(i + idx))
                    for idx, item in enumerate(batch)
                ]
            )

            all_results.extend(batch_results)

        overall_duration = time.time() - overall_start

        # Finalize metrics
        self.metrics_collector.finalize_collection()

        return all_results, overall_duration

    async def _execute_item_with_limit(self, item: Any, honcho_url: str) -> ResultT:
        """Wrapper that applies concurrency limiting if configured."""
        if self._concurrency_semaphore:
            async with self._concurrency_semaphore:
                return await self.execute_item(item, honcho_url)
        return await self.execute_item(item, honcho_url)

    async def execute_item(self, item: Any, honcho_url: str) -> ResultT:
        """
        Execute a single benchmark item.

        This method orchestrates the standard flow:
        1. Create workspace and client
        2. Setup peers and session
        3. Ingest messages
        4. Wait for queue to empty
        5. Trigger dreams
        6. Execute questions
        7. Cleanup (if configured)

        Args:
            item: The item to process
            honcho_url: URL of the Honcho instance to use

        Returns:
            Result for this item
        """
        workspace_id = self.get_workspace_id(item)
        session_id = self.get_session_id(item, workspace_id)

        print(f"\n{'=' * 80}")
        print(f"Executing {workspace_id}")
        print(f"Using Honcho instance: {honcho_url}")
        print(f"{'=' * 80}")

        # Create context
        ctx = ItemContext(
            workspace_id=workspace_id,
            honcho_client=self._create_honcho_client(workspace_id, honcho_url),
            honcho_url=honcho_url,
            session_id=session_id,
        )

        start_time = time.time()

        try:
            # Setup peers
            await self.setup_peers(ctx, item)

            # Setup session
            await self.setup_session(ctx, item)

            # Ingest messages
            print(f"[{workspace_id}] Ingesting messages...")
            message_count = await self.ingest_messages(ctx, item)
            print(f"[{workspace_id}] Ingested {message_count} messages")

            # Wait for deriver queue
            print(f"[{workspace_id}] Waiting for deriver queue to empty...")
            await asyncio.sleep(1)  # Give time for tasks to be queued

            queue_empty = await self._wait_for_queue_empty(ctx.honcho_client)
            if not queue_empty:
                raise TimeoutError(
                    f"Deriver queue timeout after {self.config.timeout_seconds}s"
                )

            # Trigger dreams
            dream_observers = self.get_dream_observers(item)
            dream_session_ids = self.get_dream_session_ids(ctx, item)
            if not dream_session_ids:
                raise ValueError(
                    f"No dream session IDs available for {workspace_id}. "
                    + "Dream scheduling requires at least one session id."
                )

            print(
                f"[{workspace_id}] Deriver queue empty. Triggering dreams for "
                + f"{len(dream_observers)} observer(s) across "
                + f"{len(dream_session_ids)} session(s)..."
            )
            for observer in dream_observers:
                for dream_session_id in dream_session_ids:
                    success = await self._trigger_dream(
                        ctx.honcho_client, workspace_id, observer, dream_session_id
                    )
                    if not success:
                        print(
                            f"[{workspace_id}] Warning: Dream for {observer} in "
                            + f"session {dream_session_id} did not complete"
                        )

            # Execute questions
            print(f"[{workspace_id}] Executing questions...")
            result = await self.execute_questions(ctx, item)

            # Cleanup
            if self.config.cleanup_workspace:
                try:
                    await ctx.honcho_client.aio.delete_workspace(workspace_id)
                    print(f"[{workspace_id}] Cleaned up workspace")
                except Exception as e:
                    print(f"[{workspace_id}] Failed to delete workspace: {e}")

            duration = time.time() - start_time
            print(f"[{workspace_id}] Completed in {format_duration(duration)}")

            return result

        except Exception as e:
            self.logger.error(f"Error executing {workspace_id}: {e}")
            # Let subclass handle error result creation
            raise

    def run_and_summarize(self) -> int:
        """
        Run the benchmark, print summary, and generate output.

        This is a convenience method that runs the full benchmark flow.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            results, total_duration = asyncio.run(self.run())

            self.print_summary(results, total_duration)
            self.metrics_collector.print_summary()
            self.generate_output(results, total_duration)

            # Export metrics
            metrics_output = Path(
                f"tests/bench/perf_metrics/{self.get_metrics_prefix()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.metrics_collector.export_to_json(metrics_output)
            self.metrics_collector.cleanup_collection()

            return 0

        except KeyboardInterrupt:
            print("\nTest execution interrupted by user")
            return 1
        except Exception as e:
            print(f"Error running tests: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # -------------------------------------------------------------------------
    # Infrastructure methods
    # -------------------------------------------------------------------------

    def _get_honcho_url(self, index: int) -> str:
        """Get the Honcho URL for a given index using round-robin distribution."""
        if self.config.base_url:
            return self.config.base_url
        instance_id = index % self.config.pool_size
        port = self.config.base_api_port + instance_id
        return f"http://localhost:{port}"

    def _create_honcho_client(self, workspace_id: str, honcho_url: str) -> Honcho:
        """Create a Honcho client for a specific workspace."""
        return Honcho(
            workspace_id=workspace_id,
            base_url=honcho_url,
            api_key=self.config.api_key,
        )

    def _get_session_configuration(self) -> SessionConfiguration:
        """Get default session configuration with summaries disabled."""
        return SessionConfiguration(summary=SummaryConfiguration(enabled=False))

    async def _wait_for_queue_empty(
        self, honcho_client: Honcho, session_id: str | None = None
    ) -> bool:
        """Wait for the deriver queue to be empty."""
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.aio.queue_status(session=session_id)
            except Exception:
                await asyncio.sleep(1)
                if time.time() - start_time >= self.config.timeout_seconds:
                    return False
                continue

            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return True

            if time.time() - start_time >= self.config.timeout_seconds:
                return False
            await asyncio.sleep(1)

    async def _trigger_dream(
        self,
        honcho_client: Honcho,
        workspace_id: str,
        observer: str,
        session_id: str,
        observed: str | None = None,
    ) -> bool:
        """
        Trigger a dream task and wait for it to complete.

        Returns:
            True if dream completed (or was skipped), False on timeout
        """
        if self.config.skip_dream:
            print(f"[{workspace_id}] Skipping dream for {observer} (--skip-dream)")
            return True

        observed = observed or observer

        try:
            await honcho_client.aio.schedule_dream(
                observer=observer,
                session=session_id,
                observed=observed,
            )
        except Exception as e:
            print(f"[{workspace_id}] ERROR: Dream trigger exception: {e}")
            return False

        print(f"[{workspace_id}] Dream triggered for {observer}/{observed}")

        # Wait for dream to complete
        await asyncio.sleep(2)
        success = await self._wait_for_queue_empty(honcho_client)
        if success:
            print(f"[{workspace_id}] Dream for {observer} completed")
        else:
            print(f"[{workspace_id}] Dream for {observer} timed out")
        return success
