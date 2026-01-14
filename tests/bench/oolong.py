"""
Honcho OOLONG Benchmark Test Runner

A script that executes OOLONG benchmark tests against a running Honcho instance.

OOLONG tests long-context reasoning and aggregation by requiring models to:
1. Identify relevant segments of input
2. Classify or categorize those segments
3. Aggregate results to answer distributional questions

Two variants:
- OOLONG-synth: Questions over ICL datasets (counting, user, temporal tasks)
- OOLONG-real: Questions over D&D transcripts (dice rolls, spells, character actions)

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```

1. Run the test harness:
```
python -m tests.bench.harness
```

2. Run this file with dataset selection:
```
# For OOLONG-synth (default)
python -m tests.bench.oolong --variant synth --data-dir /path/to/oolong-synth

# For OOLONG-real (D&D transcripts)
python -m tests.bench.oolong --variant real --data-dir /path/to/oolong-real
```

Required arguments:
```
--data-dir: Path to the dataset directory (e.g., /path/to/oolong-synth or /path/to/oolong-real)
```

Optional arguments:
```
--variant: Which OOLONG variant to run ("synth" or "real", default: "synth")
--split: Dataset split to use ("test" or "validation", default: "test")
--max-examples: Maximum number of examples to run (default: all)
--context-size: Discrete context size (e.g., "8K", "16K", "32K", or exact like "16384")
              Overrides --min-context-len and --max-context-len
              Available: 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 2M, 4M
--max-context-len: Maximum context length in tokens (default: no limit)
--min-context-len: Minimum context length in tokens (default: 0)
--context-window-id: Run only examples with this context_window_id
--timeout: Timeout for deriver queue in seconds (default: 600)
--base-api-port: Base port for Honcho API instances (default: 8000)
--pool-size: Number of Honcho instances in the pool (default: 1)
--batch-size: Number of questions to run concurrently (default: 5)
--json-output: Path to write JSON summary results (auto-generated if not provided)
--no-merge-sessions: Disable merging of contexts into a single session (default: enabled)
--cleanup-workspace: Delete workspace after each question (default: False)
--use-dialectic-agentic: Use agentic dialectic mode for answering (default: False)

Examples:
```bash
# Test at exact 16K context length
python -m tests.bench.oolong --variant synth --data-dir /path/to/oolong-synth --context-size 16K

# Test at exact 128K context length
python -m tests.bench.oolong --variant synth --data-dir /path/to/oolong-synth --context-size 128K

# Test with exact token count
python -m tests.bench.oolong --variant synth --data-dir /path/to/oolong-synth --context-size 16384
```
"""

import argparse
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)
from typing_extensions import TypedDict

from src.utils.metrics_collector import MetricsCollector

from .oolong_common import (
    BaseTestResult,
    calculate_context_length,
    calculate_task_statistics,
    calculate_timing_statistics,
    filter_dataset,
    format_duration,
    load_oolong_real_dataset,
    load_oolong_synth_dataset,
    parse_real_answer,
    parse_real_context_messages,
    parse_synth_answer,
    parse_synth_context_messages,
    score_real_response,
    score_synth_response,
    write_json_summary,
)

load_dotenv()


def parse_context_size(size_str: str) -> int:
    """Parse a context size string to exact token count.

    Args:
        size_str: Context size string (e.g., "8K", "16K", "1M") or numeric string ("16384")

    Returns:
        Exact token count (power of 2)

    Raises:
        ValueError: If the size string is invalid
    """
    size_str = size_str.strip().upper()

    # Check if it's a plain number
    if size_str.isdigit():
        return int(size_str)

    # Parse suffix-based sizes (e.g., "8K", "16K", "1M")
    multipliers = {
        "K": 1024,
        "M": 1024 * 1024,
    }

    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            try:
                base = int(size_str[:-1])
                return base * multiplier
            except ValueError:
                raise ValueError(f"Invalid context size format: {size_str}")

    raise ValueError(f"Invalid context size format: {size_str}. Use formats like '8K', '16K', '1M', or exact numbers like '16384'.")


class QueryResult(TypedDict):
    """Type definition for query execution results."""

    question: str
    expected_answer: Any
    actual_response: str
    score: float
    context_length_tokens: int


class TestResult(BaseTestResult):
    """Type definition for OOLONG test execution results."""

    context_window_id: str
    answer_type: str
    query_executed: QueryResult | None


class OolongBenchmarkRunner:
    """
    Executes OOLONG benchmark tests against a Honcho instance.
    """

    def __init__(
        self,
        variant: str = "synth",
        data_dir: str | Path | None = None,
        base_api_port: int = 8000,
        pool_size: int = 1,
        timeout_seconds: int = 600,
        merge_sessions: bool = True,
        cleanup_workspace: bool = False,
        use_dialectic_agentic: bool = False,
    ):
        """
        Initialize the benchmark runner.

        Args:
            variant: Which OOLONG variant to run ("synth" or "real")
            data_dir: Path to the dataset directory (required)
            base_api_port: Base port for Honcho API instances
            pool_size: Number of Honcho instances in the pool
            timeout_seconds: Timeout for deriver queue in seconds
            merge_sessions: If True, merge all context into one session
            cleanup_workspace: If True, delete workspace after executing question
            use_dialectic_agentic: If True, use agentic=true mode for dialectic
        """
        self.variant: str = variant
        self.data_dir: str | Path | None = data_dir
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.timeout_seconds: int = timeout_seconds
        self.merge_sessions: bool = merge_sessions
        self.cleanup_workspace: bool = cleanup_workspace
        self.use_dialectic_agentic: bool = use_dialectic_agentic

        # Initialize metrics collector
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.metrics_collector.start_collection(
            f"oolong_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

    def get_honcho_url_for_index(self, question_index: int) -> str:
        """
        Get the Honcho URL for a given question index using round-robin distribution.

        Args:
            question_index: Index of the question

        Returns:
            URL of the Honcho instance to use
        """
        instance_id = question_index % self.pool_size
        port = self.base_api_port + instance_id
        return f"http://localhost:{port}"

    async def create_honcho_client(
        self, workspace_id: str, honcho_url: str
    ) -> AsyncHoncho:
        """
        Create a Honcho client for a specific workspace.

        Args:
            workspace_id: Workspace ID
            honcho_url: URL of the Honcho instance

        Returns:
            AsyncHoncho client instance
        """
        return AsyncHoncho(
            environment="local",
            workspace_id=workspace_id,
            base_url=honcho_url,
        )

    async def wait_for_deriver_queue_empty(
        self, honcho_client: AsyncHoncho, session_id: str | None = None
    ) -> bool:
        """Wait for deriver queue to be empty."""
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.get_queue_status(session=session_id)
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

    async def execute_question(
        self, example_data: dict[str, Any], _question_index: int, honcho_url: str
    ) -> TestResult:
        """
        Execute a single OOLONG benchmark question.

        Args:
            example_data: Dictionary containing example from dataset
            _question_index: Index of this question (unused, URL already computed)
            honcho_url: URL of the Honcho instance to use

        Returns:
            Test execution results
        """
        # Extract fields based on variant
        example_id = example_data["id"]
        context_window_id = example_data["context_window_id"]
        question = example_data["question"]
        answer_str = example_data["answer"]

        # Variant-specific fields
        if self.variant == "synth":
            # Use context with labels for synth variant
            context_text = example_data.get("context_window_text_with_labels", example_data["context_window_text"])
            task_group = example_data.get("task_group", "unknown")
            dataset_name = example_data.get("dataset", "unknown")
            answer_type = example_data.get("answer_type", "unknown")
            gold_answer = parse_synth_answer(answer_str)
        else:  # real
            context_text = example_data["context_window_text"]
            task_group = example_data.get("question_type", "unknown")
            dataset_name = "oolong-real"
            answer_type = "varied"
            gold_answer = parse_real_answer(answer_str)

        output_lines: list[str] = []
        output_lines.append(
            f"\033[1mExecuting {self.variant} question {example_id}\033[0m"
        )
        output_lines.append(f"Task: {task_group} | Dataset: {dataset_name}")
        output_lines.append(f"Question: {question}")
        output_lines.append(f"Expected: {gold_answer}")

        # Create workspace for this question
        workspace_id = f"oolong_{self.variant}_{example_id}"
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        # Calculate context length
        context_length = calculate_context_length(context_text)

        results: TestResult = {
            "question_id": example_id,
            "context_window_id": context_window_id,
            "task_group": task_group,
            "dataset": dataset_name,
            "answer_type": answer_type,
            "query_executed": None,
            "passed": False,
            "score": 0.0,
            "error": None,
            "start_time": time.time(),
            "end_time": 0.0,
            "duration_seconds": 0.0,
            "output_lines": output_lines,
        }

        try:
            # Create peers
            user_peer = await honcho_client.peer(id="user")

            # Parse context into messages
            if self.variant == "synth":
                messages = parse_synth_context_messages(context_text)
            else:  # real
                messages = parse_real_context_messages(context_text)

            output_lines.append(
                f"Ingesting {len(messages)} messages ({context_length} tokens)..."
            )

            # Create session and add messages
            session_id = f"{workspace_id}_session"
            session = await honcho_client.session(id=session_id)

            # Configure peer to observe itself
            await session.add_peers(
                [
                    (
                        user_peer,
                        SessionPeerConfig(observe_me=True, observe_others=False),
                    ),
                ]
            )

            # Convert messages to Honcho format and batch them
            honcho_messages: list[MessageCreateParam] = []
            for msg in messages:
                content = msg["content"]
                metadata = msg.get("metadata", {})

                # Create message with metadata
                honcho_messages.append(
                    user_peer.message(
                        content=content,
                        metadata=metadata,
                    )
                )

            # Add messages in batches of 100
            for i in range(0, len(honcho_messages), 100):
                batch = honcho_messages[i : i + 100]
                await session.add_messages(batch)

            output_lines.append("Messages ingested. Waiting for deriver...")
            await asyncio.sleep(1)  # Give time for tasks to be queued

            # Wait for deriver to process
            queue_empty = await self.wait_for_deriver_queue_empty(
                honcho_client, session_id
            )
            if not queue_empty:
                output_lines.append("ERROR: Deriver queue timeout!")
                results["error"] = "Deriver queue timeout"
                return results

            output_lines.append("Deriver complete. Executing question...")

            # Query via dialectic (scoped to session for better message access)
            try:
                actual_response = await user_peer.chat(question, session=session)

                # Clean up workspace if requested
                if self.cleanup_workspace:
                    try:
                        await honcho_client.delete_workspace(workspace_id)
                        output_lines.append("Workspace cleaned up")
                    except Exception as e:
                        output_lines.append(f"Warning: Failed to delete workspace: {e}")

                actual_response = (
                    actual_response if isinstance(actual_response, str) else ""
                )

                # Score the response
                if self.variant == "synth":
                    score = score_synth_response(
                        gold_answer, actual_response, answer_type
                    )
                else:  # real
                    score = score_real_response(gold_answer, actual_response)

                query_result: QueryResult = {
                    "question": question,
                    "expected_answer": str(gold_answer),
                    "actual_response": actual_response,
                    "score": score,
                    "context_length_tokens": context_length,
                }

                results["query_executed"] = query_result
                results["score"] = score
                results["passed"] = score >= 0.99  # Consider 99%+ as pass

                output_lines.append(
                    f"  score: \033[1m{score:.3f}\033[0m"
                    + (
                        " \033[32m✓\033[0m"
                        if score >= 0.99
                        else (" \033[33m~\033[0m" if score > 0 else " \033[31m✗\033[0m")
                    )
                )
                if score < 0.99:
                    output_lines.append(f"  response: \033[3m{actual_response}\033[0m")
                    output_lines.append(f"  expected: {gold_answer}")

            except Exception as e:
                self.logger.error(f"Error executing question: {e}")
                query_result = QueryResult(
                    question=question,
                    expected_answer=str(gold_answer),
                    actual_response=f"ERROR: {e}",
                    score=0.0,
                    context_length_tokens=context_length,
                )
                results["query_executed"] = query_result
                results["score"] = 0.0
                results["passed"] = False
                output_lines.append(f"ERROR: {e}")

            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]

        except Exception as e:
            self.logger.error(f"Error in execute_question: {e}")
            results["error"] = str(e)
            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]

        return results

    async def run_benchmark(
        self,
        split: str = "test",
        max_examples: int | None = None,
        max_context_len: int | None = None,
        min_context_len: int | None = None,
        context_window_id: str | None = None,
        batch_size: int = 5,
        json_output: Path | None = None,
    ) -> dict[str, Any]:
        """
        Run the OOLONG benchmark.

        Args:
            split: Dataset split to use
            max_examples: Maximum number of examples to run
            max_context_len: Maximum context length filter
            min_context_len: Minimum context length filter
            context_window_id: Filter to specific context window
            batch_size: Number of questions to run concurrently
            json_output: Path to write JSON results

        Returns:
            Summary dictionary
        """
        print(f"\n{'=' * 80}")
        print(f"OOLONG-{self.variant} Benchmark")
        print(f"{'=' * 80}\n")

        # Load dataset
        print(f"Loading OOLONG-{self.variant} dataset (split: {split})...")
        if self.variant == "synth":
            dataset = load_oolong_synth_dataset(split, data_dir=self.data_dir)
        else:
            dataset = load_oolong_real_dataset(split, data_dir=self.data_dir)

        # Filter dataset
        dataset = filter_dataset(
            dataset,
            max_context_len=max_context_len,
            min_context_len=min_context_len,
            max_examples=max_examples,
            context_window_id=context_window_id,
        )

        print(f"Running {len(dataset)} examples...")
        print(f"Using {self.pool_size} Honcho instance(s)")
        print(f"Batch size: {batch_size}\n")

        start_time = time.time()
        results: list[TestResult] = []

        # Process in batches
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            # Convert HuggingFace dataset slice to list of dicts
            batch = [dataset[i] for i in range(batch_start, batch_end)]

            print(
                f"\n--- Batch {batch_start // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size} ---"
            )

            # Run batch concurrently
            tasks = [
                self.execute_question(
                    example,
                    batch_start + i,
                    self.get_honcho_url_for_index(batch_start + i),
                )
                for i, example in enumerate(batch)
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Print batch results
            for result in batch_results:
                for line in result["output_lines"]:
                    print(line)
                print()

        end_time = time.time()
        total_elapsed = end_time - start_time

        # Calculate statistics
        total_score = sum(r["score"] for r in results)
        avg_score = total_score / len(results) if results else 0.0
        perfect_scores = sum(1 for r in results if r["score"] >= 0.99)

        task_stats = calculate_task_statistics(results)
        timing_stats = calculate_timing_statistics(results, total_elapsed)

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"OOLONG-{self.variant} Results Summary")
        print(f"{'=' * 80}\n")
        print(f"Total examples: {len(results)}")
        print(f"Average score: {avg_score:.3f}")
        perfect_rate = (perfect_scores/len(results)*100) if len(results) > 0 else 0.0
        print(f"Perfect scores (≥0.99): {perfect_scores} ({perfect_rate:.1f}%)")
        print(f"Total time: {format_duration(total_elapsed)}")

        if len(results) > 0:
            print(f"\nTask Group Statistics:")
            for task_name, stats in task_stats.items():
                print(
                    f"  {task_name}: avg={stats['average_score']:.3f}, " +
                    f"perfect={stats['perfect_score_rate']:.1f}%"
                )
        else:
            print("\nNo examples were processed. Check dataset loading or filters.")

        # Write JSON summary
        summary = {
            "variant": self.variant,
            "split": split,
            "total_examples": len(results),
            "average_score": avg_score,
            "perfect_scores": perfect_scores,
            "perfect_score_rate": perfect_scores / len(results) if results else 0.0,
            "task_statistics": task_stats,
            "timing_statistics": timing_stats,
            "results": [
                {
                    "question_id": r["question_id"],
                    "context_window_id": r["context_window_id"],
                    "task_group": r["task_group"],
                    "dataset": r["dataset"],
                    "score": r["score"],
                    "passed": r["passed"],
                    "error": r["error"],
                    "duration_seconds": r["duration_seconds"],
                    "query": r["query_executed"],
                }
                for r in results
            ],
        }

        if json_output is None:
            # Auto-generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_output = (
                Path("tests/bench/eval_results")
                / f"oolong_{self.variant}_{timestamp}.json"
            )

        write_json_summary(summary, json_output)

        return summary


async def main():
    """Main entry point for the OOLONG benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run OOLONG benchmark tests against Honcho",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="synth",
        choices=["synth", "real"],
        help="Which OOLONG variant to run (default: synth)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the dataset directory (e.g., /path/to/oolong-synth or /path/to/oolong-real)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to run (default: all)",
    )
    parser.add_argument(
        "--context-size",
        type=str,
        default=None,
        help=(
            "Discrete context size (e.g., '8K', '16K', '32K', or exact token count like '16384'). "
            "Overrides --min-context-len and --max-context-len. "
            "Available sizes: 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 2M, 4M"
        ),
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Maximum context length in tokens (default: no limit). Ignored if --context-size is set.",
    )
    parser.add_argument(
        "--min-context-len",
        type=int,
        default=0,
        help="Minimum context length in tokens (default: 0). Ignored if --context-size is set.",
    )
    parser.add_argument(
        "--context-window-id",
        type=str,
        default=None,
        help="Run only examples with this context_window_id",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for deriver queue in seconds (default: 600)",
    )
    parser.add_argument(
        "--base-api-port",
        type=int,
        default=8000,
        help="Base port for Honcho API (default: 8000)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1,
        help="Number of Honcho instances in the pool (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of questions to run concurrently (default: 5)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Path to write JSON summary (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-merge-sessions",
        action="store_false",
        dest="merge_sessions",
        default=True,
        help="Disable merging of contexts into a single session (default: enabled)",
    )
    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        default=False,
        help="Delete workspace after each question (default: False)",
    )
    parser.add_argument(
        "--use-dialectic-agentic",
        action="store_true",
        default=False,
        help="Use agentic=true mode for dialectic (default: False)",
    )

    args = parser.parse_args()

    # Handle --context-size parameter (overrides min/max context len)
    if args.context_size:
        try:
            exact_size = parse_context_size(args.context_size)
            args.min_context_len = exact_size
            args.max_context_len = exact_size
            print(f"Using discrete context size: {args.context_size} ({exact_size} tokens)")
        except ValueError as e:
            print(f"Error: {e}")
            return

    # Create runner
    runner = OolongBenchmarkRunner(
        variant=args.variant,
        data_dir=args.data_dir,
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        timeout_seconds=args.timeout,
        merge_sessions=args.merge_sessions,
        cleanup_workspace=args.cleanup_workspace,
        use_dialectic_agentic=args.use_dialectic_agentic,
    )

    # Run benchmark
    await runner.run_benchmark(
        split=args.split,
        max_examples=args.max_examples,
        max_context_len=args.max_context_len,
        min_context_len=args.min_context_len,
        context_window_id=args.context_window_id,
        batch_size=args.batch_size,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    asyncio.run(main())
