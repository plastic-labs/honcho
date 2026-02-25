"""
Honcho RefusalBench Benchmark Test Runner

A script that executes RefusalBench benchmark tests against a running Honcho instance.
Based on the RefusalBench methodology (arXiv:2510.10390), this evaluates whether a
memory system can correctly identify when it shouldn't answer a question.

This script:
1. Loads RefusalBench entries (perturbed queries + contexts) from a JSONL file
2. Creates a workspace for each entry (isolation)
3. Ingests the perturbed context as user messages
4. Waits for deriver processing and triggers dreams
5. Asks the perturbed query via dialectic chat
6. Uses an LLM judge to classify the response as answer or refusal

## Perturbation Classes

RefusalBench uses 6 perturbation classes across 3 intensity levels:
1. P-Ambiguity      - Query becomes ambiguous
2. P-Contradiction   - Context contains contradictions
3. P-MissingInfo     - Key information removed from context
4. P-FalsePremise    - Query contains a false assumption
5. P-GranularityMismatch - Query asks for precision context can't support
6. P-EpistemicMismatch   - Query asks for opinion/speculation as fact

## To use

0. Set up env and run the test harness:
```
uv sync
python -m tests.bench.harness
```

1. Run with the sample dataset:
```
python -m tests.bench.refusalbench
```

Optional arguments:
```
--data-file: Path to JSONL dataset (default: tests/bench/refusalbench_data/refusalbench_sample.jsonl)
--test-count: Number of items to run (default: all)
--unique-id: Run a specific item by unique_id
--perturbation-class: Filter by perturbation class
--intensity: Filter by intensity level (LOW, MEDIUM, HIGH)
--timeout: Timeout for deriver queue in seconds (default: 600)
--json-output: Path to write JSON results
```
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from honcho.api_types import MessageCreateParams
from honcho.session import SessionPeerConfig
from openai import AsyncOpenAI

from src.config import settings

from .refusalbench_common import (
    INTENSITY_LEVELS,
    PERTURBATION_CLASSES,
    ItemResult,
    RefusalBenchEntry,
    evaluate_single_item,
    generate_json_summary,
    judge_refusal_response,
    load_refusalbench_data,
    print_summary,
)
from .runner_common import (
    BaseRunner,
    ItemContext,
    RunnerConfig,
    add_common_arguments,
    create_openai_client,
    validate_common_arguments,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")


class RefusalBenchRunner(BaseRunner[ItemResult]):
    """Executes RefusalBench benchmark tests against a Honcho instance."""

    def __init__(
        self,
        config: RunnerConfig,
        data_file: Path,
        test_count: int | None = None,
        unique_id: str | None = None,
        perturbation_class: str | None = None,
        intensity: str | None = None,
    ):
        """
        Initialize the RefusalBench test runner.

        Args:
            config: Common runner configuration.
            data_file: Path to the JSONL dataset file.
            test_count: Optional limit on number of items to process.
            unique_id: Optional specific item unique_id to run.
            perturbation_class: Optional filter by perturbation class.
            intensity: Optional filter by intensity level.
        """
        self.data_file: Path = data_file
        self.test_count: int | None = test_count
        self.unique_id_filter: str | None = unique_id
        self.perturbation_class_filter: str | None = perturbation_class
        self.intensity_filter: str | None = intensity

        # Store loaded entries for use in execute_questions
        self._entries_by_id: dict[str, RefusalBenchEntry] = {}

        # Initialize base class
        super().__init__(config)

        # Initialize OpenRouter client for judging
        openrouter_base_url = os.getenv(
            "LLM_OPENAI_COMPATIBLE_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.openrouter_client: AsyncOpenAI = create_openai_client(
            base_url=openrouter_base_url,
            env_key_name="LLM_OPENAI_COMPATIBLE_API_KEY",
        )

        # Model to use for judging (OpenRouter format)
        self.judge_model: str = os.getenv(
            "REFUSALBENCH_JUDGE_MODEL", "anthropic/claude-sonnet-4.5"
        )

    def get_metrics_prefix(self) -> str:
        return "refusalbench"

    def load_items(self) -> list[Any]:
        """Load and filter RefusalBench entries."""
        entries = load_refusalbench_data(self.data_file)

        # Apply filters
        if self.unique_id_filter:
            entries = [e for e in entries if e["unique_id"] == self.unique_id_filter]

        if self.perturbation_class_filter:
            entries = [
                e
                for e in entries
                if e["perturbation_class"] == self.perturbation_class_filter
            ]

        if self.intensity_filter:
            entries = [
                e for e in entries if e["intensity"] == self.intensity_filter
            ]

        if self.test_count is not None:
            entries = entries[: self.test_count]

        # Build lookup for execute_questions
        self._entries_by_id = {e["unique_id"]: e for e in entries}

        return cast(list[Any], entries)

    def get_workspace_id(self, item: Any) -> str:
        """Return workspace ID for an entry."""
        entry = cast(RefusalBenchEntry, item)
        return f"rb_{entry['unique_id']}"

    def get_session_id(self, item: Any, workspace_id: str) -> str:
        """Return session ID for an entry."""
        return f"{workspace_id}_session"

    async def setup_peers(self, ctx: ItemContext, item: Any) -> None:
        """Create user and assistant peers."""
        ctx.peers["user"] = await ctx.honcho_client.aio.peer(id="user")
        ctx.peers["assistant"] = await ctx.honcho_client.aio.peer(id="assistant")

    async def setup_session(self, ctx: ItemContext, item: Any) -> None:
        """Create session with user peer observed."""
        user_peer = ctx.peers["user"]
        assistant_peer = ctx.peers["assistant"]

        ctx.session = await ctx.honcho_client.aio.session(
            id=ctx.session_id, configuration=self._get_session_configuration()
        )

        await ctx.session.aio.add_peers(
            [
                (user_peer, SessionPeerConfig(observe_me=True, observe_others=False)),
                (
                    assistant_peer,
                    SessionPeerConfig(observe_me=False, observe_others=False),
                ),
            ]
        )

    async def ingest_messages(self, ctx: ItemContext, item: Any) -> int:
        """
        Ingest the perturbed context as user messages.

        Each entry's perturbed_context is ingested as user messages.
        Long contexts are split into chunks of 25000 characters.
        """
        entry = cast(RefusalBenchEntry, item)
        user_peer = ctx.peers["user"]
        context = entry["perturbed_context"]

        # Store entry for later use in execute_questions
        ctx.peers["_entry"] = entry

        messages: list[MessageCreateParams] = []

        # Split long contexts
        if len(context) > 25000:
            chunks = [context[i : i + 25000] for i in range(0, len(context), 25000)]
            for chunk in chunks:
                messages.append(user_peer.message(chunk))
        else:
            messages.append(user_peer.message(context))

        # Batch messages (up to 100 per call)
        for i in range(0, len(messages), 100):
            batch = messages[i : i + 100]
            await ctx.session.aio.add_messages(batch)

        return len(messages)

    def get_dream_observers(self, item: Any) -> list[str]:
        """Return the observer - always user for RefusalBench."""
        return ["user"]

    async def execute_questions(self, ctx: ItemContext, item: Any) -> ItemResult:
        """
        Ask the perturbed query via dialectic and judge the response.

        Args:
            ctx: Item execution context.
            item: The RefusalBenchEntry being processed.

        Returns:
            ItemResult with evaluation details.
        """
        entry = cast(RefusalBenchEntry, item)
        workspace_id = ctx.workspace_id
        query = entry["perturbed_query"]

        start_time = time.time()

        result: ItemResult = {
            "unique_id": entry["unique_id"],
            "perturbation_class": entry["perturbation_class"],
            "intensity": entry["intensity"],
            "expected_behavior": entry["expected_rag_behavior"],
            "predicted_type": "ANSWER_CORRECTLY",
            "answer_quality_score": None,
            "refusal_match_correct": False,
            "is_correct": False,
            "judge_explanation": "",
            "actual_response": "",
            "query": query,
            "workspace_id": workspace_id,
            "error": None,
            "start_time": start_time,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        try:
            # Ask the perturbed query via dialectic chat
            print(f"  [{workspace_id}] Asking: {query[:100]}...")

            actual_response = await ctx.peers["user"].aio.chat(
                query,
                reasoning_level=self.config.reasoning_level,
            )
            actual_response = (
                actual_response if isinstance(actual_response, str) else ""
            )
            result["actual_response"] = actual_response

            # Judge the response
            predicted_type, quality_score, explanation = await judge_refusal_response(
                self.openrouter_client,
                self.judge_model,
                query,
                actual_response,
                entry["original_answers"],
            )

            result["predicted_type"] = predicted_type
            result["answer_quality_score"] = quality_score
            result["judge_explanation"] = explanation

            # Evaluate correctness
            is_correct = evaluate_single_item(
                entry, predicted_type, quality_score, explanation
            )
            result["is_correct"] = is_correct
            result["refusal_match_correct"] = (
                predicted_type == entry["expected_rag_behavior"]
            )

            status = "CORRECT" if is_correct else "WRONG"
            expected = entry["expected_rag_behavior"]
            print(
                f"  [{workspace_id}] Expected: {expected} | Got: {predicted_type} [{status}]"
            )

        except Exception as e:
            self.logger.exception("Error executing %s: %s", workspace_id, e)
            result["error"] = str(e)

        result["end_time"] = time.time()
        result["duration_seconds"] = result["end_time"] - result["start_time"]

        return result

    def print_summary(
        self, results: list[ItemResult], total_duration: float
    ) -> None:
        """Print summary using the common function."""
        print_summary(results, total_duration)

    def generate_output(
        self, results: list[ItemResult], total_duration: float
    ) -> None:
        """Generate JSON output file."""
        if self.config.json_output:
            output_file = self.config.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/refusalbench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            total_duration,
            output_file,
            metadata_extra={
                "data_file": str(self.data_file),
                "base_api_port": self.config.base_api_port,
                "pool_size": self.config.pool_size,
                "timeout_seconds": self.config.timeout_seconds,
                "reasoning_level": self.config.reasoning_level,
                "judge_model": self.judge_model,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
            },
        )


def main() -> int:
    """Main entry point for the RefusalBench test runner."""
    parser = argparse.ArgumentParser(
        description="Run RefusalBench benchmark tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --data-file tests/bench/refusalbench_data/refusalbench_sample.jsonl
  %(prog)s --perturbation-class P-Ambiguity
  %(prog)s --intensity HIGH --test-count 6
  %(prog)s --unique-id pamb_med_01
        """,
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).parent / "refusalbench_data" / "refusalbench_sample.jsonl",
        help="Path to JSONL dataset file (default: refusalbench_data/refusalbench_sample.jsonl)",
    )

    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Number of items to run (default: all)",
    )

    parser.add_argument(
        "--unique-id",
        type=str,
        default=None,
        help="Run a specific item by unique_id",
    )

    parser.add_argument(
        "--perturbation-class",
        type=str,
        default=None,
        choices=list(PERTURBATION_CLASSES.keys()),
        help="Filter by perturbation class",
    )

    parser.add_argument(
        "--intensity",
        type=str,
        default=None,
        choices=INTENSITY_LEVELS,
        help="Filter by intensity level (LOW, MEDIUM, HIGH)",
    )

    # Add common arguments shared across all runners
    add_common_arguments(parser)

    args = parser.parse_args()

    # Validate common arguments
    error = validate_common_arguments(args)
    if error:
        print(error)
        return 1

    # Validate data file exists
    if not args.data_file.exists():
        print(f"Error: Data file not found: {args.data_file}")
        return 1

    # Create config and runner
    config = RunnerConfig.from_args(args, default_timeout=600)

    runner = RefusalBenchRunner(
        config=config,
        data_file=args.data_file,
        test_count=args.test_count,
        unique_id=args.unique_id,
        perturbation_class=args.perturbation_class,
        intensity=args.intensity,
    )

    return runner.run_and_summarize()


if __name__ == "__main__":
    exit(main())
