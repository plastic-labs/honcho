"""
RefusalBench Baseline Test Runner (Direct LLM Context)

A script that executes RefusalBench benchmark tests directly against an LLM
by feeding the context and query with refusal code instructions in a single prompt.

This serves as a baseline comparison against Honcho's memory framework.
Based on the RefusalBench methodology (arXiv:2510.10390).

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```

1. Run with the sample dataset:
```
python -m tests.bench.refusalbench_baseline
```

Optional arguments:
```
--data-file: Path to JSONL dataset (default: tests/bench/refusalbench_data/refusalbench_sample.jsonl)
--test-count: Number of items to run (default: all)
--unique-id: Run a specific item by unique_id
--perturbation-class: Filter by perturbation class
--intensity: Filter by intensity level (LOW, MEDIUM, HIGH)
--batch-size: Number of items to run concurrently (default: 5)
--json-output: Path to write JSON results
```

## Other notes
- Uses OpenRouter API (configured via LLM_OPENAI_COMPATIBLE_API_KEY in tests/bench/.env)
- Default answer model is anthropic/claude-haiku-4.5
- Default judge model is anthropic/claude-sonnet-4.5
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .refusalbench_common import (
    INTENSITY_LEVELS,
    PERTURBATION_CLASSES,
    VALID_REFUSAL_CODES,
    ItemResult,
    RefusalBenchEntry,
    evaluate_single_item,
    generate_json_summary,
    judge_refusal_response,
    load_refusalbench_data,
    print_summary,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")

# OpenRouter model for baseline testing
MODEL_BEING_TESTED = "anthropic/claude-haiku-4.5"


class RefusalBenchBaselineRunner:
    """Executes RefusalBench benchmark tests directly against an LLM."""

    def __init__(self, data_file: Path):
        """
        Initialize the RefusalBench baseline test runner.

        Args:
            data_file: Path to the JSONL dataset file.
        """
        self.data_file: Path = data_file

        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Initialize OpenRouter client
        openrouter_api_key = os.getenv("LLM_OPENAI_COMPATIBLE_API_KEY")
        openrouter_base_url = os.getenv(
            "LLM_OPENAI_COMPATIBLE_BASE_URL", "https://openrouter.ai/api/v1"
        )

        if not openrouter_api_key:
            raise ValueError(
                "LLM_OPENAI_COMPATIBLE_API_KEY is not set in tests/bench/.env"
            )

        self.openrouter_client: AsyncOpenAI = AsyncOpenAI(
            api_key=openrouter_api_key,
            base_url=openrouter_base_url,
        )

        # Model to use for judging
        self.judge_model: str = os.getenv(
            "REFUSALBENCH_JUDGE_MODEL", "anthropic/claude-sonnet-4.5"
        )

        # Model to use for answering
        self.answer_model: str = MODEL_BEING_TESTED

    def _build_refusalbench_prompt(self, entry: RefusalBenchEntry) -> str:
        """
        Build the RefusalBench prompt template with context, query, and refusal codes.

        This follows the original RefusalBench paper's prompt design where the
        system is given explicit refusal codes to choose from.

        Args:
            entry: The dataset entry.

        Returns:
            Formatted prompt string.
        """
        refusal_codes_formatted = "\n".join(
            [f"- {code}" for code in VALID_REFUSAL_CODES]
        )

        return f"""You are a question-answering system with access to the following context.

## Context
{entry['perturbed_context']}

## Question
{entry['perturbed_query']}

## Instructions
Answer the question based ONLY on the provided context. If you cannot answer the question correctly and confidently, you MUST refuse by responding with exactly one of the following refusal codes:

{refusal_codes_formatted}

Guidelines for when to refuse:
- REFUSE_AMBIGUOUS_QUERY: The question is ambiguous or could refer to multiple things in the context
- REFUSE_CONTRADICTORY_CONTEXT: The context contains contradictory information relevant to the question
- REFUSE_INFO_MISSING_IN_CONTEXT: The information needed to answer is not present in the context
- REFUSE_FALSE_PREMISE_IN_QUERY: The question contains a false assumption not supported by the context
- REFUSE_GRANULARITY_MISMATCH: The question asks for a level of detail the context cannot support
- REFUSE_NONFACTUAL_QUERY: The question asks for opinion, prediction, or speculation rather than facts

If you CAN answer correctly and confidently, provide a direct answer without any refusal code."""

    async def _process_single_item(
        self,
        entry: RefusalBenchEntry,
        semaphore: asyncio.Semaphore,
    ) -> ItemResult:
        """Process a single RefusalBench item."""
        async with semaphore:
            unique_id = entry["unique_id"]
            start_time = time.time()

            result: ItemResult = {
                "unique_id": unique_id,
                "perturbation_class": entry["perturbation_class"],
                "intensity": entry["intensity"],
                "expected_behavior": entry["expected_rag_behavior"],
                "predicted_type": "ANSWER_CORRECTLY",
                "answer_quality_score": None,
                "refusal_match_correct": False,
                "is_correct": False,
                "judge_explanation": "",
                "actual_response": "",
                "query": entry["perturbed_query"],
                "workspace_id": f"baseline_{unique_id}",
                "error": None,
                "start_time": start_time,
                "end_time": 0.0,
                "duration_seconds": 0.0,
            }

            try:
                print(f"  [{unique_id}] Asking: {entry['perturbed_query'][:80]}...")

                # Build prompt with refusal codes
                prompt = self._build_refusalbench_prompt(entry)

                # Call answer model
                response = await self.openrouter_client.chat.completions.create(
                    model=self.answer_model,
                    max_tokens=1000,
                    temperature=0.0,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )

                if not response.choices or not response.choices[0].message.content:
                    actual_response = ""
                else:
                    actual_response = response.choices[0].message.content

                result["actual_response"] = actual_response

                # Check if the model responded with an explicit refusal code
                response_stripped = actual_response.strip()
                if response_stripped in VALID_REFUSAL_CODES:
                    # Direct refusal code match
                    predicted_type = response_stripped
                    quality_score: float | None = None
                    explanation = f"Model returned explicit refusal code: {predicted_type}"
                else:
                    # Use judge to classify the response
                    predicted_type, quality_score, explanation = (
                        await judge_refusal_response(
                            self.openrouter_client,
                            self.judge_model,
                            entry["perturbed_query"],
                            actual_response,
                            entry["original_answers"],
                        )
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
                    f"  [{unique_id}] Expected: {expected} | Got: {predicted_type} [{status}]"
                )

            except Exception as e:
                self.logger.exception(
                    "Error processing %s: %s", unique_id, e
                )
                result["error"] = str(e)

            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - result["start_time"]

            return result

    async def run_items(
        self,
        entries: list[RefusalBenchEntry],
        batch_size: int = 5,
    ) -> tuple[list[ItemResult], float]:
        """
        Run multiple RefusalBench items.

        Args:
            entries: List of dataset entries to process.
            batch_size: Number of items to run concurrently in each batch.

        Returns:
            Tuple of (list of results, total duration).
        """
        print(
            f"Running {len(entries)} RefusalBench items [BASELINE - {self.answer_model}]"
        )

        overall_start = time.time()
        all_results: list[ItemResult] = []

        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(entries) + batch_size - 1) // batch_size

            print(f"\n{'=' * 60}")
            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)"
            )
            print(f"{'=' * 60}")

            semaphore = asyncio.Semaphore(batch_size)
            batch_results = await asyncio.gather(
                *[self._process_single_item(entry, semaphore) for entry in batch]
            )

            all_results.extend(batch_results)

        overall_duration = time.time() - overall_start
        return all_results, overall_duration


async def main() -> int:
    """Main entry point for the RefusalBench baseline test runner."""
    parser = argparse.ArgumentParser(
        description="Run RefusalBench benchmark tests directly against an LLM (baseline, no Honcho)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of items to run concurrently in each batch (default: 5)",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to write JSON summary results for analytics (optional)",
    )

    args = parser.parse_args()

    # Validate data file
    if not args.data_file.exists():
        print(f"Error: Data file not found: {args.data_file}")
        return 1

    # Load and filter data
    entries = load_refusalbench_data(args.data_file)

    if args.unique_id:
        entries = [e for e in entries if e["unique_id"] == args.unique_id]

    if args.perturbation_class:
        entries = [
            e for e in entries if e["perturbation_class"] == args.perturbation_class
        ]

    if args.intensity:
        entries = [e for e in entries if e["intensity"] == args.intensity]

    if args.test_count is not None:
        entries = entries[: args.test_count]

    if not entries:
        print("No items to process after filtering")
        return 1

    # Create runner and execute
    runner = RefusalBenchBaselineRunner(data_file=args.data_file)

    try:
        results, total_elapsed = await runner.run_items(entries, args.batch_size)

        print_summary(results, total_elapsed)

        # Generate JSON output
        if args.json_output:
            output_file = args.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/refusalbench_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            total_elapsed,
            output_file,
            metadata_extra={
                "runner_type": "baseline_direct_context",
                "answer_model": MODEL_BEING_TESTED,
                "data_file": str(args.data_file),
            },
        )

        return 0

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
