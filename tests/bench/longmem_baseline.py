"""
LongMemEval Baseline Test Runner (Direct Claude Context)

A script that executes longmemeval tests directly against a model
by feeding the entire haystack content into the context window.

This serves as a baseline comparison against Honcho's memory framework.

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```

1. Run this file with a selected test file:
```
python -m tests.bench.longmem_baseline --test-file tests/bench/longmemeval_data/longmemeval_oracle.json
```

Optional arguments:
```
--anthropic-api-key: Anthropic API key (can be set in .env as LLM_ANTHROPIC_API_KEY)
--batch-size: Number of questions to run concurrently in each batch (default: 10)
--json-output: Path to write JSON summary results for analytics
--test-count: Number of tests to run (default: all)
--question-id: Run only the question with this question_id
```
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from typing_extensions import TypedDict

from src.config import settings

from .longmem_common import (
    calculate_timing_statistics,
    calculate_total_tokens,
    calculate_type_statistics,
    filter_questions,
    format_duration,
    judge_response,
    load_test_file,
    write_json_summary,
)

load_dotenv()


MODEL_BEING_TESTED = "claude-haiku-4-5"


class QueryResult(TypedDict):
    """Type definition for query execution results."""

    question: str
    expected_answer: str
    actual_response: str
    judgment: dict[str, Any]
    input_tokens: int
    output_tokens: int


class TestResult(TypedDict):
    """Type definition for test execution results."""

    question_id: str
    question_type: str
    query_executed: QueryResult | None
    passed: bool
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float
    total_context_tokens: int
    output_lines: list[str]


class LongMemEvalBaselineRunner:
    """
    Executes longmemeval tests directly against a model.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
    ):
        """
        Initialize the baseline test runner.

        Args:
            anthropic_api_key: Anthropic API key for API calls
        """
        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        if anthropic_api_key:
            self.anthropic_client: AsyncAnthropic = AsyncAnthropic(
                api_key=anthropic_api_key
            )
        else:
            api_key = os.getenv("LLM_ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("LLM_ANTHROPIC_API_KEY is not set")
            self.anthropic_client = AsyncAnthropic(api_key=api_key)

    def _format_conversation_context(
        self,
        haystack_sessions: list[list[dict[str, str]]],
        haystack_dates: list[str],
        _question_type: str,
    ) -> str:
        """
        Format haystack sessions into a conversation transcript for context.

        Args:
            haystack_sessions: List of sessions, each containing messages
            haystack_dates: List of date strings corresponding to sessions
            question_type: Type of question (used to determine perspective)

        Returns:
            Formatted conversation transcript string
        """
        lines: list[str] = []
        lines.append("=== CONVERSATION HISTORY ===\n")

        for session_idx, (session_messages, date_str) in enumerate(
            zip(haystack_sessions, haystack_dates, strict=True)
        ):
            lines.append(f"--- Session {session_idx + 1} ({date_str}) ---\n")

            for msg in session_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                role_label = "User" if role == "user" else "Assistant"
                lines.append(f"{role_label}: {content}\n")

            lines.append("")  # Blank line between sessions

        lines.append("=== END CONVERSATION HISTORY ===")
        return "\n".join(lines)

    async def execute_question(
        self, question_data: dict[str, Any], _question_index: int
    ) -> TestResult:
        """
        Execute a single longmemeval question by sending full context to Claude.

        Args:
            question_data: Dictionary containing question data
            question_index: Index of the question (for logging)

        Returns:
            Test execution results
        """
        question_id = question_data["question_id"]
        question_type = question_data["question_type"]
        question = question_data["question"]
        expected_answer = question_data["answer"]
        question_date = question_data.get("question_date", "")

        question_with_date = (
            f"[{question_date}] {question}" if question_date else question
        )

        output_lines: list[str] = []
        output_lines.append(
            f"\033[1mExecuting question {question_id} ({question_type})\033[0m"
        )
        output_lines.append(f"Question: {question_with_date}")
        output_lines.append(f"Expected: {expected_answer}")

        results: TestResult = {
            "question_id": question_id,
            "question_type": question_type,
            "query_executed": None,
            "passed": False,
            "error": None,
            "start_time": time.time(),
            "end_time": 0.0,
            "duration_seconds": 0.0,
            "total_context_tokens": 0,
            "output_lines": output_lines,
        }

        try:
            haystack_dates = question_data.get("haystack_dates", [])
            haystack_sessions = question_data.get("haystack_sessions", [])

            # Calculate total tokens
            total_context_tokens = calculate_total_tokens(haystack_sessions)
            results["total_context_tokens"] = total_context_tokens

            haystack_total_messages = sum(len(s) for s in haystack_sessions)
            output_lines.append(
                f"Context: {len(haystack_sessions)} sessions, {haystack_total_messages} messages, ~{total_context_tokens} tokens"
            )

            # Format conversation history as context
            conversation_context = self._format_conversation_context(
                haystack_sessions, haystack_dates, question_type
            )

            # Build system prompt based on question type
            if question_type == "single-session-assistant":
                perspective = "You are the assistant in these conversations."
            else:
                perspective = "You are helping a user recall information from their past conversations."

            system_prompt = f"""{perspective}

Below is a history of past conversations. Use this history to answer the user's question accurately.

{conversation_context}"""

            # Call Claude with full context, using prompt caching for the conversation context
            response = await self.anthropic_client.messages.create(
                model=MODEL_BEING_TESTED,
                max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": question_with_date,
                    }
                ],
            )

            if not response.content:
                raise ValueError("Anthropic returned empty response")

            content_block = response.content[0]
            actual_response = getattr(content_block, "text", "")

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            output_lines.append(
                f"  API usage: {input_tokens} input tokens, {output_tokens} output tokens"
            )

            # Judge the response
            judgment = await judge_response(
                self.anthropic_client,
                question_with_date,
                expected_answer,
                actual_response,
            )

            query_result: QueryResult = {
                "question": question_with_date,
                "expected_answer": expected_answer,
                "actual_response": actual_response,
                "judgment": judgment,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

            results["query_executed"] = query_result
            results["passed"] = judgment["passed"]

            output_lines.append(
                "  judgment: \033[1m\033[32mPASS\033[0m"
                if judgment["passed"]
                else "  judgment: \033[1m\033[31mFAIL\033[0m"
            )
            if not judgment["passed"]:
                output_lines.append(f"  got response: \033[3m{actual_response}\033[0m")
                output_lines.append(f"  expected: {expected_answer}")
            output_lines.append(f"  reasoning: {judgment['reasoning']}")

        except Exception as e:
            self.logger.error(f"Error executing question {question_id}: {e}")
            results["error"] = str(e)
            results["passed"] = False
            output_lines.append(f"Error executing question {question_id}: {e}")

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        output_lines.append(
            f"\nQuestion {question_id} completed. Status: {'PASS' if results['passed'] else 'FAIL'} (Duration: {format_duration(results['duration_seconds'])})"
        )

        return results

    async def run_all_questions(
        self,
        test_file: Path,
        batch_size: int = 10,
        test_count: int | None = None,
        question_id: str | None = None,
    ) -> tuple[list[TestResult], float]:
        """
        Run all questions in a longmemeval test file.

        Args:
            test_file: Path to the longmemeval JSON file
            batch_size: Number of questions to run concurrently in each batch
            test_count: Optional number of tests to run (runs first N tests)
            question_id: Optional question_id to run (skips all others)

        Returns:
            Tuple of (list of test results, total duration)
        """
        questions = load_test_file(test_file)
        questions = filter_questions(questions, test_file, question_id, test_count)
        if not questions:
            return [], 0.0

        print(
            f"found {len(questions)} {'question' if len(questions) == 1 else 'questions'} in {test_file}"
        )
        print("Running baseline test (direct Claude context, no Honcho)")

        overall_start = time.time()

        # Process questions in batches
        all_results: list[TestResult] = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(questions) + batch_size - 1) // batch_size

            print(f"\n{'=' * 60}")
            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} questions)"
            )
            print(f"{'=' * 60}")

            # Run questions in current batch concurrently
            batch_results: list[TestResult] = await asyncio.gather(
                *[self.execute_question(q, i + idx) for idx, q in enumerate(batch)]
            )

            # Print detailed per-question outputs for this batch
            for result in batch_results:
                print(f"\n{'=' * 60}")
                print("\n".join(result.get("output_lines", [])))
                print(f"{'=' * 60}\n")

            all_results.extend(batch_results)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        return all_results, overall_duration

    def print_summary(
        self, results: list[TestResult], total_elapsed_seconds: float | None = None
    ) -> None:
        """Print a summary of all test results."""
        print(f"\n{'=' * 80}")
        print("LONGMEMEVAL BASELINE TEST SUMMARY (Direct Claude Context)")
        print(f"{'=' * 80}")

        total_questions = len(results)
        passed_questions = sum(1 for r in results if r.get("passed", False))
        failed_questions = total_questions - passed_questions
        total_test_time = (
            total_elapsed_seconds
            if total_elapsed_seconds is not None
            else sum(r["duration_seconds"] for r in results)
        )

        print(f"Total Questions: {total_questions}")
        print(f"Passed: {passed_questions}")
        print(f"Failed: {failed_questions}")
        print(f"Success Rate: {(passed_questions / total_questions) * 100:.1f}%")
        print(f"Total Test Time: {format_duration(total_test_time)}")

        # Token usage statistics
        total_input_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0
        for result in results:
            total_context_tokens += result.get("total_context_tokens", 0)
            query = result.get("query_executed")
            if query:
                total_input_tokens += query.get("input_tokens", 0)
                total_output_tokens += query.get("output_tokens", 0)

        print("\nToken Usage:")
        print(f"  Total Context Tokens (estimated): {total_context_tokens:,}")
        print(f"  Total Input Tokens (API): {total_input_tokens:,}")
        print(f"  Total Output Tokens (API): {total_output_tokens:,}")

        print("\nDetailed Results:")
        print(
            f"{'Question ID':<15} {'Type':<25} {'Status':<8} {'Duration':<10} {'Input Tokens':<15}"
        )
        print(f"{'-' * 15} {'-' * 25} {'-' * 8} {'-' * 10} {'-' * 15}")

        for result in results:
            question_id = result["question_id"]
            question_type = result["question_type"]
            status = "PASS" if result.get("passed", False) else "FAIL"
            duration = format_duration(result["duration_seconds"])
            query = result.get("query_executed")
            input_tokens = query.get("input_tokens", 0) if query else 0

            print(
                f"{question_id:<15} {question_type:<25} {status:<8} {duration:<10} {input_tokens:<15,}"
            )

        print(f"{'=' * 80}")

    def generate_json_summary(
        self,
        results: list[TestResult],
        test_file: Path,
        total_elapsed_seconds: float,
        output_file: Path | None = None,
    ) -> None:
        """Generate a comprehensive JSON summary of test results."""
        total_questions = len(results)
        passed_questions = sum(1 for r in results if r.get("passed", False))
        failed_questions = total_questions - passed_questions

        # Calculate statistics by question type
        type_stats = calculate_type_statistics(results)

        # Calculate timing statistics
        timing_stats = calculate_timing_statistics(results, total_elapsed_seconds)

        # Calculate token usage statistics
        total_input_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0
        for result in results:
            total_context_tokens += result.get("total_context_tokens", 0)
            query = result.get("query_executed")
            if query:
                total_input_tokens += query.get("input_tokens", 0)
                total_output_tokens += query.get("output_tokens", 0)

        token_stats = {
            "total_context_tokens_estimated": total_context_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "mean_input_tokens": total_input_tokens / len(results) if results else 0,
        }

        summary = {
            "metadata": {
                "test_file": str(test_file),
                "execution_timestamp": datetime.now().isoformat(),
                "runner_type": "baseline_direct_context",
                "model": MODEL_BEING_TESTED,
            },
            "summary_statistics": {
                "total_questions": total_questions,
                "passed": passed_questions,
                "failed": failed_questions,
                "success_rate_percent": (passed_questions / total_questions) * 100
                if total_questions > 0
                else 0,
                "statistics_by_type": type_stats,
            },
            "timing": timing_stats,
            "token_usage": token_stats,
            "detailed_results": [
                {
                    "question_id": result["question_id"],
                    "question_type": result["question_type"],
                    "passed": result.get("passed", False),
                    "duration_seconds": result["duration_seconds"],
                    "start_time": result["start_time"],
                    "end_time": result["end_time"],
                    "total_context_tokens": result.get("total_context_tokens", 0),
                    "error": result.get("error"),
                    "query_executed": result.get("query_executed"),
                }
                for result in results
            ],
        }

        if output_file:
            write_json_summary(summary, output_file)


async def main() -> int:
    """Main entry point for the baseline test runner."""
    parser = argparse.ArgumentParser(
        description="Run longmemeval tests directly against Claude (baseline, no Honcho)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test-file tests/bench/longmemeval_data/longmemeval_s.json
  %(prog)s --test-file test.json --test-count 50
  %(prog)s --test-file test.json --question-id "q123"
        """,
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="Path to longmemeval JSON file (required)",
    )

    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key (optional, can use LLM_ANTHROPIC_API_KEY env var)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions to run concurrently in each batch (default: 10)",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to write JSON summary results for analytics (optional)",
    )

    parser.add_argument(
        "--test-count",
        type=int,
        help="Number of tests to run from the test file (default: all tests)",
    )

    parser.add_argument(
        "--question-id",
        type=str,
        help="Run only the question with this question_id (skips all others)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.test_file.exists():
        print(f"Error: Test file {args.test_file} does not exist")
        return 1

    if args.batch_size <= 0:
        print(f"Error: Batch size must be positive, got {args.batch_size}")
        return 1

    if args.test_count is not None and args.test_count <= 0:
        print(f"Error: Test count must be positive, got {args.test_count}")
        return 1

    # Create test runner
    runner = LongMemEvalBaselineRunner(
        anthropic_api_key=args.anthropic_api_key,
    )

    try:
        # Run all questions
        results, total_elapsed = await runner.run_all_questions(
            args.test_file, args.batch_size, args.test_count, args.question_id
        )
        runner.print_summary(results, total_elapsed_seconds=total_elapsed)

        # Generate JSON output
        if args.json_output:
            runner.generate_json_summary(
                results, args.test_file, total_elapsed, args.json_output
            )
        else:
            default_output = Path(
                f"tests/bench/eval_results/baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            runner.generate_json_summary(
                results, args.test_file, total_elapsed, default_output
            )

        # Return exit code based on results
        all_passed = all(r.get("passed", False) for r in results)
        return 0 if all_passed else 1

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
