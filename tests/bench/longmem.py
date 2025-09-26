"""
Honcho LongMemEval Test Runner

A script that executes longmemeval tests against a running Honcho instance.
This script:
1. Loads longmemeval test definitions from JSON files
2. Creates a workspace for each question (using question_id and question_type)
3. Creates sessions with haystack conversations
4. Adds the answer session if present
5. Waits for the deriver queue to be empty
6. Executes the question and judges the response using an LLM

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```
NOTE: you may create a .env file in this directory to customize honcho config. The harness will print the config it is using.

1. Run the test harness:
```
python -m tests.bench.harness
```

2. Choose a test file:
    should be formatted as longmemeval_data.
    see: https://github.com/xiaowu0162/LongMemEval?tab=readme-ov-file
    or: https://huggingface.co/datasets/xiaowu0162/longmemeval

3. Run this file with a selected test file:
```
python -m tests.bench.longmem --test-file tests/bench/longmemeval_data/longmemeval_oracle.json
```

Optional arguments:
```
--anthropic-api-key: Anthropic API key for response judging (can be set in .env as LLM_ANTHROPIC_API_KEY or provided as an argument)
--timeout: Timeout for deriver queue to empty in seconds (default: 10 minutes)
--honcho-url: URL of the running Honcho instance (default: http://localhost:8000)
--batch-size: Number of questions to run concurrently in each batch (default: 10)
--json-output: Path to write JSON summary results for analytics (if not provided, creates timestamped file in tests/bench/eval_results)
```

## Other notes
- Judge is Claude Sonnet 4
- If processing lots of data, set timeout very high or all will be lost
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)
from typing_extensions import TypedDict

from src.config import settings
from src.utils.metrics_collector import MetricsCollector

load_dotenv()


class SessionResult(TypedDict):
    """Type definition for session creation results."""

    name: str
    message_count: int


class QueryResult(TypedDict):
    """Type definition for query execution results."""

    question: str
    expected_answer: str
    actual_response: str
    judgment: dict[str, Any]


class TestResult(TypedDict):
    """Type definition for test execution results."""

    question_id: str
    question_type: str
    workspace_id: str
    sessions_created: list[SessionResult]
    query_executed: QueryResult | None
    passed: bool
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float
    output_lines: list[str]


class LongMemEvalRunner:
    """
    Executes longmemeval JSON tests against a Honcho instance.
    """

    def __init__(
        self,
        honcho_url: str = "http://localhost:8000",
        anthropic_api_key: str | None = None,
        timeout_seconds: int | None = None,
    ):
        """
        Initialize the test runner.

        Args:
            honcho_url: URL of the running Honcho instance
            anthropic_api_key: Anthropic API key for judging responses
        """
        self.honcho_url: str = honcho_url
        self.anthropic_api_key: str | None = anthropic_api_key
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 10000
        )

        # Initialize metrics collector
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.metrics_collector.start_collection(
            f"longmem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Suppress HTTP request logs from the Honcho SDK
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

        if self.anthropic_api_key:
            self.anthropic_client: AsyncAnthropic = AsyncAnthropic(
                api_key=self.anthropic_api_key
            )
        else:
            api_key = os.getenv("LLM_ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("LLM_ANTHROPIC_API_KEY is not set")
            self.anthropic_client = AsyncAnthropic(api_key=api_key)

    def _format_duration(self, total_seconds: float) -> str:
        """Format a duration in seconds into a human-readable string.

        If the duration is at least one minute, this returns a string in the
        form "XmYYs" with zero-padded seconds. Otherwise, it returns the
        duration in seconds with two decimal places, e.g., "12.34s".

        Args:
            total_seconds: The duration in seconds.

        Returns:
            A formatted duration string.
        """
        minutes = int(total_seconds // 60)
        if minutes > 0:
            seconds_rounded = int(round(total_seconds - minutes * 60))
            if seconds_rounded == 60:
                minutes += 1
                seconds_rounded = 0
            return f"{minutes}m{seconds_rounded:02d}s"
        return f"{total_seconds:.2f}s"

    def _parse_date(self, date_str: str) -> datetime:
        """Parse longmemeval date format to datetime.

        Args:
            date_str: Date string in format "YYYY/MM/DD (Day) HH:MM"

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If date format is invalid
        """
        try:
            # Extract the date and time parts, ignoring the day name in parentheses
            # Format: "2023/05/20 (Sat) 02:21"
            parts = date_str.split(") ")
            if len(parts) != 2:
                raise ValueError(f"Invalid date format: {date_str}")

            date_part = parts[0].split(" (")[0]  # "2023/05/20"
            time_part = parts[1]  # "02:21"

            # Combine and parse
            datetime_str = f"{date_part} {time_part}"
            return datetime.strptime(datetime_str, "%Y/%m/%d %H:%M")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse date '{date_str}': {e}") from e

    def load_test_file(self, test_file: Path) -> list[dict[str, Any]]:
        """
        Load longmemeval test definitions from a JSON file.

        Args:
            test_file: Path to the JSON test file

        Returns:
            List of test question dictionaries
        """
        with open(test_file) as f:
            return json.load(f)

    async def create_honcho_client(self, workspace_id: str) -> AsyncHoncho:
        """
        Create a Honcho client for a specific workspace.

        Args:
            workspace_id: Workspace ID for the test

        Returns:
            AsyncHoncho client instance
        """
        return AsyncHoncho(
            environment="local",
            workspace_id=workspace_id,
            base_url=self.honcho_url,
        )

    async def wait_for_deriver_queue_empty(
        self, honcho_client: AsyncHoncho, session_id: str | None = None
    ) -> bool:
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.get_deriver_status(session_id)
            except Exception as _e:
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

    async def judge_response(
        self, question: str, expected_answer: str, actual_response: str
    ) -> dict[str, Any]:
        """
        Use an LLM to judge if the actual response matches the expected answer.

        Args:
            question: The question asked
            expected_answer: Expected answer from the test
            actual_response: Actual response from Honcho

        Returns:
            Judgment result with pass/fail and reasoning
        """
        try:
            system_prompt = """
You are an expert judge evaluating AI responses to memory questions. Your task is to determine if an actual response contains the correct answer from long-term memory.

CRITICAL JUDGING PRINCIPLES:
1. SEMANTIC UNDERSTANDING: Focus on whether the actual response conveys the same core factual information as expected, even if expressed differently
2. FLEXIBLE INTERPRETATION: Accept responses that are longer, more detailed, or use different phrasing as long as they contain the correct answer
3. MEMORY ACCURACY: The key is whether the AI correctly recalled and stated the factual information from memory
4. PARTIAL CREDIT: If the response shows the AI accessed relevant memories but made minor errors in details, consider partial credit
5. IMPLICIT vs EXPLICIT: Accept responses that clearly imply the correct answer through context

ONLY FAIL when:
- The core factual answer is demonstrably wrong
- The response shows no evidence of accessing the relevant memory
- The AI explicitly states incorrect information that contradicts the expected answer

Always respond with valid JSON: {"passed": boolean, "reasoning": "short (1-3 sentences) explanation of why the response is correct or incorrect"}"""

            user_prompt = f"""Question: "{question}"
Expected answer: "{expected_answer}"
Actual response: "{actual_response}"

Evaluate whether the actual response correctly answers the question based on the expected answer. Focus on factual accuracy and evidence that the AI accessed the correct memory."""

            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            if not response.content:
                raise ValueError("Anthropic returned empty response")

            content_block = response.content[0]
            judgment_text = getattr(content_block, "text", None)
            if judgment_text is None:
                raise ValueError(
                    f"No text content in response block: {type(content_block)}"
                )

            # Extract JSON from the response if it's wrapped in markdown
            if "```json" in judgment_text:
                json_start = judgment_text.find("```json") + 7
                json_end = judgment_text.find("```", json_start)
                judgment_text = judgment_text[json_start:json_end].strip()
            elif "```" in judgment_text:
                json_start = judgment_text.find("```") + 3
                json_end = judgment_text.find("```", json_start)
                judgment_text = judgment_text[json_start:json_end].strip()

            judgment = json.loads(judgment_text)
            return judgment

        except Exception as e:
            self.logger.error(f"Error judging response: {e}")
            # Fallback to simple string matching
            is_correct = expected_answer.lower() in actual_response.lower()
            return {
                "passed": is_correct,
                "reasoning": f"Fallback string matching due to error: {'Match found' if is_correct else 'No match found'}",
            }

    async def execute_question(self, question_data: dict[str, Any]) -> TestResult:
        """
        Execute a single longmemeval question.

        Args:
            question_data: Dictionary containing question data

        Returns:
            Test execution results
        """
        question_id = question_data["question_id"]
        question_type = question_data["question_type"]
        question = question_data["question"]
        expected_answer = question_data["answer"]

        output_lines: list[str] = []
        output_lines.append(
            f"\033[1mExecuting question {question_id} ({question_type})\033[0m"
        )
        output_lines.append(f"Question: {question}")
        output_lines.append(f"Expected: {expected_answer}")

        # Create workspace for this question
        workspace_id = f"{question_id}_{question_type}"
        honcho_client = await self.create_honcho_client(workspace_id)

        results: TestResult = {
            "question_id": question_id,
            "question_type": question_type,
            "workspace_id": workspace_id,
            "sessions_created": [],
            "query_executed": None,
            "passed": False,
            "error": None,
            "start_time": time.time(),
            "end_time": 0.0,
            "duration_seconds": 0.0,
            "output_lines": output_lines,
        }

        try:
            user_peer = await honcho_client.peer(id="user")
            assistant_peer = await honcho_client.peer(id="assistant")

            # Process haystack sessions
            haystack_dates = question_data.get("haystack_dates", [])
            haystack_sessions = question_data.get("haystack_sessions", [])
            haystack_session_ids = question_data.get("haystack_session_ids", [])

            # Validate alignment of dates, session IDs, and sessions
            if len(haystack_dates) != len(haystack_sessions):
                raise ValueError(
                    f"Misaligned data: {len(haystack_dates)} dates but {len(haystack_sessions)} sessions"
                )
            if len(haystack_session_ids) != len(haystack_sessions):
                raise ValueError(
                    f"Misaligned data: {len(haystack_session_ids)} session IDs but {len(haystack_sessions)} sessions"
                )

            # Parse all dates upfront to catch parsing errors early
            parsed_dates: list[datetime] = []
            for date_str in haystack_dates:
                try:
                    parsed_dates.append(self._parse_date(date_str))
                except ValueError as e:
                    raise ValueError(f"Error parsing date '{date_str}': {e}") from e

            haystack_total_messages = sum(len(session) for session in haystack_sessions)

            print(
                f"[{workspace_id}] processing {len(haystack_sessions)} sessions with {haystack_total_messages} total messages"
            )

            # Determine which peer should be observed based on question type
            is_assistant_type = question_type == "single-session-assistant"

            # Zip together dates, session IDs, and session content
            for session_date, session_id, session_messages in zip(
                parsed_dates, haystack_session_ids, haystack_sessions, strict=True
            ):
                session = await honcho_client.session(id=session_id)

                # Configure peer observation based on question type
                if is_assistant_type:
                    # For assistant questions, observe the assistant peer
                    await session.add_peers(
                        [
                            (
                                user_peer,
                                SessionPeerConfig(
                                    observe_me=False, observe_others=False
                                ),
                            ),
                            (
                                assistant_peer,
                                SessionPeerConfig(
                                    observe_me=True, observe_others=False
                                ),
                            ),
                        ]
                    )
                else:
                    # For user questions, observe the user peer (default behavior)
                    await session.add_peers(
                        [
                            (
                                user_peer,
                                SessionPeerConfig(
                                    observe_me=True, observe_others=False
                                ),
                            ),
                            (
                                assistant_peer,
                                SessionPeerConfig(
                                    observe_me=False, observe_others=False
                                ),
                            ),
                        ]
                    )

                honcho_messages: list[MessageCreateParam] = []
                for msg in session_messages:
                    role = msg["role"]
                    content = msg["content"]

                    # Use the session date as the timestamp for all messages in this session
                    if role == "user":
                        honcho_messages.append(
                            user_peer.message(content, created_at=session_date)
                        )
                    elif role == "assistant":
                        honcho_messages.append(
                            assistant_peer.message(content, created_at=session_date)
                        )

                if honcho_messages:
                    await session.add_messages(honcho_messages)

                results["sessions_created"].append(
                    SessionResult(name=session_id, message_count=len(honcho_messages))
                )

            print(
                f"[{workspace_id}] fired all messages.\nwaiting for deriver queue to be empty... will time out in {self.timeout_seconds} seconds"
            )
            await asyncio.sleep(
                1
            )  # Give time for at least some tasks to be queued, so deriver queue size check doesn't immediately return 0

            queue_empty = await self.wait_for_deriver_queue_empty(honcho_client)
            if not queue_empty:
                output_lines.append("Deriver queue never emptied!!!")
                results["error"] = "Deriver queue timeout"
                return results

            # Execute the question
            output_lines.append(f"\nAsking question: {question}")

            try:
                # Use the appropriate peer based on question type
                if is_assistant_type:
                    # For assistant questions, use the assistant peer
                    actual_response = await assistant_peer.chat(question)
                else:
                    # For user questions, use the user peer (default behavior)
                    actual_response = await user_peer.chat(question)

                actual_response = actual_response if actual_response is not None else ""

                # Judge the response
                judgment = await self.judge_response(
                    question, expected_answer, actual_response
                )

                query_result: QueryResult = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_response": actual_response,
                    "judgment": judgment,
                }

                results["query_executed"] = query_result
                results["passed"] = judgment["passed"]

                output_lines.append(
                    "  judgment: \033[1m\033[32mPASS\033[0m"
                    if judgment["passed"]
                    else "  judgment: \033[1m\033[31mFAIL\033[0m"
                )
                if not judgment["passed"]:
                    output_lines.append(
                        f"  got response: \033[3m{actual_response}\033[0m"
                    )
                    output_lines.append(f"  expected: {expected_answer}")
                output_lines.append(f"  reasoning: {judgment['reasoning']}")

            except Exception as e:
                self.logger.error(f"Error executing question: {e}")
                query_result = QueryResult(
                    question=question,
                    expected_answer=expected_answer,
                    actual_response=f"ERROR: {e}",
                    judgment={
                        "passed": False,
                        "reasoning": f"Question execution failed: {e}",
                    },
                )
                results["query_executed"] = query_result
                results["passed"] = False

            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]

            output_lines.append(
                f"\nQuestion {question_id} completed. Status: {'PASS' if results['passed'] else 'FAIL'} (Duration: {self._format_duration(results['duration_seconds'])})"
            )

        except Exception as e:
            self.logger.error(f"Error executing question {question_id}: {e}")
            results["error"] = str(e)
            results["passed"] = False
            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]
            output_lines.append(f"Error executing question {question_id}: {e}")

        return results

    async def run_all_questions(
        self, test_file: Path, batch_size: int = 10
    ) -> tuple[list[TestResult], float]:
        """
        Run all questions in a longmemeval test file.

        Args:
            test_file: Path to the longmemeval JSON file
            batch_size: Number of questions to run concurrently in each batch

        Returns:
            Tuple of (list of test results, total duration)
        """
        questions = self.load_test_file(test_file)
        print(
            f"found {len(questions)} {'question' if len(questions) == 1 else 'questions'} in {test_file}"
        )

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
                *[self.execute_question(q) for q in batch]
            )

            # Print detailed per-question outputs for this batch
            for result in batch_results:
                print(f"\n{'=' * 60}")
                print("\n".join(result.get("output_lines", [])))
                print(f"{'=' * 60}\n")

            all_results.extend(batch_results)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        # Finalize metrics collection
        self.metrics_collector.finalize_collection()

        return all_results, overall_duration

    def print_summary(
        self, results: list[TestResult], total_elapsed_seconds: float | None = None
    ) -> None:
        """
        Print a summary of all test results.

        Args:
            results: List of test results
            total_elapsed_seconds: Total elapsed time
        """
        print(f"\n{'=' * 80}")
        print("LONGMEMEVAL TEST EXECUTION SUMMARY")
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
        print(f"Total Test Time: {self._format_duration(total_test_time)}")

        print("\nDetailed Results:")
        print(
            f"{'Question ID':<15} {'Type':<20} {'Status':<8} {'Duration':<10} {'Workspace ID':<30}"
        )
        print(f"{'-' * 15} {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 30}")

        for result in results:
            question_id = result["question_id"]
            question_type = result["question_type"]
            status = "PASS" if result.get("passed", False) else "FAIL"
            duration = self._format_duration(result["duration_seconds"])
            workspace = result["workspace_id"]

            print(
                f"{question_id:<15} {question_type:<20} {status:<8} {duration:<10} {workspace:<30}"
            )

        print(f"{'=' * 80}")

    def generate_json_summary(
        self,
        results: list[TestResult],
        test_file: Path,
        total_elapsed_seconds: float,
        output_file: Path | None = None,
    ) -> None:
        """
        Generate a comprehensive JSON summary of test results for analytics.

        Args:
            results: List of test results
            test_file: Path to the test file that was executed
            total_elapsed_seconds: Total elapsed time for all tests
            output_file: Optional path to write JSON output to
        """
        total_questions = len(results)
        passed_questions = sum(1 for r in results if r.get("passed", False))
        failed_questions = total_questions - passed_questions

        # Calculate statistics by question type
        type_stats: dict[str, dict[str, int | float]] = {}
        for result in results:
            q_type = result["question_type"]
            if q_type not in type_stats:
                type_stats[q_type] = {"total": 0, "passed": 0, "failed": 0}
            type_stats[q_type]["total"] += 1
            if result.get("passed", False):
                type_stats[q_type]["passed"] += 1
            else:
                type_stats[q_type]["failed"] += 1

        # Add success rates to type stats
        for q_type in type_stats:
            stats = type_stats[q_type]
            stats["success_rate"] = (
                (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            )

        # Calculate timing statistics
        durations = [r["duration_seconds"] for r in results]
        timing_stats = {
            "total_duration_seconds": total_elapsed_seconds,
            "individual_test_durations": {
                "min_seconds": min(durations) if durations else 0,
                "max_seconds": max(durations) if durations else 0,
                "mean_seconds": sum(durations) / len(durations) if durations else 0,
                "median_seconds": sorted(durations)[len(durations) // 2]
                if durations
                else 0,
            },
        }

        # Create the full summary
        summary = {
            "metadata": {
                "test_file": str(test_file),
                "execution_timestamp": datetime.now().isoformat(),
                "runner_version": "1.0.0",
                "honcho_url": self.honcho_url,
                "timeout_seconds": self.timeout_seconds,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
                "summary_settings": settings.SUMMARY.model_dump(),
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
            "detailed_results": [
                {
                    "question_id": result["question_id"],
                    "question_type": result["question_type"],
                    "workspace_id": result["workspace_id"],
                    "passed": result.get("passed", False),
                    "duration_seconds": result["duration_seconds"],
                    "start_time": result["start_time"],
                    "end_time": result["end_time"],
                    "error": result.get("error"),
                    "query_executed": result.get("query_executed"),
                }
                for result in results
            ],
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nJSON summary written to: {output_file}")


async def main() -> int:
    """
    Main entry point for the longmemeval test runner.
    """
    parser = argparse.ArgumentParser(
        description="Run longmemeval tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test-file tests/bench/longmemeval_data/longmemeval_s.json    # Run longmemeval tests
  %(prog)s --honcho-url http://localhost:8000                             # Custom Honcho URL
        """,
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        help="Path to longmemeval JSON file",
    )

    parser.add_argument(
        "--honcho-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the running Honcho instance (default: http://localhost:8000)",
    )

    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key for response judging (optional)",
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
        help="Number of questions to run concurrently in each batch (default: 10)",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to write JSON summary results for analytics (optional)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.test_file.exists():
        print(f"Error: Test file {args.test_file} does not exist")
        return 1

    if args.batch_size <= 0:
        print(f"Error: Batch size must be positive, got {args.batch_size}")
        return 1

    # Create test runner
    runner = LongMemEvalRunner(
        honcho_url=args.honcho_url,
        anthropic_api_key=args.anthropic_api_key,
        timeout_seconds=args.timeout,
    )

    try:
        # Run all questions
        results, total_elapsed = await runner.run_all_questions(
            args.test_file, args.batch_size
        )
        runner.print_summary(results, total_elapsed_seconds=total_elapsed)

        runner.metrics_collector.load_from_file(Path(settings.LOCAL_METRICS_FILE))

        # Print metrics summary
        runner.metrics_collector.print_summary()

        # Generate JSON output if requested
        if args.json_output:
            runner.generate_json_summary(
                results, args.test_file, total_elapsed, args.json_output
            )
        else:
            # Always generate a default JSON output file with timestamp
            default_output = Path(
                f"tests/bench/eval_results/longmemeval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            runner.generate_json_summary(
                results, args.test_file, total_elapsed, default_output
            )

        # Export metrics to JSON file
        metrics_output = Path(
            f"tests/bench/perf_metrics/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        runner.metrics_collector.export_to_json(metrics_output)
        runner.metrics_collector.cleanup_collection()

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
