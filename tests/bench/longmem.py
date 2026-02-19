"""
Honcho LongMemEval Test Runner

A script that executes longmemeval tests against a running Honcho instance.
This script:
1. Loads longmemeval test definitions from JSON files
2. Creates a workspace for each question (using question_id and question_type)
3. Creates sessions with haystack conversations
4. Adds the answer session if present
5. Waits for the deriver queue to be empty
6. Triggers a dream for memory consolidation
7. Executes the question and judges the response using an LLM

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
--base-api-port: Base port for Honcho API instances (default: 8000)
--pool-size: Number of Honcho instances in the pool (default: 1)
--batch-size: Number of questions to run concurrently in each batch (default: 10)
--json-output: Path to write JSON summary results for analytics (if not provided, creates timestamped file in tests/bench/eval_results)
--merge-sessions: Merge all sessions within a question into a single session (default: False)
--cleanup-workspace: Delete workspace after executing each question (default: False)
--use-get-context: Use get_context + judge LLM instead of dialectic .chat endpoint (default: False)
--question-id: Run only the question with this question_id (skips all others)
```

## Other notes
- Judge is GPT-4o (per LongMemEval paper)
- If processing lots of data, set timeout very high or all will be lost
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from honcho.api_types import MessageCreateParams
from honcho.session import SessionPeerConfig
from openai import AsyncOpenAI
from typing_extensions import TypedDict

from src.config import settings

from .longmem_common import (
    calculate_timing_statistics,
    calculate_total_tokens,
    calculate_type_statistics,
    filter_questions,
    judge_response,
    load_test_file,
    parse_longmemeval_date,
    write_json_summary,
)
from .runner_common import (
    BaseRunner,
    ItemContext,
    RunnerConfig,
    add_common_arguments,
    create_anthropic_client,
    create_openai_client,
    format_duration,
    validate_common_arguments,
)

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
    token_efficiency: dict[str, Any] | None


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


class LongMemEvalRunner(BaseRunner[TestResult]):
    """
    Executes longmemeval JSON tests against a Honcho instance.
    """

    def __init__(
        self,
        config: RunnerConfig,
        test_file: Path,
        anthropic_api_key: str | None = None,
        merge_sessions: bool = False,
        test_count: int | None = None,
        question_id: str | None = None,
    ):
        """
        Initialize the test runner.

        Args:
            config: Common runner configuration
            test_file: Path to the longmemeval JSON file
            anthropic_api_key: Anthropic API key for judging responses
            merge_sessions: If True, merge all sessions within a question into one session
            test_count: Optional number of tests to run (runs first N tests)
            question_id: Optional question_id to run (skips all others)
        """
        self.test_file: Path = test_file
        self.merge_sessions: bool = merge_sessions
        self.test_count: int | None = test_count
        self.question_id_filter: str | None = question_id

        # Initialize base class (sets up metrics collector and logger)
        super().__init__(config)

        # Initialize LLM clients
        self.anthropic_client: AsyncAnthropic = create_anthropic_client(
            anthropic_api_key
        )
        self.openai_client: AsyncOpenAI = create_openai_client()

    def get_metrics_prefix(self) -> str:
        return "longmem"

    def load_items(self) -> list[Any]:
        """Load questions from the test file."""
        questions = load_test_file(self.test_file)
        questions = filter_questions(
            questions, self.test_file, self.question_id_filter, self.test_count
        )
        return questions

    def get_workspace_id(self, item: Any) -> str:
        """Return workspace ID for a question."""
        return f"{item['question_id']}_{item['question_type']}"

    def get_session_id(self, item: Any, workspace_id: str) -> str:
        """Return session ID for a question."""
        if self.merge_sessions:
            return f"{workspace_id}_merged"
        # For non-merged, we use the first haystack session ID
        haystack_session_ids = item.get("haystack_session_ids", [])
        return (
            haystack_session_ids[0]
            if haystack_session_ids
            else f"{workspace_id}_session"
        )

    async def setup_peers(self, ctx: ItemContext, item: Any) -> None:
        """Create user and assistant peers."""
        ctx.peers["user"] = await ctx.honcho_client.aio.peer(id="user")
        ctx.peers["assistant"] = await ctx.honcho_client.aio.peer(id="assistant")

    async def setup_session(self, ctx: ItemContext, item: Any) -> None:
        """Create and configure session with appropriate observation settings."""
        is_assistant_type = item["question_type"] == "single-session-assistant"
        user_peer = ctx.peers["user"]
        assistant_peer = ctx.peers["assistant"]

        if self.merge_sessions:
            # Create a single merged session
            ctx.session = await ctx.honcho_client.aio.session(
                id=ctx.session_id, configuration=self._get_session_configuration()
            )

            if is_assistant_type:
                await ctx.session.aio.add_peers(
                    [
                        (
                            user_peer,
                            SessionPeerConfig(observe_me=False, observe_others=False),
                        ),
                        (
                            assistant_peer,
                            SessionPeerConfig(observe_me=True, observe_others=False),
                        ),
                    ]
                )
            else:
                await ctx.session.aio.add_peers(
                    [
                        (
                            user_peer,
                            SessionPeerConfig(observe_me=True, observe_others=False),
                        ),
                        (
                            assistant_peer,
                            SessionPeerConfig(observe_me=False, observe_others=False),
                        ),
                    ]
                )
        else:
            # Sessions are created during ingestion for non-merged mode
            ctx.session = None

    async def ingest_messages(self, ctx: ItemContext, item: Any) -> int:
        """Ingest haystack messages into session(s)."""
        is_assistant_type = item["question_type"] == "single-session-assistant"
        user_peer = ctx.peers["user"]
        assistant_peer = ctx.peers["assistant"]

        haystack_dates = item.get("haystack_dates", [])
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_session_ids = item.get("haystack_session_ids", [])

        # Parse dates
        parsed_dates = [parse_longmemeval_date(d) for d in haystack_dates]

        total_messages = 0

        if self.merge_sessions:
            # Collect all messages from all sessions
            all_messages: list[MessageCreateParams] = []
            for session_date, session_messages in zip(
                parsed_dates, haystack_sessions, strict=True
            ):
                for msg in session_messages:
                    role = msg["role"]
                    content = msg["content"]

                    # Split long messages
                    if len(content) > 25000:
                        chunks = [
                            content[i : i + 25000]
                            for i in range(0, len(content), 25000)
                        ]
                        for chunk in chunks:
                            if role == "user":
                                all_messages.append(
                                    user_peer.message(chunk, created_at=session_date)
                                )
                            elif role == "assistant":
                                all_messages.append(
                                    assistant_peer.message(
                                        chunk, created_at=session_date
                                    )
                                )
                    else:
                        if role == "user":
                            all_messages.append(
                                user_peer.message(content, created_at=session_date)
                            )
                        elif role == "assistant":
                            all_messages.append(
                                assistant_peer.message(content, created_at=session_date)
                            )

            # Add messages in batches
            for i in range(0, len(all_messages), 100):
                batch = all_messages[i : i + 100]
                await ctx.session.aio.add_messages(batch)

            total_messages = len(all_messages)
        else:
            # Create separate sessions for each haystack session
            for session_date, session_id, session_messages in zip(
                parsed_dates, haystack_session_ids, haystack_sessions, strict=True
            ):
                session = await ctx.honcho_client.aio.session(
                    id=session_id, configuration=self._get_session_configuration()
                )

                if is_assistant_type:
                    await session.aio.add_peers(
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
                    await session.aio.add_peers(
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

                honcho_messages: list[MessageCreateParams] = []
                for msg in session_messages:
                    role = msg["role"]
                    content = msg["content"]

                    if len(content) > 25000:
                        chunks = [
                            content[i : i + 25000]
                            for i in range(0, len(content), 25000)
                        ]
                        for chunk in chunks:
                            if role == "user":
                                honcho_messages.append(
                                    user_peer.message(chunk, created_at=session_date)
                                )
                            elif role == "assistant":
                                honcho_messages.append(
                                    assistant_peer.message(
                                        chunk, created_at=session_date
                                    )
                                )
                    else:
                        if role == "user":
                            honcho_messages.append(
                                user_peer.message(content, created_at=session_date)
                            )
                        elif role == "assistant":
                            honcho_messages.append(
                                assistant_peer.message(content, created_at=session_date)
                            )

                for i in range(0, len(honcho_messages), 100):
                    batch = honcho_messages[i : i + 100]
                    await session.aio.add_messages(batch)

                total_messages += len(honcho_messages)

        return total_messages

    def get_dream_observers(self, item: Any) -> list[str]:
        """Return the observer based on question type."""
        is_assistant_type = item["question_type"] == "single-session-assistant"
        return ["assistant"] if is_assistant_type else ["user"]

    def _get_latest_input_tokens_used(self) -> int | None:
        """Get the uncached input tokens from the most recent dialectic_chat metric."""
        metrics_file = Path(settings.LOCAL_METRICS_FILE)
        if not metrics_file.exists():
            return None

        try:
            with open(metrics_file) as f:
                lines = f.readlines()

            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name", "")
                    if task_name.startswith("dialectic_chat_"):
                        for metric in data.get("metrics", []):
                            metric_name = metric.get("name", "")
                            if metric_name.endswith("uncached_input_tokens"):
                                return int(metric.get("value", 0))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        except Exception as e:
            self.logger.warning(f"Error reading metrics file: {e}")

        return None

    async def execute_questions(self, ctx: ItemContext, item: Any) -> TestResult:
        """Execute the question and judge the response."""
        start_time = time.time()
        workspace_id = ctx.workspace_id

        question_id = item["question_id"]
        question_type = item["question_type"]
        question = item["question"]
        expected_answer = item["answer"]
        question_date = item.get("question_date", "")

        question_with_date = (
            f"[{question_date}] {question}" if question_date else question
        )
        is_assistant_type = question_type == "single-session-assistant"

        # Calculate total tokens for efficiency metrics
        haystack_sessions = item.get("haystack_sessions", [])
        total_available_tokens = calculate_total_tokens(haystack_sessions)

        result: TestResult = {
            "question_id": question_id,
            "question_type": question_type,
            "workspace_id": workspace_id,
            "sessions_created": [],  # Populated during ingestion tracking
            "query_executed": None,
            "passed": False,
            "error": None,
            "start_time": start_time,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        try:
            print(f"  Asking: {question_with_date}")

            if self.config.use_get_context:
                # Use get_context instead of dialectic .chat endpoint
                if not self.merge_sessions or ctx.session is None:
                    raise ValueError("Merged session required for get_context mode")

                peer_id = "assistant" if is_assistant_type else "user"
                context = await ctx.session.aio.context(
                    summary=True,
                    peer_target=peer_id,
                    search_query=question,
                )

                context_messages = context.to_anthropic(assistant="assistant")
                context_messages.append({"role": "user", "content": question_with_date})

                response = await self.anthropic_client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=1024,
                    messages=cast(list[MessageParam], context_messages),
                )

                if not response.content:
                    raise ValueError("Anthropic returned empty response")

                actual_response = getattr(response.content[0], "text", "")
            else:
                # Use dialectic .chat endpoint
                peer = (
                    ctx.peers["assistant"] if is_assistant_type else ctx.peers["user"]
                )
                actual_response = await peer.aio.chat(
                    question_with_date,
                    reasoning_level=self.config.reasoning_level,
                )
                actual_response = (
                    actual_response if isinstance(actual_response, str) else ""
                )

            # Get token efficiency
            input_tokens_used = self._get_latest_input_tokens_used()
            token_efficiency = None
            if input_tokens_used is not None and total_available_tokens > 0:
                efficiency_ratio = input_tokens_used / total_available_tokens
                token_efficiency = {
                    "total_available_tokens": total_available_tokens,
                    "tokens_used": input_tokens_used,
                    "efficiency_ratio": efficiency_ratio,
                }
                print(
                    f"  Token efficiency: {efficiency_ratio:.4f} ({input_tokens_used}/{total_available_tokens})"
                )

            # Judge the response
            judgment = await judge_response(
                self.openai_client,
                question_with_date,
                expected_answer,
                actual_response,
                question_type,
                question_id,
            )

            result["query_executed"] = QueryResult(
                question=question_with_date,
                expected_answer=expected_answer,
                actual_response=actual_response,
                judgment=judgment,
                token_efficiency=token_efficiency,
            )
            result["passed"] = judgment["passed"]

            status = (
                "\033[1m\033[32mPASS\033[0m"
                if judgment["passed"]
                else "\033[1m\033[31mFAIL\033[0m"
            )
            print(f"  Judgment: {status}")
            if not judgment["passed"]:
                print(f"  Got: \033[3m{actual_response}\033[0m")
                print(f"  Expected: {expected_answer}")
            print(f"  Reasoning: {judgment['reasoning']}")

        except Exception as e:
            self.logger.error(f"Error executing question: {e}")
            result["query_executed"] = QueryResult(
                question=question_with_date,
                expected_answer=expected_answer,
                actual_response=f"ERROR: {e}",
                judgment={
                    "passed": False,
                    "reasoning": f"Question execution failed: {e}",
                },
                token_efficiency=None,
            )
            result["passed"] = False
            result["error"] = str(e)

        result["end_time"] = time.time()
        result["duration_seconds"] = result["end_time"] - result["start_time"]

        return result

    def print_summary(self, results: list[TestResult], total_duration: float) -> None:
        """Print a summary of all test results."""
        print(f"\n{'=' * 80}")
        print("LONGMEMEVAL TEST EXECUTION SUMMARY")
        print(f"{'=' * 80}")

        total_questions = len(results)
        passed_questions = sum(1 for r in results if r.get("passed", False))
        failed_questions = total_questions - passed_questions

        print(f"Total Questions: {total_questions}")
        print(f"Passed: {passed_questions}")
        print(f"Failed: {failed_questions}")
        print(
            f"Success Rate: {(passed_questions / total_questions) * 100:.1f}%"
            if total_questions > 0
            else "N/A"
        )
        print(f"Total Test Time: {format_duration(total_duration)}")

        # Token efficiency stats
        efficiency_ratios: list[Any] = []
        for result in results:
            query = result.get("query_executed")
            if query:
                token_eff = query.get("token_efficiency")
                if token_eff:
                    efficiency_ratios.append(token_eff["efficiency_ratio"])

        if efficiency_ratios:
            avg_efficiency = sum(efficiency_ratios) / len(efficiency_ratios)
            print("\nToken Efficiency:")
            print(
                f"  Average: {avg_efficiency:.4f} ({avg_efficiency * 100:.2f}% of available tokens used)"
            )
            print(f"  Min: {min(efficiency_ratios):.4f}")
            print(f"  Max: {max(efficiency_ratios):.4f}")

        print("\nDetailed Results:")
        print(f"{'Question ID':<15} {'Type':<20} {'Status':<8} {'Duration':<10}")
        print(f"{'-' * 15} {'-' * 20} {'-' * 8} {'-' * 10}")

        for result in results:
            question_id = result["question_id"]
            question_type = result["question_type"]
            status = "PASS" if result.get("passed", False) else "FAIL"
            duration = format_duration(result["duration_seconds"])
            print(f"{question_id:<15} {question_type:<20} {status:<8} {duration:<10}")

        print(f"{'=' * 80}")

    def generate_output(self, results: list[TestResult], total_duration: float) -> None:
        """Generate JSON output file."""
        total_questions = len(results)
        passed_questions = sum(1 for r in results if r.get("passed", False))

        # Calculate statistics
        type_stats = calculate_type_statistics(results)
        timing_stats = calculate_timing_statistics(results, total_duration)

        # Token efficiency stats
        efficiency_ratios: list[Any] = []
        total_available_tokens_list: list[Any] = []
        tokens_used_list: list[Any] = []
        for result in results:
            query = result.get("query_executed")
            if query:
                eff = query.get("token_efficiency")
                if eff:
                    efficiency_ratios.append(eff["efficiency_ratio"])
                    total_available_tokens_list.append(eff["total_available_tokens"])
                    tokens_used_list.append(eff["tokens_used"])

        token_efficiency_stats = None
        if efficiency_ratios:
            token_efficiency_stats = {
                "mean_efficiency_ratio": sum(efficiency_ratios)
                / len(efficiency_ratios),
                "min_efficiency_ratio": min(efficiency_ratios),
                "max_efficiency_ratio": max(efficiency_ratios),
                "median_efficiency_ratio": sorted(efficiency_ratios)[
                    len(efficiency_ratios) // 2
                ],
                "mean_tokens_available": sum(total_available_tokens_list)
                / len(total_available_tokens_list),
                "mean_tokens_used": sum(tokens_used_list) / len(tokens_used_list),
                "total_questions_with_metrics": len(efficiency_ratios),
            }

        summary = {
            "metadata": {
                "test_file": str(self.test_file),
                "execution_timestamp": datetime.now().isoformat(),
                "runner_version": "2.0.0",
                "base_api_port": self.config.base_api_port,
                "pool_size": self.config.pool_size,
                "timeout_seconds": self.config.timeout_seconds,
                "reasoning_level": self.config.reasoning_level,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
                "summary_settings": settings.SUMMARY.model_dump(),
            },
            "summary_statistics": {
                "total_questions": total_questions,
                "passed": passed_questions,
                "failed": total_questions - passed_questions,
                "success_rate_percent": (passed_questions / total_questions) * 100
                if total_questions > 0
                else 0,
                "statistics_by_type": type_stats,
            },
            "timing": timing_stats,
            "token_efficiency": token_efficiency_stats,
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

        # Determine output file
        if self.config.json_output:
            output_file = self.config.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/longmemeval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        write_json_summary(summary, output_file)


def main() -> int:
    """Main entry point for the longmemeval test runner."""
    parser = argparse.ArgumentParser(
        description="Run longmemeval tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test-file tests/bench/longmemeval_data/longmemeval_s.json    # Run longmemeval tests
  %(prog)s --test-file test.json --pool-size 4                            # Use 4 Honcho instances
  %(prog)s --test-file test.json --base-api-port 8000 --pool-size 4       # Custom base port with pool
  %(prog)s --test-file test.json --test-count 50                          # Run only first 50 tests
  %(prog)s --test-file test.json --question-id "q123"                    # Run only question with ID "q123"
  %(prog)s --test-file test.json --reasoning-level high                   # Use high reasoning level
        """,
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="Path to longmemeval JSON file (required)",
    )

    # Add common arguments shared across all runners
    add_common_arguments(parser)

    # LongMemEval-specific arguments
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key for response judging (optional)",
    )

    parser.add_argument(
        "--merge-sessions",
        action="store_true",
        help="Merge all sessions within a question into a single session (default: False)",
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

    # Validate common arguments
    error = validate_common_arguments(args)
    if error:
        print(error)
        return 1

    # Validate longmem-specific arguments
    if not args.test_file.exists():
        print(f"Error: Test file {args.test_file} does not exist")
        return 1

    if args.test_count is not None and args.test_count <= 0:
        print(f"Error: Test count must be positive, got {args.test_count}")
        return 1

    # Create config and runner
    config = RunnerConfig.from_args(args, default_timeout=10000)

    runner = LongMemEvalRunner(
        config=config,
        test_file=args.test_file,
        anthropic_api_key=args.anthropic_api_key,
        merge_sessions=args.merge_sessions,
        test_count=args.test_count,
        question_id=args.question_id,
    )

    return runner.run_and_summarize()


if __name__ == "__main__":
    exit(main())
