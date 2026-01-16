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
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import httpx
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from honcho import Honcho
from honcho.api_types import MessageCreateParams
from honcho.session import SessionPeerConfig
from openai import AsyncOpenAI
from typing_extensions import TypedDict

from src.config import settings
from src.utils.metrics_collector import MetricsCollector

from .longmem_common import (
    calculate_timing_statistics,
    calculate_total_tokens,
    calculate_type_statistics,
    filter_questions,
    format_duration,
    judge_response,
    load_test_file,
    parse_longmemeval_date,
    write_json_summary,
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
    output_lines: list[str]


class LongMemEvalRunner:
    """
    Executes longmemeval JSON tests against a Honcho instance.
    """

    def __init__(
        self,
        base_api_port: int = 8000,
        pool_size: int = 1,
        anthropic_api_key: str | None = None,
        timeout_seconds: int | None = None,
        merge_sessions: bool = False,
        cleanup_workspace: bool = False,
        use_get_context: bool = False,
    ):
        """
        Initialize the test runner.

        Args:
            base_api_port: Base port for Honcho API instances (default: 8000)
            pool_size: Number of Honcho instances in the pool (default: 1)
            anthropic_api_key: Anthropic API key for judging responses
            timeout_seconds: Timeout for deriver queue in seconds
            merge_sessions: If True, merge all sessions within a question into one session
            cleanup_workspace: If True, delete workspace after executing question (default: False)
            use_get_context: If True, use get_context + judge LLM instead of dialectic .chat endpoint
        """
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.anthropic_api_key: str | None = anthropic_api_key
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 10000
        )
        self.merge_sessions: bool = merge_sessions
        self.cleanup_workspace: bool = cleanup_workspace
        self.use_get_context: bool = use_get_context

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

        # OpenAI client for GPT-4o judge (per LongMemEval paper)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set (required for GPT-4o judge)")
        self.openai_client: AsyncOpenAI = AsyncOpenAI(api_key=openai_api_key)

    def get_honcho_url_for_index(self, question_index: int) -> str:
        """
        Get the Honcho URL for a given question index using round-robin distribution.

        Args:
            question_index: Index of the question in the test file

        Returns:
            URL of the Honcho instance to use for this question
        """
        instance_id = question_index % self.pool_size
        port = self.base_api_port + instance_id
        return f"http://localhost:{port}"

    def _get_latest_input_tokens_used(self) -> int | None:
        """Get the uncached input tokens from the most recent dialectic_chat metric.

        Returns:
            Number of tokens used, or None if not found
        """
        metrics_file = Path(settings.LOCAL_METRICS_FILE)
        if not metrics_file.exists():
            return None

        # Read the file and find the most recent dialectic_chat metric
        try:
            with open(metrics_file) as f:
                lines = f.readlines()

            # Search backwards through the file for the most recent dialectic_chat
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

    def create_honcho_client(self, workspace_id: str, honcho_url: str) -> Honcho:
        """
        Create a Honcho client for a specific workspace.

        Args:
            workspace_id: Workspace ID for the test
            honcho_url: URL of the Honcho instance

        Returns:
            Honcho client instance
        """
        return Honcho(
            environment="local",
            workspace_id=workspace_id,
            base_url=honcho_url,
        )

    async def wait_for_deriver_queue_empty(
        self, honcho_client: Honcho, session_id: str | None = None
    ) -> bool:
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.aio.queue_status(session=session_id)
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

        # Trigger the dream via API
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
        success = await self.wait_for_deriver_queue_empty(honcho_client)
        if success:
            print(f"[{workspace_id}] Dream queue empty")
        else:
            print(f"[{workspace_id}] Dream queue timeout")
        return success

    async def execute_question(
        self, question_data: dict[str, Any], honcho_url: str
    ) -> TestResult:
        """
        Execute a single longmemeval question.

        Args:
            question_data: Dictionary containing question data
            honcho_url: URL of the Honcho instance to use

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
        output_lines.append(f"Using Honcho instance: {honcho_url}")

        # Create workspace for this question
        workspace_id = f"{question_id}_{question_type}"
        honcho_client = self.create_honcho_client(workspace_id, honcho_url)

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
            user_peer = await honcho_client.aio.peer(id="user")
            assistant_peer = await honcho_client.aio.peer(id="assistant")

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
                    parsed_dates.append(parse_longmemeval_date(date_str))
                except ValueError as e:
                    raise ValueError(f"Error parsing date '{date_str}': {e}") from e

            haystack_total_messages = sum(len(session) for session in haystack_sessions)

            # Calculate total tokens available in the sessions for this question
            total_available_tokens = calculate_total_tokens(haystack_sessions)

            print(
                f"[{workspace_id}] processing {len(haystack_sessions)} sessions with {haystack_total_messages} total messages ({total_available_tokens} total tokens)"
            )

            # Determine which peer should be observed based on question type
            is_assistant_type = question_type == "single-session-assistant"

            # Initialize merged_session_id for potential use in dream trigger
            merged_session_id: str | None = None

            if self.merge_sessions:
                # Create a single merged session for all messages
                merged_session_id = f"{workspace_id}_merged"
                session = await honcho_client.aio.session(id=merged_session_id)

                # Configure peer observation based on question type
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

                # Collect all messages from all sessions in chronological order
                all_messages: list[MessageCreateParams] = []
                for session_date, session_messages in zip(
                    parsed_dates, haystack_sessions, strict=True
                ):
                    for msg in session_messages:
                        role = msg["role"]
                        content = msg["content"]

                        # Split message if it exceeds 25000 characters
                        if len(content) > 25000:
                            chunks = [
                                content[i : i + 25000]
                                for i in range(0, len(content), 25000)
                            ]
                            for chunk in chunks:
                                if role == "user":
                                    all_messages.append(
                                        user_peer.message(
                                            chunk, created_at=session_date
                                        )
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
                                    assistant_peer.message(
                                        content, created_at=session_date
                                    )
                                )

                # Add messages in batches of 100 (max supported by add_messages)
                if all_messages:
                    for i in range(0, len(all_messages), 100):
                        batch = all_messages[i : i + 100]
                        await session.aio.add_messages(batch)

                results["sessions_created"].append(
                    SessionResult(
                        name=merged_session_id, message_count=len(all_messages)
                    )
                )
            else:
                # create separate sessions
                # Zip together dates, session IDs, and session content
                for session_date, session_id, session_messages in zip(
                    parsed_dates, haystack_session_ids, haystack_sessions, strict=True
                ):
                    session = await honcho_client.aio.session(id=session_id)

                    # Configure peer observation based on question type
                    if is_assistant_type:
                        # For assistant questions, observe the assistant peer
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
                        # For user questions, observe the user peer (default behavior)
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

                        # Split message if it exceeds 25000 characters
                        if len(content) > 25000:
                            chunks = [
                                content[i : i + 25000]
                                for i in range(0, len(content), 25000)
                            ]
                            for chunk in chunks:
                                # Use the session date as the timestamp for all messages in this session
                                if role == "user":
                                    honcho_messages.append(
                                        user_peer.message(
                                            chunk, created_at=session_date
                                        )
                                    )
                                elif role == "assistant":
                                    honcho_messages.append(
                                        assistant_peer.message(
                                            chunk, created_at=session_date
                                        )
                                    )
                        else:
                            # Use the session date as the timestamp for all messages in this session
                            if role == "user":
                                honcho_messages.append(
                                    user_peer.message(content, created_at=session_date)
                                )
                            elif role == "assistant":
                                honcho_messages.append(
                                    assistant_peer.message(
                                        content, created_at=session_date
                                    )
                                )

                    if honcho_messages:
                        for i in range(0, len(honcho_messages), 100):
                            batch = honcho_messages[i : i + 100]
                            await session.aio.add_messages(batch)

                    results["sessions_created"].append(
                        SessionResult(
                            name=session_id, message_count=len(honcho_messages)
                        )
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

            # Trigger dream for memory consolidation before questions
            print(
                f"[{workspace_id}] Deriver queue empty. Triggering dream consolidation..."
            )

            # Determine session_id for dream
            dream_session_id = (
                merged_session_id
                if self.merge_sessions and merged_session_id
                else (
                    haystack_session_ids[0]
                    if haystack_session_ids
                    else f"{workspace_id}_session"
                )
            )

            # Determine observer based on question type
            observer_peer = "assistant" if is_assistant_type else "user"

            # Single orchestrated dream handles all reasoning types
            dream_success = await self.trigger_dream_and_wait(
                honcho_client,
                workspace_id,
                observer=observer_peer,
                session_id=dream_session_id,
            )
            if not dream_success:
                print(f"[{workspace_id}] Warning: Dream did not complete")
            print(f"[{workspace_id}] Dream completed. Executing question...")

            # Execute the question
            output_lines.append(f"\nAsking question: {question_with_date}")

            try:
                if self.use_get_context:
                    # Use get_context instead of dialectic .chat endpoint
                    # Get the session to retrieve context from
                    if not self.merge_sessions or merged_session_id is None:
                        raise ValueError(
                            "Merged session ID is required when using get_context. Set --merge-sessions to True."
                        )
                    session = await honcho_client.aio.session(id=merged_session_id)

                    # Get context for the appropriate peer
                    peer_id = "assistant" if is_assistant_type else "user"
                    context = await session.aio.context(
                        summary=True,
                        peer_target=peer_id,
                        last_user_message=question,
                    )

                    # Format context using to_anthropic method
                    context_messages = context.to_anthropic(assistant="assistant")

                    # Add the question as the final user message
                    context_messages.append(
                        {"role": "user", "content": question_with_date}
                    )

                    # Call Anthropic API to generate response
                    response = await self.anthropic_client.messages.create(
                        model="claude-sonnet-4-5",
                        max_tokens=1024,
                        messages=cast(list[MessageParam], context_messages),
                    )

                    if not response.content:
                        raise ValueError("Anthropic returned empty response")

                    content_block = response.content[0]
                    actual_response = getattr(content_block, "text", "")
                else:
                    # Use the appropriate peer based on question type
                    if is_assistant_type:
                        # For assistant questions, use the assistant peer
                        actual_response = await assistant_peer.aio.chat(
                            question_with_date
                        )
                    else:
                        # For user questions, use the user peer (default behavior)
                        actual_response = await user_peer.aio.chat(question_with_date)

                # Clean up workspace if requested
                if self.cleanup_workspace:
                    try:
                        await honcho_client.aio.delete_workspace(workspace_id)
                        print(f"[{workspace_id}] cleaned up workspace")
                    except Exception as e:
                        print(f"Failed to delete workspace: {e}")

                actual_response = (
                    actual_response if isinstance(actual_response, str) else ""
                )

                input_tokens_used = self._get_latest_input_tokens_used()

                token_efficiency = None
                if input_tokens_used is not None and total_available_tokens > 0:
                    efficiency_ratio = input_tokens_used / total_available_tokens
                    token_efficiency = {
                        "total_available_tokens": total_available_tokens,
                        "tokens_used": input_tokens_used,
                        "efficiency_ratio": efficiency_ratio,
                    }
                    output_lines.append(
                        f"  token efficiency: {efficiency_ratio:.4f} ({input_tokens_used}/{total_available_tokens} tokens, {efficiency_ratio * 100:.2f}%)"
                    )

                judgment = await judge_response(
                    self.openai_client,
                    question_with_date,
                    expected_answer,
                    actual_response,
                    question_type,
                    question_id,
                )

                query_result: QueryResult = {
                    "question": question_with_date,
                    "expected_answer": expected_answer,
                    "actual_response": actual_response,
                    "judgment": judgment,
                    "token_efficiency": token_efficiency,
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
                    question=question_with_date,
                    expected_answer=expected_answer,
                    actual_response=f"ERROR: {e}",
                    judgment={
                        "passed": False,
                        "reasoning": f"Question execution failed: {e}",
                    },
                    token_efficiency=None,
                )
                results["query_executed"] = query_result
                results["passed"] = False

            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]

            output_lines.append(
                f"\nQuestion {question_id} completed. Status: {'PASS' if results['passed'] else 'FAIL'} (Duration: {format_duration(results['duration_seconds'])})"
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
        if self.pool_size > 1:
            print(
                f"distributing questions across {self.pool_size} Honcho instances (ports {self.base_api_port}-{self.base_api_port + self.pool_size - 1})"
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

            # Run questions in current batch concurrently, distributing via round-robin
            batch_results: list[TestResult] = await asyncio.gather(
                *[
                    self.execute_question(q, self.get_honcho_url_for_index(i + idx))
                    for idx, q in enumerate(batch)
                ]
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
        print(f"Total Test Time: {format_duration(total_test_time)}")

        efficiency_ratios: list[float] = []
        for result in results:
            query = result.get("query_executed")
            if query:
                token_eff = query.get("token_efficiency")
                if token_eff:
                    efficiency_ratios.append(token_eff["efficiency_ratio"])

        if efficiency_ratios:
            avg_efficiency = sum(efficiency_ratios) / len(efficiency_ratios)
            min_efficiency = min(efficiency_ratios)
            max_efficiency = max(efficiency_ratios)
            print("\nToken Efficiency:")
            print(
                f"  Average: {avg_efficiency:.4f} ({avg_efficiency * 100:.2f}% of available tokens used)"
            )
            print(f"  Min: {min_efficiency:.4f} ({min_efficiency * 100:.2f}%)")
            print(f"  Max: {max_efficiency:.4f} ({max_efficiency * 100:.2f}%)")

        print("\nDetailed Results:")
        print(
            f"{'Question ID':<15} {'Type':<20} {'Status':<8} {'Duration':<10} {'Workspace ID':<30}"
        )
        print(f"{'-' * 15} {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 30}")

        for result in results:
            question_id = result["question_id"]
            question_type = result["question_type"]
            status = "PASS" if result.get("passed", False) else "FAIL"
            duration = format_duration(result["duration_seconds"])
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
        type_stats = calculate_type_statistics(results)

        # Calculate timing statistics
        timing_stats = calculate_timing_statistics(results, total_elapsed_seconds)

        # Calculate token efficiency statistics
        efficiency_ratios: list[float] = []
        total_available_tokens_list: list[int] = []
        tokens_used_list: list[int] = []
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

        # Create the full summary
        summary = {
            "metadata": {
                "test_file": str(test_file),
                "execution_timestamp": datetime.now().isoformat(),
                "runner_version": "1.0.0",
                "base_api_port": self.base_api_port,
                "pool_size": self.pool_size,
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

        if output_file:
            write_json_summary(summary, output_file)


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
  %(prog)s --test-file test.json --pool-size 4                            # Use 4 Honcho instances
  %(prog)s --test-file test.json --base-api-port 8000 --pool-size 4       # Custom base port with pool
  %(prog)s --test-file test.json --test-count 50                          # Run only first 50 tests
  %(prog)s --test-file test.json --question-id "q123"                    # Run only question with ID "q123"
        """,
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="Path to longmemeval JSON file (required)",
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

    parser.add_argument(
        "--merge-sessions",
        action="store_true",
        help="Merge all sessions within a question into a single session (default: False)",
    )

    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after executing each question (default: False)",
    )

    parser.add_argument(
        "--use-get-context",
        action="store_true",
        help="Use get_context + judge LLM instead of dialectic .chat endpoint (default: False)",
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

    if args.pool_size <= 0:
        print(f"Error: Pool size must be positive, got {args.pool_size}")
        return 1

    if args.test_count is not None and args.test_count <= 0:
        print(f"Error: Test count must be positive, got {args.test_count}")
        return 1

    # Create test runner
    runner = LongMemEvalRunner(
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        anthropic_api_key=args.anthropic_api_key,
        timeout_seconds=args.timeout,
        merge_sessions=args.merge_sessions,
        cleanup_workspace=args.cleanup_workspace,
        use_get_context=args.use_get_context,
    )

    try:
        # Run all questions
        results, total_elapsed = await runner.run_all_questions(
            args.test_file, args.batch_size, args.test_count, args.question_id
        )
        runner.print_summary(results, total_elapsed_seconds=total_elapsed)

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
