"""
Honcho LoCoMo Benchmark Test Runner

A script that executes LoCoMo benchmark tests against a running Honcho instance.
This script:
1. Loads LoCoMo conversation data from JSON files
2. Creates a workspace for each conversation sample
3. Ingests conversation sessions as messages between two peers
4. Waits for the deriver queue to process everything
5. Triggers a dream for memory consolidation
6. Executes questions and judges responses using an LLM

## LoCoMo Overview

LoCoMo evaluates very long-term conversational memory across five question categories:
1. Single-hop - Direct factual recall from conversations
2. Multi-hop - Reasoning across multiple pieces of information
3. Temporal - Understanding time-based relationships and sequences
4. Commonsense/World knowledge - Applying broader contextual understanding
5. Adversarial - Questions that cannot be answered (filtered out by default)

Reference: https://github.com/snap-research/locomo
Paper: https://arxiv.org/abs/2402.17753

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```
NOTE: you may create a .env file in this directory to customize honcho config.

1. Run the test harness:
```
python -m tests.bench.harness
```

2. Run this file with the LoCoMo dataset:
```
python -m tests.bench.locomo --data-file tests/bench/locomo_data/locomo10.json
```

Optional arguments:
```
--anthropic-api-key: Anthropic API key for response judging (can be set in .env as LLM_ANTHROPIC_API_KEY)
--timeout: Timeout for deriver queue to empty in seconds (default: 10 minutes)
--base-api-port: Base port for Honcho API instances (default: 8000)
--pool-size: Number of Honcho instances in the pool (default: 1)
--batch-size: Number of conversations to run concurrently in each batch (default: 1)
--json-output: Path to write JSON summary results for analytics
--cleanup-workspace: Delete workspace after executing each conversation (default: False)
--use-get-context: Use get_context + judge LLM instead of dialectic .chat endpoint (default: False)
--sample-id: Run only the conversation with this sample_id (skips all others)
--test-count: Number of conversations to run (default: all)
--question-count: Number of questions per conversation to run (default: all)
```

## Other notes
- Judge is Claude Sonnet 4.5
- Evaluation uses F1 score following the LoCoMo paper methodology
"""

import argparse
import asyncio
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
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)

from src.config import settings
from src.utils.metrics_collector import MetricsCollector

from .locomo_common import (
    CATEGORY_NAMES,
    ConversationResult,
    QuestionResult,
    calculate_category_scores,
    calculate_tokens,
    extract_sessions,
    filter_questions,
    format_duration,
    generate_json_summary,
    judge_response,
    load_locomo_data,
    parse_locomo_date,
    print_summary,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")


class LoCoMoRunner:
    """
    Executes LoCoMo benchmark tests against a Honcho instance.
    """

    def __init__(
        self,
        base_api_port: int = 8000,
        pool_size: int = 1,
        anthropic_api_key: str | None = None,
        timeout_seconds: int | None = None,
        cleanup_workspace: bool = False,
        use_get_context: bool = False,
    ):
        """
        Initialize the LoCoMo test runner.

        Args:
            base_api_port: Base port for Honcho API instances (default: 8000)
            pool_size: Number of Honcho instances in the pool (default: 1)
            anthropic_api_key: Anthropic API key for judging responses
            timeout_seconds: Timeout for deriver queue in seconds
            cleanup_workspace: If True, delete workspace after executing conversation
            use_get_context: If True, use get_context + judge LLM instead of dialectic .chat endpoint
        """
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.anthropic_api_key: str | None = anthropic_api_key
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 600
        )
        self.cleanup_workspace: bool = cleanup_workspace
        self.use_get_context: bool = use_get_context

        # Initialize metrics collector
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.metrics_collector.start_collection(
            f"locomo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    def get_honcho_url_for_index(self, index: int) -> str:
        """Get the Honcho URL for a given index using round-robin distribution."""
        instance_id = index % self.pool_size
        port = self.base_api_port + instance_id
        return f"http://localhost:{port}"

    async def create_honcho_client(
        self, workspace_id: str, honcho_url: str
    ) -> AsyncHoncho:
        """Create a Honcho client for a specific workspace."""
        return AsyncHoncho(
            environment="local",
            workspace_id=workspace_id,
            base_url=honcho_url,
        )

    async def wait_for_deriver_queue_empty(
        self, honcho_client: AsyncHoncho, session_id: str | None = None
    ) -> bool:
        """Wait for the deriver queue to be empty."""
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.get_deriver_status(session_id=session_id)
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
        honcho_client: AsyncHoncho,
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

        url = f"{honcho_url}/v2/workspaces/{workspace_id}/trigger_dream"
        payload = {
            "observer": observer,
            "observed": observed,
            "dream_type": "consolidate",
            "session_id": session_id or f"{workspace_id}_session",
        }

        print(f"[{workspace_id}] Triggering dream at {url}")

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
        await asyncio.sleep(2)
        success = await self.wait_for_deriver_queue_empty(honcho_client)
        if success:
            print(f"[{workspace_id}] Dream queue empty")
        else:
            print(f"[{workspace_id}] Dream queue timeout")
        return success

    async def execute_conversation(
        self,
        conversation_data: dict[str, Any],
        honcho_url: str,
        question_count: int | None = None,
    ) -> ConversationResult:
        """
        Execute LoCoMo benchmark for a single conversation.

        Args:
            conversation_data: Dictionary containing conversation and QA data
            honcho_url: URL of the Honcho instance to use
            question_count: Optional limit on number of questions to run

        Returns:
            Conversation execution results
        """
        start_time = time.time()

        sample_id = conversation_data.get("sample_id", "unknown")
        conversation = conversation_data.get("conversation", {})
        qa_list = conversation_data.get("qa", [])

        speaker_a = conversation.get("speaker_a", "User")
        speaker_b = conversation.get("speaker_b", "Assistant")

        print(f"\n{'=' * 80}")
        print(f"Executing LoCoMo conversation {sample_id}")
        print(f"Speakers: {speaker_a} and {speaker_b}")
        print(f"{'=' * 80}")

        # Create workspace for this conversation
        workspace_id = f"locomo_{sample_id}"
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        result: ConversationResult = {
            "sample_id": sample_id,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "total_sessions": 0,
            "total_turns": 0,
            "total_tokens": 0,
            "question_results": [],
            "category_scores": {},
            "overall_score": 0.0,
            "overall_f1": 0.0,
            "error": None,
            "start_time": start_time,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        try:
            # Create peers - speaker_a as "user", speaker_b as "assistant"
            user_peer = await honcho_client.peer(id="user")
            assistant_peer = await honcho_client.peer(id="assistant")

            # Create session for this conversation
            session_id = f"{workspace_id}_session"
            session = await honcho_client.session(id=session_id)

            # Configure peer observation - observe the user peer (speaker_a)
            await session.add_peers(
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

            # Extract and ingest all sessions
            sessions = extract_sessions(conversation)
            result["total_sessions"] = len(sessions)

            print(f"[{workspace_id}] Ingesting {len(sessions)} sessions...")

            messages: list[MessageCreateParam] = []
            total_tokens = 0

            for date_str, session_messages in sessions:
                session_date = parse_locomo_date(date_str) if date_str else None

                for msg in session_messages:
                    speaker = msg.get("speaker", "")
                    text = msg.get("text", "")
                    result["total_turns"] += 1
                    total_tokens += calculate_tokens(text)

                    # Map speaker to peer
                    if speaker == speaker_a:
                        if session_date:
                            messages.append(
                                user_peer.message(text, created_at=session_date)
                            )
                        else:
                            messages.append(user_peer.message(text))
                    elif speaker == speaker_b:
                        if session_date:
                            messages.append(
                                assistant_peer.message(text, created_at=session_date)
                            )
                        else:
                            messages.append(assistant_peer.message(text))

            result["total_tokens"] = total_tokens

            # Add messages in batches of 100
            for i in range(0, len(messages), 100):
                batch = messages[i : i + 100]
                await session.add_messages(batch)

            print(
                f"[{workspace_id}] Ingested {len(messages)} messages (~{total_tokens:,} tokens). Waiting for deriver queue..."
            )

            # Wait for deriver queue to empty
            await asyncio.sleep(1)
            queue_empty = await self.wait_for_deriver_queue_empty(honcho_client)
            if not queue_empty:
                result["error"] = "Deriver queue timeout"
                result["end_time"] = time.time()
                result["duration_seconds"] = result["end_time"] - result["start_time"]
                print(
                    f"\n[{workspace_id}] ERROR: Deriver queue timeout after {self.timeout_seconds}s"
                )
                return result

            print(
                f"[{workspace_id}] Deriver queue empty. Triggering dream consolidation..."
            )

            # Trigger dream for memory consolidation
            dream_success = await self.trigger_dream_and_wait(
                honcho_client,
                workspace_id,
                observer="user",
                session_id=session_id,
            )

            if not dream_success:
                print(
                    f"[{workspace_id}] Warning: Dream did not complete, proceeding anyway"
                )
            else:
                print(f"[{workspace_id}] Dream completed. Executing questions...")

            # Filter questions
            filtered_qa = filter_questions(
                qa_list,
                exclude_adversarial=False,
                test_count=question_count,
            )

            print(f"[{workspace_id}] Executing {len(filtered_qa)} questions...")

            # Execute questions
            for q_idx, qa in enumerate(filtered_qa):
                question = qa.get("question", "")
                expected_answer = qa.get("answer", "")
                category = qa.get("category", 0)
                evidence = qa.get("evidence", [])
                category_name = CATEGORY_NAMES.get(category, f"category_{category}")

                print(f"  Q{q_idx + 1} [{category_name}]: {question[:80]}...")

                try:
                    if self.use_get_context:
                        # Use get_context + LLM
                        context = await session.get_context(
                            summary=True,
                            peer_target="user",
                            last_user_message=question,
                        )
                        context_messages = context.to_anthropic(assistant="assistant")
                        context_messages.append({"role": "user", "content": question})

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
                        # Use dialectic .chat endpoint
                        actual_response = await user_peer.chat(question)
                        actual_response = (
                            actual_response if isinstance(actual_response, str) else ""
                        )

                    # Judge the response
                    judgment = await judge_response(
                        self.anthropic_client,
                        question,
                        str(expected_answer),
                        actual_response,
                    )

                    passed = judgment.get("passed", False)
                    score = judgment.get("score", 0.0)

                    question_result: QuestionResult = {
                        "question_id": q_idx,
                        "question": question,
                        "expected_answer": str(expected_answer),
                        "actual_response": actual_response,
                        "category": category,
                        "category_name": category_name,
                        "evidence": evidence,
                        "judgment": judgment,
                        "passed": passed,
                    }

                    result["question_results"].append(question_result)

                    status = "PASS" if passed else "FAIL"
                    print(f"    Score: {score:.2f} [{status}]")
                    if not passed:
                        print(f"      Expected: {expected_answer}")
                        print(f"      Got: {actual_response[:200]}...")

                except Exception as e:
                    self.logger.error(f"Error executing question {q_idx}: {e}")
                    question_result = QuestionResult(
                        question_id=q_idx,
                        question=question,
                        expected_answer=str(expected_answer),
                        actual_response=f"ERROR: {e}",
                        category=category,
                        category_name=category_name,
                        evidence=evidence,
                        judgment={"passed": False, "score": 0.0, "reasoning": str(e)},
                        passed=False,
                    )
                    result["question_results"].append(question_result)

            # Calculate category scores
            result["category_scores"] = calculate_category_scores(
                result["question_results"]
            )

            # Calculate overall scores
            if result["question_results"]:
                scores = [
                    qr["judgment"].get("score", 0.0)
                    for qr in result["question_results"]
                ]
                f1s = [
                    qr["judgment"].get("f1", 0.0) for qr in result["question_results"]
                ]
                result["overall_score"] = sum(scores) / len(scores)
                result["overall_f1"] = sum(f1s) / len(f1s)

            # Cleanup workspace if requested
            if self.cleanup_workspace:
                try:
                    await honcho_client.delete_workspace(workspace_id)
                    print(f"[{workspace_id}] Cleaned up workspace")
                except Exception as e:
                    print(f"Failed to delete workspace: {e}")

            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - result["start_time"]

            print(
                f"\n[{workspace_id}] Completed in {format_duration(result['duration_seconds'])}"
            )
            print(f"Overall Score: {result['overall_score']:.3f}")
            print(f"Overall F1: {result['overall_f1']:.3f}")

        except Exception as e:
            self.logger.error(f"Error executing conversation {sample_id}: {e}")
            result["error"] = str(e)
            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - result["start_time"]

        return result

    async def run_conversations(
        self,
        data_file: Path,
        batch_size: int = 1,
        test_count: int | None = None,
        sample_id: str | None = None,
        question_count: int | None = None,
    ) -> tuple[list[ConversationResult], float]:
        """
        Run multiple conversations from the LoCoMo benchmark.

        Args:
            data_file: Path to the LoCoMo JSON file
            batch_size: Number of conversations to run concurrently in each batch
            test_count: Optional number of conversations to run
            sample_id: Optional sample_id to run only that conversation
            question_count: Optional limit on questions per conversation

        Returns:
            Tuple of (list of conversation results, total duration)
        """
        conversations = load_locomo_data(data_file)

        # Filter by sample_id if specified
        if sample_id is not None:
            conversations = [
                c for c in conversations if c.get("sample_id") == sample_id
            ]
            if not conversations:
                print(f"Error: No conversation found with sample_id '{sample_id}'")
                return [], 0.0
            print(f"Filtering to sample_id '{sample_id}'")

        # Limit by test_count
        if test_count is not None and test_count > 0:
            conversations = conversations[:test_count]
            print(f"Limiting to {len(conversations)} conversations")

        print(f"Running {len(conversations)} conversations from {data_file}")
        if self.pool_size > 1:
            print(
                f"Distributing conversations across {self.pool_size} Honcho instances"
            )

        overall_start = time.time()
        all_results: list[ConversationResult] = []

        for i in range(0, len(conversations), batch_size):
            batch = conversations[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(conversations) + batch_size - 1) // batch_size

            print(f"\n{'=' * 80}")
            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} conversations)"
            )
            print(f"{'=' * 80}")

            # Run conversations in current batch concurrently
            batch_results: list[ConversationResult] = await asyncio.gather(
                *[
                    self.execute_conversation(
                        conv,
                        self.get_honcho_url_for_index(i + idx),
                        question_count=question_count,
                    )
                    for idx, conv in enumerate(batch)
                ]
            )

            all_results.extend(batch_results)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        # Finalize metrics collection
        self.metrics_collector.finalize_collection()

        return all_results, overall_duration


async def main() -> int:
    """Main entry point for the LoCoMo test runner."""
    parser = argparse.ArgumentParser(
        description="Run LoCoMo benchmark tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data-file tests/bench/locomo_data/locomo10.json
  %(prog)s --data-file locomo10.json --pool-size 4
  %(prog)s --data-file locomo10.json --sample-id "sample_0"
  %(prog)s --data-file locomo10.json --test-count 5 --question-count 20
        """,
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to LoCoMo JSON file (required)",
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
        default=1,
        help="Number of conversations to run concurrently in each batch (default: 1)",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to write JSON summary results for analytics (optional)",
    )

    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after executing each conversation (default: False)",
    )

    parser.add_argument(
        "--use-get-context",
        action="store_true",
        help="Use get_context + judge LLM instead of dialectic .chat endpoint (default: False)",
    )

    parser.add_argument(
        "--sample-id",
        type=str,
        help="Run only the conversation with this sample_id (skips all others)",
    )

    parser.add_argument(
        "--test-count",
        type=int,
        help="Number of conversations to run from the data file (default: all)",
    )

    parser.add_argument(
        "--question-count",
        type=int,
        help="Number of questions per conversation to run (default: all)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.data_file.exists():
        print(f"Error: Data file {args.data_file} does not exist")
        return 1

    if args.batch_size <= 0:
        print(f"Error: Batch size must be positive, got {args.batch_size}")
        return 1

    if args.pool_size <= 0:
        print(f"Error: Pool size must be positive, got {args.pool_size}")
        return 1

    # Create test runner
    runner = LoCoMoRunner(
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        anthropic_api_key=args.anthropic_api_key,
        timeout_seconds=args.timeout,
        cleanup_workspace=args.cleanup_workspace,
        use_get_context=args.use_get_context,
    )

    try:
        # Run conversations
        results, total_elapsed = await runner.run_conversations(
            args.data_file,
            args.batch_size,
            args.test_count,
            args.sample_id,
            args.question_count,
        )

        print_summary(results, total_elapsed)

        # Print metrics summary
        runner.metrics_collector.print_summary()

        # Generate JSON output
        if args.json_output:
            output_file = args.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/locomo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            total_elapsed,
            output_file,
            metadata_extra={
                "data_file": str(args.data_file),
                "base_api_port": runner.base_api_port,
                "pool_size": runner.pool_size,
                "timeout_seconds": runner.timeout_seconds,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
                "summary_settings": settings.SUMMARY.model_dump(),
            },
        )

        # Export metrics to JSON file
        metrics_output = Path(
            f"tests/bench/perf_metrics/locomo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        runner.metrics_collector.export_to_json(metrics_output)
        runner.metrics_collector.cleanup_collection()

        # Return exit code based on results
        avg_score = (
            sum(r["overall_score"] for r in results) / len(results) if results else 0
        )
        return 0 if avg_score >= 0.5 else 1

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
