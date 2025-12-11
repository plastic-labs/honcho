"""
Honcho BEAM Benchmark Test Runner

A script that executes BEAM (Beyond a Million Tokens) benchmark tests against a running Honcho instance.
This script:
1. Loads BEAM conversations and probing questions from the data directory
2. Creates a workspace for each conversation
3. Ingests conversation turns as messages between user/assistant peers
4. Waits for the deriver queue to process everything
5. Executes probing questions across 10 memory ability categories
6. Evaluates responses using nugget-based LLM judging

## BEAM Overview

BEAM evaluates long-term memory capabilities across ten distinct memory abilities:
1. Abstention - Determines if models avoid answering without evidence
2. Contradiction Resolution - Detects inconsistencies across distant dialogue turns
3. Event Ordering - Assesses sequence recognition of evolving information
4. Information Extraction - Measures factual recall from lengthy histories
5. Instruction Following - Tests sustained adherence to user constraints
6. Knowledge Update - Evaluates fact revision when new information emerges
7. Multi-Session Reasoning - Probes inference integrating evidence across non-adjacent segments
8. Preference Following - Captures personalized, adaptive responses
9. Summarization - Tests content compression and abstraction
10. Temporal Reasoning - Examines explicit and implicit time-relation understanding

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

2. Run this file with the 100K dataset:
```
python -m tests.bench.beam --context-length 100K
```

Optional arguments:
```
--context-length: Context length subset to test (100K, 500K, 1M, 10M) (default: 100K)
--conversation-ids: Comma-separated list of conversation IDs to test (default: all in context length)
--timeout: Timeout for deriver queue to empty in seconds (default: 10 minutes (600s))
--base-api-port: Base port for Honcho API instances (default: 8000)
--pool-size: Number of Honcho instances in the pool (default: 1)
--batch-size: Number of conversations to run concurrently in each batch (default: 1)
--json-output: Path to write JSON summary results for analytics
--cleanup-workspace: Delete workspace after executing each conversation (default: True)
--use-get-context: Use get_context + judge LLM instead of dialectic .chat endpoint (default: False)
```

## Other notes
- Judge uses OpenRouter (configured via LLM_OPENAI_COMPATIBLE_API_KEY and LLM_OPENAI_COMPATIBLE_BASE_URL in tests/bench/.env)
- Default judge model is anthropic/claude-sonnet-4.5 (can be overridden with BEAM_JUDGE_MODEL env var)
- Evaluation follows the paper's nugget-based methodology with 0/0.5/1 scoring
- Event ordering uses Kendall tau-b coefficient
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)
from openai import AsyncOpenAI

from src.config import settings
from src.utils.metrics_collector import MetricsCollector

from .beam_common import (
    ConversationResult,
    QuestionResult,
    calculate_ability_scores,
    format_duration,
    generate_json_summary,
    judge_event_ordering,
    judge_nugget_based,
    list_conversations,
    load_conversation,
    print_summary,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")


class BEAMRunner:
    """
    Executes BEAM benchmark tests against a Honcho instance.
    """

    def __init__(
        self,
        data_dir: Path,
        base_api_port: int = 8000,
        pool_size: int = 1,
        timeout_seconds: int | None = None,
        cleanup_workspace: bool = True,
        use_get_context: bool = False,
        use_orchestrated_dream: bool = True,
    ):
        """
        Initialize the BEAM test runner.

        Args:
            data_dir: Path to the BEAM data directory
            base_api_port: Base port for Honcho API instances (default: 8000)
            pool_size: Number of Honcho instances in the pool (default: 1)
            timeout_seconds: Timeout for deriver queue in seconds
            cleanup_workspace: If True, delete workspace after executing conversation
            use_get_context: If True, use get_context + judge LLM instead of dialectic .chat endpoint
            use_orchestrated_dream: If True, use new orchestrated specialist architecture
        """
        self.data_dir: Path = data_dir
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 600
        )
        self.cleanup_workspace: bool = cleanup_workspace
        self.use_get_context: bool = use_get_context
        self.use_orchestrated_dream: bool = use_orchestrated_dream

        # Initialize metrics collector
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.metrics_collector.start_collection(
            f"beam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Suppress HTTP request logs from the Honcho SDK
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

        # Initialize OpenRouter client for judging
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

        # Model to use for judging (OpenRouter format)
        self.judge_model: str = os.getenv(
            "BEAM_JUDGE_MODEL", "anthropic/claude-sonnet-4.5"
        )

    def get_honcho_url_for_index(self, conversation_index: int) -> str:
        """
        Get the Honcho URL for a given conversation index using round-robin distribution.

        Args:
            conversation_index: Index of the conversation

        Returns:
            URL of the Honcho instance to use for this conversation
        """
        instance_id = conversation_index % self.pool_size
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
        """Wait for the deriver queue to be empty."""
        start_time = time.time()
        while True:
            try:
                status = await honcho_client.get_deriver_status(session=session_id)
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
        reasoning_focus: str | None = None,
    ) -> bool:
        """
        Trigger a dream task and wait for it to complete.

        Args:
            honcho_client: Honcho client instance
            workspace_id: Workspace identifier
            observer: Observer peer name
            observed: Observed peer name (defaults to observer)
            session_id: Session ID to scope the dream to
            reasoning_focus: Optional focus mode ('deduction', 'induction', 'consolidation')
                           Ignored if use_orchestrated_dream is True

        Returns:
            True if dream completed successfully, False on timeout
        """
        import httpx

        observed = observed or observer
        honcho_url = self.get_honcho_url_for_index(0)

        url = f"{honcho_url}/v2/workspaces/{workspace_id}/trigger_dream"
        payload: dict[str, Any] = {
            "observer": observer,
            "observed": observed,
            "dream_type": "consolidate",
            "session_id": session_id or f"{workspace_id}_session",
        }

        # For orchestrated dreams, don't use reasoning_focus (it handles all aspects)
        # For legacy dreams, pass the focus
        if not self.use_orchestrated_dream and reasoning_focus:
            payload["reasoning_focus"] = reasoning_focus

        mode_str = "orchestrated" if self.use_orchestrated_dream else f"focus: {reasoning_focus}" if reasoning_focus else "default"
        print(f"[{workspace_id}] Triggering dream ({mode_str}) at {url}")

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
            f"[{workspace_id}] Dream triggered successfully for {observer}/{observed} ({mode_str})"
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

    async def _process_single_question(
        self,
        session: Any,
        user_peer: Any,
        ability: str,
        q_idx: int,
        q_data: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> QuestionResult:
        """Process a single BEAM question."""
        async with semaphore:
            question = q_data["question"]
            rubric = q_data.get("rubric", [])
            answer = (
                q_data.get("answer")
                or q_data.get("ideal_response")
                or q_data.get("ideal_answer")
            )

            print(f"  [{ability}] Q{q_idx + 1}: {question[:100]}...")

            # Execute question using dialectic
            # For instruction_following, always use get_context + OpenRouter API
            # so the LLM can follow user-specified instructions from Honcho context
            if self.use_get_context or ability == "instruction_following":
                context = await session.get_context(
                    summary=True,
                    peer_target="user",
                    last_user_message=question,
                )
                context_messages = context.to_openai(assistant="assistant")
                context_messages.append({"role": "user", "content": question})

                # For instruction_following, add a system prompt that tells the LLM
                # to follow any stored user preferences/instructions in the context
                system_prompt = None
                if ability == "instruction_following":
                    system_prompt = """You are a helpful assistant with memory of the user's preferences and instructions from previous conversations.

The context provided includes observations about the user, which may contain their stated preferences, instructions, or requirements for how you should respond.

IMPORTANT: You MUST follow any instructions or preferences the user has previously stated. For example:
- If the user said "always include X when discussing Y", you must include X when discussing Y
- If the user said "I prefer responses that are Z", format your response accordingly
- If the user gave any standing instructions, follow them

Review the context carefully for any such instructions before responding."""

                # Prepare messages: OpenAI format uses system role in messages array
                messages: list[dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(cast(list[dict[str, Any]], context_messages))

                response = await self.openrouter_client.chat.completions.create(
                    model=self.judge_model,
                    max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
                    messages=cast(Any, messages),  # type: ignore[arg-type]
                )

                if not response.choices or not response.choices[0].message:
                    actual_response = ""
                else:
                    actual_response = response.choices[0].message.content or ""
            else:
                actual_response = await user_peer.chat(question)
                actual_response = (
                    actual_response if isinstance(actual_response, str) else ""
                )

            # Judge response based on memory ability
            if ability == "event_ordering":
                judgment = await judge_event_ordering(
                    self.openrouter_client,
                    self.judge_model,
                    question,
                    rubric,
                    actual_response,
                )
                nugget_scores = None
            else:
                judgment = await judge_nugget_based(
                    self.openrouter_client,
                    self.judge_model,
                    question,
                    rubric,
                    actual_response,
                )
                nugget_scores = judgment.get("nugget_scores")

            score = judgment.get("overall_score", 0.0)
            reasoning = judgment.get("overall_reasoning", "")

            question_result: QuestionResult = {
                "question": question,
                "answer": answer,
                "actual_response": actual_response,
                "memory_ability": ability,
                "rubric": rubric,
                "nugget_scores": nugget_scores,
                "score": score,
                "passed": score >= 0.5,
                "reasoning": reasoning,
            }

            status = "PASS" if score >= 0.5 else "FAIL"
            print(f"    [{ability}] Q{q_idx + 1} Score: {score:.2f} [{status}]")
            if score < 0.5 and reasoning:
                print(f"      Reasoning: {reasoning}")
                if rubric:
                    print("      Rubric:")
                    for i, rubric_item in enumerate(rubric, 1):
                        print(f"        {i}. {rubric_item}")
                if answer:
                    print(f"      Ideal Response: {answer}")
                print(f"      Our Response: {actual_response}")

            return question_result

    async def execute_conversation(
        self, context_length: str, conversation_id: str, honcho_url: str
    ) -> ConversationResult:
        """
        Execute BEAM benchmark for a single conversation.

        Args:
            context_length: Context length (100K, 500K, 1M, 10M)
            conversation_id: Conversation ID
            honcho_url: URL of the Honcho instance to use

        Returns:
            Conversation execution results
        """
        start_time = time.time()

        print(f"\n{'=' * 80}")
        print(
            f"Executing BEAM conversation {conversation_id} ({context_length} context)"
        )
        print(f"{'=' * 80}")

        # Create workspace for this conversation
        workspace_id = f"beam_{context_length}_{conversation_id}"
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        result: ConversationResult = {
            "conversation_id": conversation_id,
            "context_length": context_length,
            "workspace_id": workspace_id,
            "total_turns": 0,
            "total_messages": 0,
            "question_results": [],
            "ability_scores": {},
            "overall_score": 0.0,
            "error": None,
            "start_time": start_time,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        try:
            # Load conversation data
            conv_data = load_conversation(
                self.data_dir, context_length, conversation_id
            )
            chat_data = conv_data["chat"]
            questions_data = conv_data["questions"]

            # Create peers
            user_peer = await honcho_client.peer(id="user")
            assistant_peer = await honcho_client.peer(id="assistant")

            # Create session for this conversation
            session_id = f"{workspace_id}_session"
            session = await honcho_client.session(id=session_id)

            # Configure peer observation - observe the user peer
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

            # Ingest conversation turns
            print(f"[{workspace_id}] Ingesting conversation turns...")
            messages: list[MessageCreateParam] = []

            # Handle different data structures for 10M vs other sizes
            for batch in chat_data:
                # Check if this is a 10M conversation with plan-based structure
                if any(key.startswith("plan-") for key in batch):
                    # 10M structure: { "plan-1": [...], "plan-2": [...], ... }
                    for plan_name, plan_batches in batch.items():
                        if not plan_name.startswith("plan-"):
                            continue
                        for plan_batch in plan_batches:
                            for turn_group in plan_batch.get("turns", []):
                                for turn in turn_group:
                                    role = turn["role"]
                                    content = turn["content"]
                                    result["total_turns"] += 1

                                    # Split message if it exceeds 25000 characters
                                    if len(content) > 25000:
                                        chunks = [
                                            content[i : i + 25000]
                                            for i in range(0, len(content), 25000)
                                        ]
                                        for chunk in chunks:
                                            if role == "user":
                                                messages.append(
                                                    user_peer.message(chunk)
                                                )
                                            elif role == "assistant":
                                                messages.append(
                                                    assistant_peer.message(chunk)
                                                )
                                    else:
                                        if role == "user":
                                            messages.append(user_peer.message(content))
                                        elif role == "assistant":
                                            messages.append(
                                                assistant_peer.message(content)
                                            )
                else:
                    # Standard structure for 100K, 500K, 1M
                    for turn_group in batch.get("turns", []):
                        for turn in turn_group:
                            role = turn["role"]
                            content = turn["content"]
                            result["total_turns"] += 1

                            # Split message if it exceeds 25000 characters
                            if len(content) > 25000:
                                chunks = [
                                    content[i : i + 25000]
                                    for i in range(0, len(content), 25000)
                                ]
                                for chunk in chunks:
                                    if role == "user":
                                        messages.append(user_peer.message(chunk))
                                    elif role == "assistant":
                                        messages.append(assistant_peer.message(chunk))
                            else:
                                if role == "user":
                                    messages.append(user_peer.message(content))
                                elif role == "assistant":
                                    messages.append(assistant_peer.message(content))

            result["total_messages"] = len(messages)

            # Add messages in batches of 100
            for i in range(0, len(messages), 100):
                batch = messages[i : i + 100]
                await session.add_messages(batch)

            print(
                f"[{workspace_id}] Ingested {result['total_messages']} messages. Waiting for deriver queue..."
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
                print(
                    f"[{workspace_id}] Failed to complete in {format_duration(result['duration_seconds'])}"
                )
                return result

            print(
                f"[{workspace_id}] Deriver queue empty. Triggering dream..."
            )

            if self.use_orchestrated_dream:
                # Single orchestrated dream handles all reasoning types
                dream_success = await self.trigger_dream_and_wait(
                    honcho_client,
                    workspace_id,
                    observer="user",
                    session_id=session_id,
                )
                if not dream_success:
                    print(f"[{workspace_id}] Warning: Orchestrated dream did not complete")
                print(f"[{workspace_id}] Orchestrated dream completed. Executing questions...")
            else:
                # Legacy: multiple focused dream passes
                dream_focuses: list[str | None] = ["deduction", "induction"]
                for focus in dream_focuses:
                    dream_success = await self.trigger_dream_and_wait(
                        honcho_client,
                        workspace_id,
                        observer="user",
                        session_id=session_id,
                        reasoning_focus=focus,
                    )
                    if not dream_success:
                        print(f"[{workspace_id}] Warning: Dream ({focus}) did not complete")
                print(f"[{workspace_id}] All dream passes completed. Executing questions...")

            # Execute questions for each memory ability
            question_tasks: list[Any] = []
            semaphore = asyncio.Semaphore(5)

            for ability, questions in questions_data.items():
                print(
                    f"\n[{workspace_id}] Queuing {ability} ({len(questions)} questions)"
                )

                for q_idx, q_data in enumerate(questions):
                    question_tasks.append(
                        self._process_single_question(
                            session,
                            user_peer,
                            ability,
                            q_idx,
                            q_data,
                            semaphore,
                        )
                    )

            results = await asyncio.gather(*question_tasks)
            result["question_results"] = list(results)

            # Calculate ability scores
            result["ability_scores"] = calculate_ability_scores(
                result["question_results"]
            )

            # Calculate overall score
            if result["ability_scores"]:
                result["overall_score"] = sum(result["ability_scores"].values()) / len(
                    result["ability_scores"]
                )

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

        except Exception as e:
            self.logger.error(f"Error executing conversation {conversation_id}: {e}")
            result["error"] = str(e)
            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - result["start_time"]

        return result

    async def run_conversations(
        self,
        context_length: str,
        conversation_ids: list[str],
        batch_size: int = 1,
    ) -> tuple[list[ConversationResult], float]:
        """
        Run multiple conversations from the BEAM benchmark.

        Args:
            context_length: Context length (100K, 500K, 1M, 10M)
            conversation_ids: List of conversation IDs to run
            batch_size: Number of conversations to run concurrently in each batch

        Returns:
            Tuple of (list of conversation results, total duration)
        """
        print(
            f"Running {len(conversation_ids)} conversations from {context_length} context length"
        )
        if self.pool_size > 1:
            print(
                f"Distributing conversations across {self.pool_size} Honcho instances"
            )

        overall_start = time.time()
        all_results: list[ConversationResult] = []

        for i in range(0, len(conversation_ids), batch_size):
            batch = conversation_ids[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(conversation_ids) + batch_size - 1) // batch_size

            print(f"\n{'=' * 80}")
            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} conversations)"
            )
            print(f"{'=' * 80}")

            # Run conversations in current batch concurrently
            batch_results: list[ConversationResult] = await asyncio.gather(
                *[
                    self.execute_conversation(
                        context_length, conv_id, self.get_honcho_url_for_index(i + idx)
                    )
                    for idx, conv_id in enumerate(batch)
                ]
            )

            all_results.extend(batch_results)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        # Finalize metrics collection
        self.metrics_collector.finalize_collection()

        return all_results, overall_duration


async def main() -> int:
    """Main entry point for the BEAM test runner."""
    parser = argparse.ArgumentParser(
        description="Run BEAM benchmark tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--context-length",
        type=str,
        default="100K",
        choices=["1K", "100K", "500K", "1M", "10M"],
        help="Context length subset to test (default: 100K)",
    )

    parser.add_argument(
        "--conversation-ids",
        type=str,
        help="Comma-separated list of conversation IDs to test (default: all)",
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
        help="Timeout for deriver queue to empty in seconds (default: 10 minutes (600s))",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
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
        help="Delete workspace after executing each conversation (default: True)",
    )

    parser.add_argument(
        "--use-get-context",
        action="store_true",
        help="Use get_context + judge LLM instead of dialectic .chat endpoint (default: False)",
    )

    parser.add_argument(
        "--legacy-dream",
        action="store_true",
        help="Use legacy multi-pass dream system instead of orchestrated specialists (default: False)",
    )

    args = parser.parse_args()

    # Setup data directory
    data_dir = Path(__file__).parent / "beam_data"
    if not data_dir.exists():
        print(f"Error: BEAM data directory not found at {data_dir}")
        return 1

    # Create runner
    runner = BEAMRunner(
        data_dir=data_dir,
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        timeout_seconds=args.timeout,
        cleanup_workspace=args.cleanup_workspace,
        use_get_context=args.use_get_context,
        use_orchestrated_dream=not args.legacy_dream,
    )

    try:
        # Determine which conversations to run
        if args.conversation_ids:
            conversation_ids = args.conversation_ids.split(",")
        else:
            conversation_ids = list_conversations(data_dir, args.context_length)

        # Run conversations
        results, total_elapsed = await runner.run_conversations(
            args.context_length, conversation_ids, args.batch_size
        )

        print_summary(results, total_elapsed)

        # Generate JSON output
        if args.json_output:
            output_file = args.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/beam_{args.context_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            args.context_length,
            total_elapsed,
            output_file,
            metadata_extra={
                "base_api_port": runner.base_api_port,
                "pool_size": runner.pool_size,
                "timeout_seconds": runner.timeout_seconds,
                "use_orchestrated_dream": runner.use_orchestrated_dream,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
            },
        )

        # Export metrics
        metrics_output = Path(
            f"tests/bench/perf_metrics/beam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        runner.metrics_collector.export_to_json(metrics_output)
        runner.metrics_collector.cleanup_collection()

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
