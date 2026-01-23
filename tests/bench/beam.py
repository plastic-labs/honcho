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
from .runner_common import (
    ReasoningLevel,
    RunnerMixin,
    add_common_arguments,
    create_openai_client,
    export_metrics,
    validate_common_arguments,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")


class BEAMRunner(RunnerMixin):
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
        redis_url: str = "redis://localhost:6379/0",
        reasoning_level: ReasoningLevel | None = None,
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
            redis_url: Redis URL for flush mode signaling (default: redis://localhost:6379/0)
            reasoning_level: Reasoning level for dialectic chat (default: None)
        """
        self.data_dir: Path = data_dir
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 600
        )
        self.cleanup_workspace: bool = cleanup_workspace
        self.use_get_context: bool = use_get_context
        self.redis_url: str = redis_url
        self.reasoning_level: ReasoningLevel | None = reasoning_level

        # Initialize common components (metrics, logging)
        self._init_common("beam")

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
            "BEAM_JUDGE_MODEL", "anthropic/claude-sonnet-4.5"
        )

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
                context = await session.aio.context(
                    summary=True,
                    peer_target="user",
                    search_query=question,
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
                actual_response = await user_peer.aio.chat(
                    question,
                    reasoning_level=self.reasoning_level,
                )
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
        honcho_client = self.create_honcho_client(workspace_id, honcho_url)

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
            user_peer = await honcho_client.aio.peer(id="user")
            assistant_peer = await honcho_client.aio.peer(id="assistant")

            # Create session for this conversation
            session_id = f"{workspace_id}_session"
            session = await honcho_client.aio.session(id=session_id)

            # Configure peer observation - observe the user peer
            await session.aio.add_peers(
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
            messages: list[MessageCreateParams] = []

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
                await session.aio.add_messages(batch)

            print(
                f"[{workspace_id}] Ingested {result['total_messages']} messages. Waiting for deriver queue..."
            )

            # Wait for deriver queue to empty
            await asyncio.sleep(1)
            await self.flush_deriver_queue()
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

            print(f"[{workspace_id}] Deriver queue empty. Triggering dream...")

            # Single orchestrated dream handles all reasoning types
            dream_success = await self.trigger_dream_and_wait(
                honcho_client,
                workspace_id,
                observer="user",
                session_id=session_id,
            )
            if not dream_success:
                print(f"[{workspace_id}] Warning: Dream did not complete")
            print(f"[{workspace_id}] Dream completed. Executing questions...")

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
                    await honcho_client.aio.delete_workspace(workspace_id)
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
        epilog="""
Examples:
  %(prog)s --context-length 100K
  %(prog)s --context-length 500K --pool-size 4
  %(prog)s --context-length 100K --conversation-ids conv1,conv2
  %(prog)s --context-length 100K --reasoning-level high
        """,
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

    # Add common arguments shared across all runners
    add_common_arguments(parser)

    args = parser.parse_args()

    # Validate common arguments
    error = validate_common_arguments(args)
    if error:
        print(error)
        return 1

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
        redis_url=args.redis_url,
        reasoning_level=args.reasoning_level,
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
                "reasoning_level": runner.reasoning_level,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
            },
        )

        # Export metrics
        export_metrics(runner.metrics_collector, "beam")

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
