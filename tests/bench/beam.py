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
    generate_json_summary,
    judge_event_ordering,
    judge_nugget_based,
    list_conversations,
    load_conversation,
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


class BEAMRunner(BaseRunner[ConversationResult]):
    """
    Executes BEAM benchmark tests against a Honcho instance.
    """

    def __init__(
        self,
        config: RunnerConfig,
        data_dir: Path,
        context_length: str,
        conversation_ids: list[str] | None = None,
    ):
        """
        Initialize the BEAM test runner.

        Args:
            config: Common runner configuration
            data_dir: Path to the BEAM data directory
            context_length: Context length (100K, 500K, 1M, 10M)
            conversation_ids: Optional list of specific conversation IDs to run
        """
        self.data_dir: Path = data_dir
        self.context_length: str = context_length
        self.conversation_ids: list[str] | None = conversation_ids

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
            "BEAM_JUDGE_MODEL", "anthropic/claude-sonnet-4.5"
        )

    def get_metrics_prefix(self) -> str:
        return "beam"

    def load_items(self) -> list[Any]:
        """Load conversation IDs to process."""
        if self.conversation_ids:
            return self.conversation_ids
        return list_conversations(self.data_dir, self.context_length)

    def get_workspace_id(self, item: Any) -> str:
        """Return workspace ID for a conversation."""
        return f"beam_{self.context_length}_{item}"

    def get_session_id(self, item: Any, workspace_id: str) -> str:
        """Return session ID for a conversation."""
        return f"{workspace_id}_session"

    async def setup_peers(self, ctx: ItemContext, item: Any) -> None:
        """Create user and assistant peers."""
        ctx.peers["user"] = await ctx.honcho_client.aio.peer(id="user")
        ctx.peers["assistant"] = await ctx.honcho_client.aio.peer(id="assistant")

    async def setup_session(self, ctx: ItemContext, item: Any) -> None:
        """Create and configure session - observe the user peer."""
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
        """Ingest conversation turns into the session."""
        conversation_id = item
        user_peer = ctx.peers["user"]
        assistant_peer = ctx.peers["assistant"]

        # Load conversation data
        conv_data = load_conversation(
            self.data_dir, self.context_length, conversation_id
        )
        chat_data = conv_data["chat"]

        # Store questions data for later use
        ctx.peers["_questions_data"] = conv_data["questions"]

        messages: list[MessageCreateParams] = []

        # Handle different data structures for 10M vs other sizes
        for batch in chat_data:
            if any(key.startswith("plan-") for key in batch):
                # 10M structure: { "plan-1": [...], "plan-2": [...], ... }
                for plan_name, plan_batches in batch.items():
                    if not plan_name.startswith("plan-"):
                        continue
                    for plan_batch in plan_batches:
                        for turn_group in plan_batch.get("turns", []):
                            for turn in turn_group:
                                messages.extend(
                                    self._process_turn(turn, user_peer, assistant_peer)
                                )
            else:
                # Standard structure for 100K, 500K, 1M
                for turn_group in batch.get("turns", []):
                    for turn in turn_group:
                        messages.extend(
                            self._process_turn(turn, user_peer, assistant_peer)
                        )

        # Add messages in batches of 100
        for i in range(0, len(messages), 100):
            batch = messages[i : i + 100]
            await ctx.session.aio.add_messages(batch)

        return len(messages)

    def _process_turn(
        self, turn: dict[str, Any], user_peer: Any, assistant_peer: Any
    ) -> list[MessageCreateParams]:
        """Process a single turn, handling long message splitting."""
        role = turn["role"]
        content = turn["content"]
        messages: list[MessageCreateParams] = []

        if len(content) > 25000:
            chunks = [content[i : i + 25000] for i in range(0, len(content), 25000)]
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

        return messages

    def get_dream_observers(self, item: Any) -> list[str]:
        """Return the observer - always user for BEAM."""
        return ["user"]

    async def _process_single_question(
        self,
        ctx: ItemContext,
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
            if self.config.use_get_context or ability == "instruction_following":
                context = await ctx.session.aio.context(
                    summary=True,
                    peer_target="user",
                    search_query=question,
                )
                context_messages = context.to_openai(assistant="assistant")
                context_messages.append({"role": "user", "content": question})

                # For instruction_following, add a system prompt
                system_prompt = None
                if ability == "instruction_following":
                    system_prompt = """You are a helpful assistant with memory of the user's preferences and instructions from previous conversations.

The context provided includes observations about the user, which may contain their stated preferences, instructions, or requirements for how you should respond.

IMPORTANT: You MUST follow any instructions or preferences the user has previously stated. For example:
- If the user said "always include X when discussing Y", you must include X when discussing Y
- If the user said "I prefer responses that are Z", format your response accordingly
- If the user gave any standing instructions, follow them

Review the context carefully for any such instructions before responding."""

                messages: list[dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(cast(list[dict[str, Any]], context_messages))

                response = await self.openrouter_client.chat.completions.create(
                    model=self.judge_model,
                    max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
                    messages=cast(Any, messages),
                )

                if not response.choices or not response.choices[0].message:
                    actual_response = ""
                else:
                    actual_response = response.choices[0].message.content or ""
            else:
                actual_response = await ctx.peers["user"].aio.chat(
                    question,
                    reasoning_level=self.config.reasoning_level,
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

    async def execute_questions(
        self, ctx: ItemContext, item: Any
    ) -> ConversationResult:
        """Execute all questions for the conversation."""
        conversation_id = item
        workspace_id = ctx.workspace_id

        result: ConversationResult = {
            "conversation_id": conversation_id,
            "context_length": self.context_length,
            "workspace_id": workspace_id,
            "total_turns": 0,
            "total_messages": 0,
            "question_results": [],
            "ability_scores": {},
            "overall_score": 0.0,
            "error": None,
            "start_time": 0.0,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        questions_data = ctx.peers.get("_questions_data", {})

        # Execute questions for each memory ability
        question_tasks: list[Any] = []
        semaphore = asyncio.Semaphore(5)

        for ability, questions in questions_data.items():
            print(f"\n[{workspace_id}] Queuing {ability} ({len(questions)} questions)")

            for q_idx, q_data in enumerate(questions):
                question_tasks.append(
                    self._process_single_question(
                        ctx, ability, q_idx, q_data, semaphore
                    )
                )

        results = await asyncio.gather(*question_tasks)
        result["question_results"] = list(results)

        # Calculate ability scores
        result["ability_scores"] = calculate_ability_scores(result["question_results"])

        # Calculate overall score
        if result["ability_scores"]:
            result["overall_score"] = sum(result["ability_scores"].values()) / len(
                result["ability_scores"]
            )

        print(f"\nOverall Score: {result['overall_score']:.3f}")

        return result

    def print_summary(
        self, results: list[ConversationResult], total_duration: float
    ) -> None:
        """Print summary using the common function."""
        print_summary(results, total_duration)

    def generate_output(
        self, results: list[ConversationResult], total_duration: float
    ) -> None:
        """Generate JSON output file."""
        if self.config.json_output:
            output_file = self.config.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/beam_{self.context_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            self.context_length,
            total_duration,
            output_file,
            metadata_extra={
                "base_api_port": self.config.base_api_port,
                "pool_size": self.config.pool_size,
                "timeout_seconds": self.config.timeout_seconds,
                "reasoning_level": self.config.reasoning_level,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
            },
        )


def main() -> int:
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

    # Parse conversation IDs
    conversation_ids = None
    if args.conversation_ids:
        conversation_ids = args.conversation_ids.split(",")

    # Create config and runner
    config = RunnerConfig.from_args(args, default_timeout=600)

    runner = BEAMRunner(
        config=config,
        data_dir=data_dir,
        context_length=args.context_length,
        conversation_ids=conversation_ids,
    )

    return runner.run_and_summarize()


if __name__ == "__main__":
    exit(main())
