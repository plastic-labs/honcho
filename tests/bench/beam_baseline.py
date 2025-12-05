"""
BEAM Baseline Test Runner (Direct Claude Context)

A script that executes BEAM benchmark tests directly against Claude Sonnet 4.5
by feeding the entire conversation history into the context window.

This serves as a baseline comparison against Honcho's memory framework.

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

1. Run this file with the 100K dataset:
```
python -m tests.bench.beam_baseline --context-length 100K
```

Optional arguments:
```
--context-length: Context length subset to test (100K, 500K, 1M, 10M) (default: 100K)
--conversation-ids: Comma-separated list of conversation IDs to test (default: all in context length)
--batch-size: Number of conversations to run concurrently in each batch (default: 1)
--json-output: Path to write JSON summary results for analytics
```

## Other notes
- Uses OpenRouter API (configured via LLM_OPENAI_COMPATIBLE_API_KEY in tests/bench/.env or env var)
- Default model is anthropic/claude-haiku-4-5
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
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.config import settings

from .beam_common import (
    ConversationResult,
    QuestionResult,
    calculate_ability_scores,
    calculate_tokens,
    extract_messages_from_chat_data,
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

# OpenRouter model format for baseline testing
MODEL_BEING_TESTED = "anthropic/claude-haiku-4.5"


class BEAMBaselineRunner:
    """
    Executes BEAM benchmark tests directly against Claude Sonnet 4.5.
    """

    def __init__(
        self,
        data_dir: Path,
    ):
        """
        Initialize the BEAM baseline test runner.

        Args:
            data_dir: Path to the BEAM data directory
        """
        self.data_dir: Path = data_dir

        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Initialize OpenRouter client for model being tested and judging
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

        # Model to use for answering questions (OpenRouter format)
        self.answer_model: str = MODEL_BEING_TESTED

    def _format_conversation_context(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 140000,
    ) -> str:
        """
        Format conversation messages into a context string, truncating from the
        beginning if the total exceeds max_tokens.

        Args:
            messages: List of messages with 'role' and 'content' keys
            max_tokens: Maximum tokens allowed for the context (default: 140000
                        to account for Claude's tokenizer producing ~30% more tokens
                        than o200k_base, plus room for system prompt and question)

        Returns:
            Formatted conversation transcript string
        """
        # Calculate tokens for each message and find where to start
        # Note: We use o200k_base tokenizer but Claude's tokenizer produces ~30% more
        # tokens, so we use a conservative max_tokens limit to compensate
        message_tokens: list[int] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            role_label = "User" if role == "user" else "Assistant"
            formatted = f"{role_label}: {content}\n"
            message_tokens.append(calculate_tokens(formatted))

        total_tokens = sum(message_tokens)

        # Find the starting index to fit within max_tokens
        start_idx = 0
        if total_tokens > max_tokens:
            running_total = total_tokens
            for i, tokens in enumerate(message_tokens):
                if running_total <= max_tokens:
                    break
                running_total -= tokens
                start_idx = i + 1

            truncated_count = start_idx
            print(
                f"    [TRUNCATION] Removed {truncated_count} messages from start "
                + f"({total_tokens:,} -> {running_total:,} tokens)"
            )

        # Build the context string from start_idx onwards
        lines: list[str] = []
        lines.append("=== CONVERSATION HISTORY ===\n")

        if start_idx > 0:
            lines.append(
                f"[... {start_idx} earlier messages truncated to fit context window ...]\n"
            )

        for msg in messages[start_idx:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            role_label = "User" if role == "user" else "Assistant"
            lines.append(f"{role_label}: {content}\n")

        lines.append("=== END CONVERSATION HISTORY ===")
        return "\n".join(lines)

    async def _process_single_question(
        self,
        conversation_context: str,
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

            # Build system prompt with cache control for the conversation context
            # The context is identical for all questions, so caching saves tokens/latency
            system_prompt = f"""You are a helpful assistant with memory of past conversations.

Below is a history of past conversations. Use this history to answer the user's question accurately.

{conversation_context}"""

            # Call model via OpenRouter with full context
            try:
                response = await self.openrouter_client.chat.completions.create(
                    model=self.answer_model,
                    max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": question,
                        },
                    ],
                )

                if not response.choices or not response.choices[0].message.content:
                    actual_response = ""
                else:
                    actual_response = response.choices[0].message.content
            except Exception as e:
                self.logger.error(f"Error calling Claude API: {e}")
                actual_response = f"Error: {e}"

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
        self, context_length: str, conversation_id: str
    ) -> ConversationResult:
        """
        Execute BEAM benchmark for a single conversation using direct Claude context.

        Args:
            context_length: Context length (100K, 500K, 1M, 10M)
            conversation_id: Conversation ID

        Returns:
            Conversation execution results
        """
        start_time = time.time()

        print(f"\n{'=' * 80}")
        print(
            f"Executing BEAM conversation {conversation_id} ({context_length} context) [BASELINE]"
        )
        print(f"{'=' * 80}")

        workspace_id = f"baseline_{context_length}_{conversation_id}"

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

            # Extract all messages
            messages = extract_messages_from_chat_data(chat_data)
            result["total_messages"] = len(messages)
            result["total_turns"] = len(messages)

            # Calculate token count
            total_tokens = sum(calculate_tokens(m["content"]) for m in messages)
            print(
                f"[{workspace_id}] Context: {len(messages)} messages, ~{total_tokens:,} tokens"
            )

            # Format conversation as context
            conversation_context = self._format_conversation_context(messages)

            print(f"[{workspace_id}] Executing questions...")

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
                            conversation_context,
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
            f"Running {len(conversation_ids)} conversations from {context_length} context length [BASELINE]"
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
                    self.execute_conversation(context_length, conv_id)
                    for conv_id in batch
                ]
            )

            all_results.extend(batch_results)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        return all_results, overall_duration


async def main() -> int:
    """Main entry point for the BEAM baseline test runner."""
    parser = argparse.ArgumentParser(
        description="Run BEAM benchmark tests directly against Claude (baseline, no Honcho)",
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

    args = parser.parse_args()

    # Setup data directory
    data_dir = Path(__file__).parent / "beam_data"
    if not data_dir.exists():
        print(f"Error: BEAM data directory not found at {data_dir}")
        return 1

    # Create runner
    runner = BEAMBaselineRunner(data_dir=data_dir)

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
                f"tests/bench/eval_results/beam_baseline_{args.context_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            args.context_length,
            total_elapsed,
            output_file,
            metadata_extra={
                "runner_type": "baseline_direct_context",
                "model": MODEL_BEING_TESTED,
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
