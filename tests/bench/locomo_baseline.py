"""
LoCoMo Baseline Test Runner (Direct Claude Context)

A script that executes LoCoMo benchmark tests directly against Claude
by feeding the entire conversation history into the context window.

This serves as a baseline comparison against Honcho's memory framework.

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

1. Run this file with the LoCoMo dataset:
```
python -m tests.bench.locomo_baseline --data-file tests/bench/locomo_data/locomo10.json
```

Optional arguments:
```
--anthropic-api-key: Anthropic API key (can be set in .env as LLM_ANTHROPIC_API_KEY)
--batch-size: Number of conversations to run concurrently in each batch (default: 1)
--json-output: Path to write JSON summary results for analytics
--sample-id: Run only the conversation with this sample_id (skips all others)
--test-count: Number of conversations to run (default: all)
--question-count: Number of questions per conversation to run (default: all)
```

## Other notes
- Uses Anthropic API directly (configured via LLM_ANTHROPIC_API_KEY in tests/bench/.env or env var)
- Default model is claude-haiku-4-5 for baseline comparison
- Evaluation uses F1 score following the LoCoMo paper methodology
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
from openai import AsyncOpenAI

from src.config import settings

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
    get_evidence_context,
    judge_response,
    load_locomo_data,
    print_summary,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")

MODEL_BEING_TESTED = "claude-haiku-4-5"


class LoCoMoBaselineRunner:
    """
    Executes LoCoMo benchmark tests directly against Claude.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
    ):
        """
        Initialize the LoCoMo baseline test runner.

        Args:
            anthropic_api_key: Anthropic API key (optional, uses env var if not provided)
        """
        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Initialize Anthropic client
        api_key = anthropic_api_key or os.getenv("LLM_ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_ANTHROPIC_API_KEY is not set in tests/bench/.env or environment"
            )

        self.anthropic_client: AsyncAnthropic = AsyncAnthropic(api_key=api_key)

        # Initialize OpenAI client for judging responses
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.openai_client: AsyncOpenAI = AsyncOpenAI(api_key=openai_api_key)

    def _format_conversation_context(
        self,
        conversation: dict[str, Any],
    ) -> str:
        """
        Format conversation sessions into a context string.

        Args:
            conversation: The conversation dict containing session data

        Returns:
            Formatted conversation transcript string
        """
        speaker_a = conversation.get("speaker_a", "User")
        speaker_b = conversation.get("speaker_b", "Assistant")

        lines: list[str] = []
        lines.append("=== CONVERSATION HISTORY ===\n")
        lines.append(f"This is a conversation between {speaker_a} and {speaker_b}.\n")

        sessions = extract_sessions(conversation)

        for session_idx, (date_str, messages) in enumerate(sessions, 1):
            lines.append(f"\n--- Session {session_idx} ({date_str}) ---\n")

            for msg in messages:
                speaker = msg.get("speaker", "Unknown")
                text = msg.get("text", "")
                lines.append(f"{speaker}: {text}\n")

        lines.append("\n=== END CONVERSATION HISTORY ===")
        return "\n".join(lines)

    async def execute_conversation(
        self,
        conversation_data: dict[str, Any],
        question_count: int | None = None,
    ) -> ConversationResult:
        """
        Execute LoCoMo benchmark for a single conversation using direct Claude context.

        Args:
            conversation_data: Dictionary containing conversation and QA data
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
        print(f"Executing LoCoMo conversation {sample_id} [BASELINE]")
        print(f"Speakers: {speaker_a} and {speaker_b}")
        print(f"{'=' * 80}")

        workspace_id = f"baseline_{sample_id}"

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
            "error": None,
            "start_time": start_time,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        try:
            # Extract sessions and count turns/tokens
            sessions = extract_sessions(conversation)
            result["total_sessions"] = len(sessions)

            total_tokens = 0
            for _date_str, messages in sessions:
                for msg in messages:
                    result["total_turns"] += 1
                    total_tokens += calculate_tokens(msg.get("text", ""))

            result["total_tokens"] = total_tokens

            print(
                f"[{workspace_id}] Context: {len(sessions)} sessions, {result['total_turns']} turns, ~{total_tokens:,} tokens"
            )

            # Format conversation as context
            conversation_context = self._format_conversation_context(conversation)

            # Filter questions
            filtered_qa = filter_questions(
                qa_list,
                exclude_adversarial=False,
                test_count=question_count,
            )

            print(f"[{workspace_id}] Executing {len(filtered_qa)} questions...")

            # Build system prompt with cache control for the conversation context
            system_prompt = f"""You are a helpful assistant with memory of past conversations between {speaker_a} and {speaker_b}.

Below is the history of their past conversations. Use this history to answer the user's question accurately.

{conversation_context}"""

            # Execute questions
            for q_idx, qa in enumerate(filtered_qa):
                question = qa.get("question", "")
                expected_answer = qa.get("answer", "")
                category = qa.get("category", 0)
                evidence = qa.get("evidence", [])
                category_name = CATEGORY_NAMES.get(category, f"category_{category}")

                print(f"  Q{q_idx + 1} [{category_name}]: {question[:80]}...")

                try:
                    # Call Claude with full context, using prompt caching
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
                                "content": question,
                            }
                        ],
                    )

                    if not response.content:
                        actual_response = ""
                    else:
                        content_block = response.content[0]
                        actual_response = getattr(content_block, "text", "")

                    # Get evidence context for the judge
                    evidence_context = get_evidence_context(conversation, evidence)

                    # Judge the response
                    judgment = await judge_response(
                        self.openai_client,
                        question,
                        str(expected_answer),
                        actual_response,
                        evidence_context=evidence_context,
                    )

                    passed = judgment.get("passed", False)

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
                    print(f"    [{status}]")
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
                        judgment={"passed": False, "reasoning": str(e)},
                        passed=False,
                    )
                    result["question_results"].append(question_result)

            # Calculate category scores
            result["category_scores"] = calculate_category_scores(
                result["question_results"]
            )

            # Calculate overall score (pass rate)
            if result["question_results"]:
                passed_count = sum(
                    1 for qr in result["question_results"] if qr["passed"]
                )
                result["overall_score"] = passed_count / len(result["question_results"])

            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - result["start_time"]

            print(
                f"\n[{workspace_id}] Completed in {format_duration(result['duration_seconds'])}"
            )
            print(f"Overall Score: {result['overall_score']:.3f}")

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

        print(f"Running {len(conversations)} conversations from {data_file} [BASELINE]")

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
                    self.execute_conversation(conv, question_count=question_count)
                    for conv in batch
                ]
            )

            all_results.extend(batch_results)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        return all_results, overall_duration


async def main() -> int:
    """Main entry point for the LoCoMo baseline test runner."""
    parser = argparse.ArgumentParser(
        description="Run LoCoMo benchmark tests directly against Claude (baseline, no Honcho)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data-file tests/bench/locomo_data/locomo10.json
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
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key (optional, can use LLM_ANTHROPIC_API_KEY env var)",
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

    # Create test runner
    runner = LoCoMoBaselineRunner(
        anthropic_api_key=args.anthropic_api_key,
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

        # Generate JSON output
        if args.json_output:
            output_file = args.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/locomo_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            total_elapsed,
            output_file,
            metadata_extra={
                "data_file": str(args.data_file),
                "runner_type": "baseline_direct_context",
                "model": MODEL_BEING_TESTED,
            },
        )

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
