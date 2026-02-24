"""
Honcho LoCoMo Summary Evaluation Runner

Evaluates Honcho's summary-making process by using only session summaries
as context for a base model to answer LoCoMo questions.

This isolates the quality of Honcho's summarization: instead of using the
full dialectic agent or raw conversation context, we retrieve the generated
summaries and feed them to a base LLM. The resulting scores measure how much
information the summaries retain.

## Comparison points

- `locomo_baseline.py`: Raw conversation as context (upper bound for the model)
- `locomo_summary.py` (this file): Honcho summaries only as context
- `locomo.py`: Full Honcho memory system (dialectic agent)

## To use

0. Set up env:
```
uv sync
source .venv/bin/activate
```

1. Run the test harness:
```
python -m tests.bench.harness
```

2. Run this file with the LoCoMo dataset:
```
python -m tests.bench.locomo_summary --data-file tests/bench/locomo_data/locomo10.json
```

Optional arguments:
```
--timeout: Timeout for deriver queue to empty in seconds (default: 10 minutes)
--base-api-port: Base port for Honcho API instances (default: 8000)
--pool-size: Number of Honcho instances in the pool (default: 1)
--batch-size: Number of conversations to run concurrently in each batch (default: 1)
--json-output: Path to write JSON summary results for analytics
--cleanup-workspace: Delete workspace after executing each conversation (default: False)
--sample-id: Run only the conversation with this sample_id (skips all others)
--test-count: Number of conversations to run (default: all)
--question-count: Number of questions per conversation to run (default: all)
--model: Model to use for answering questions given summary context (default: anthropic/claude-haiku-4.5)
```

## Other notes
- Summaries are enabled (overriding the base runner default) so the deriver generates them
- Uses OpenRouter API for the base model (configured via LLM_OPENAI_COMPATIBLE_API_KEY)
- Evaluation uses the same GPT-4o-mini judge as the standard LoCoMo runner
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from honcho.api_types import (
    MessageCreateParams,
    SessionConfiguration,
    SummaryConfiguration,
)
from honcho.session import SessionPeerConfig
from openai import AsyncOpenAI

from src.config import settings

from .locomo import format_message_with_image
from .locomo_common import (
    CATEGORY_NAMES,
    ConversationResult,
    QuestionResult,
    calculate_category_scores,
    calculate_tokens,
    extract_sessions,
    filter_questions,
    generate_json_summary,
    get_evidence_context,
    judge_response,
    load_locomo_data,
    parse_locomo_date,
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

# Default model for answering questions given summary context
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


class LoCoMoSummaryRunner(BaseRunner[ConversationResult]):
    """
    Evaluates Honcho summary quality by using only summaries as context
    for a base model to answer LoCoMo questions.
    """

    def __init__(
        self,
        config: RunnerConfig,
        data_file: Path,
        sample_id: str | None = None,
        test_count: int | None = None,
        question_count: int | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize the LoCoMo summary evaluation runner.

        Args:
            config: Common runner configuration
            data_file: Path to the LoCoMo JSON file
            sample_id: Optional sample_id to run only that conversation
            test_count: Optional number of conversations to run
            question_count: Optional limit on questions per conversation
            model: Model to use for answering questions given summary context
        """
        self.data_file: Path = data_file
        self.sample_id_filter: str | None = sample_id
        self.test_count: int | None = test_count
        self.question_count: int | None = question_count
        self.model: str = model

        # Initialize base class
        super().__init__(config)

        # Initialize OpenRouter client for the model under test
        openrouter_base_url = os.getenv(
            "LLM_OPENAI_COMPATIBLE_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.openrouter_client: AsyncOpenAI = create_openai_client(
            base_url=openrouter_base_url,
            env_key_name="LLM_OPENAI_COMPATIBLE_API_KEY",
        )

        # Initialize OpenAI client for judging responses
        self.openai_client: AsyncOpenAI = create_openai_client()

    def get_metrics_prefix(self) -> str:
        return "locomo_summary"

    def _get_session_configuration(self) -> SessionConfiguration:
        """Enable summaries so the deriver generates them during processing."""
        return SessionConfiguration(summary=SummaryConfiguration(enabled=True))

    def load_items(self) -> list[Any]:
        """Load conversations from the data file."""
        conversations = load_locomo_data(self.data_file)

        # Filter by sample_id if specified
        if self.sample_id_filter is not None:
            conversations = [
                c for c in conversations if c.get("sample_id") == self.sample_id_filter
            ]
            if not conversations:
                print(
                    f"Error: No conversation found with sample_id '{self.sample_id_filter}'"
                )
                return []
            print(f"Filtering to sample_id '{self.sample_id_filter}'")

        # Limit by test_count
        if self.test_count is not None and self.test_count > 0:
            conversations = conversations[: self.test_count]
            print(f"Limiting to {len(conversations)} conversations")

        return conversations

    def get_workspace_id(self, item: Any) -> str:
        """Return workspace ID for a conversation."""
        sample_id = item.get("sample_id", "unknown")
        return f"locomo_summary_{sample_id}"

    def get_session_id(self, item: Any, workspace_id: str) -> str:
        """Return session ID for a conversation."""
        return f"{workspace_id}_session"

    async def setup_peers(self, ctx: ItemContext, item: Any) -> None:
        """Create peers using speaker names as IDs."""
        conversation = item.get("conversation", {})
        speaker_a = conversation.get("speaker_a", "User")
        speaker_b = conversation.get("speaker_b", "Assistant")

        ctx.peers["speaker_a"] = await ctx.honcho_client.aio.peer(id=speaker_a)
        ctx.peers["speaker_b"] = await ctx.honcho_client.aio.peer(id=speaker_b)
        ctx.peers["_speaker_a_name"] = speaker_a
        ctx.peers["_speaker_b_name"] = speaker_b

    async def setup_session(self, ctx: ItemContext, item: Any) -> None:
        """Create and configure session - observe BOTH peers."""
        peer_a = ctx.peers["speaker_a"]
        peer_b = ctx.peers["speaker_b"]

        ctx.session = await ctx.honcho_client.aio.session(
            id=ctx.session_id, configuration=self._get_session_configuration()
        )

        # Observe both peers since questions ask about both speakers
        await ctx.session.aio.add_peers(
            [
                (peer_a, SessionPeerConfig(observe_me=True, observe_others=False)),
                (peer_b, SessionPeerConfig(observe_me=True, observe_others=False)),
            ]
        )

    async def ingest_messages(self, ctx: ItemContext, item: Any) -> int:
        """Ingest conversation messages into the session."""
        conversation = item.get("conversation", {})
        speaker_a = ctx.peers["_speaker_a_name"]
        speaker_b = ctx.peers["_speaker_b_name"]
        peer_a = ctx.peers["speaker_a"]
        peer_b = ctx.peers["speaker_b"]

        # Extract and ingest all sessions
        sessions = extract_sessions(conversation)

        messages: list[MessageCreateParams] = []
        total_tokens = 0

        for date_str, session_messages in sessions:
            session_date = parse_locomo_date(date_str) if date_str else None

            for msg in session_messages:
                speaker = msg.get("speaker", "")
                content, metadata = format_message_with_image(msg)
                total_tokens += calculate_tokens(content)

                # Map speaker to peer by name
                if speaker == speaker_a:
                    messages.append(
                        peer_a.message(
                            content, metadata=metadata, created_at=session_date
                        )
                    )
                elif speaker == speaker_b:
                    messages.append(
                        peer_b.message(
                            content, metadata=metadata, created_at=session_date
                        )
                    )

        # Store token count for results
        ctx.peers["_total_tokens"] = total_tokens
        ctx.peers["_total_sessions"] = len(sessions)

        # Add messages in batches of 100
        for i in range(0, len(messages), 100):
            batch = messages[i : i + 100]
            await ctx.session.aio.add_messages(batch)

        return len(messages)

    def get_dream_observers(self, item: Any) -> list[str]:
        """Return both speaker names - LoCoMo triggers dreams for both."""
        conversation = item.get("conversation", {})
        speaker_a = conversation.get("speaker_a", "User")
        speaker_b = conversation.get("speaker_b", "Assistant")
        return [speaker_a, speaker_b]

    async def _retrieve_summary_context(self, ctx: ItemContext) -> str:
        """
        Retrieve summaries from the session and format them as context.

        Prefers the long summary if available, falls back to short summary.

        Returns:
            Formatted summary context string, or a note if no summaries available.
        """
        workspace_id = ctx.workspace_id

        summaries = await ctx.session.aio.summaries()

        parts: list[str] = []

        if summaries.long_summary:
            parts.append(
                f"[Long Summary ({summaries.long_summary.token_count} tokens)]"
            )
            parts.append(summaries.long_summary.content)
            print(
                f"  [{workspace_id}] Retrieved long summary: {summaries.long_summary.token_count} tokens"
            )

        if summaries.short_summary:
            parts.append(
                f"\n[Short Summary ({summaries.short_summary.token_count} tokens)]"
            )
            parts.append(summaries.short_summary.content)
            print(
                f"  [{workspace_id}] Retrieved short summary: {summaries.short_summary.token_count} tokens"
            )

        if not parts:
            print(f"  [{workspace_id}] WARNING: No summaries available!")
            return "(No summaries were generated for this conversation.)"

        return "\n".join(parts)

    async def execute_questions(
        self, ctx: ItemContext, item: Any
    ) -> ConversationResult:
        """Execute all questions using only summary context."""
        start_time = time.time()
        sample_id = item.get("sample_id", "unknown")
        conversation = item.get("conversation", {})
        qa_list = item.get("qa", [])
        workspace_id = ctx.workspace_id

        speaker_a = ctx.peers["_speaker_a_name"]
        speaker_b = ctx.peers["_speaker_b_name"]

        result: ConversationResult = {
            "sample_id": sample_id,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "total_sessions": ctx.peers.get("_total_sessions", 0),
            "total_turns": 0,
            "total_tokens": ctx.peers.get("_total_tokens", 0),
            "question_results": [],
            "category_scores": {},
            "overall_score": 0.0,
            "error": None,
            "start_time": start_time,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

        # Retrieve summaries as context
        print(f"[{workspace_id}] Retrieving summaries...")
        summary_context = await self._retrieve_summary_context(ctx)

        # Build system prompt with summary context
        system_prompt = (
            f"You are a helpful assistant with memory of past conversations "
            f"between {speaker_a} and {speaker_b}.\n\n"
            f"Below are summaries of their past conversations. Use these summaries "
            f"to answer the user's question as accurately as possible.\n\n"
            f"=== CONVERSATION SUMMARIES ===\n"
            f"{summary_context}\n"
            f"=== END SUMMARIES ==="
        )

        # Filter questions
        filtered_qa = filter_questions(
            qa_list,
            exclude_adversarial=True,
            test_count=self.question_count,
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
                # Ask the base model with summary context
                response = await self.openrouter_client.chat.completions.create(
                    model=self.model,
                    max_tokens=settings.DIALECTIC.MAX_OUTPUT_TOKENS,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                )

                if not response.choices or not response.choices[0].message.content:
                    actual_response = ""
                else:
                    actual_response = response.choices[0].message.content

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
            passed_count = sum(1 for qr in result["question_results"] if qr["passed"])
            result["overall_score"] = passed_count / len(result["question_results"])

        result["end_time"] = time.time()
        result["duration_seconds"] = result["end_time"] - result["start_time"]

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
                f"tests/bench/eval_results/locomo_summary_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            total_duration,
            output_file,
            metadata_extra={
                "data_file": str(self.data_file),
                "runner_type": "summary_context",
                "model": self.model,
                "base_api_port": self.config.base_api_port,
                "pool_size": self.config.pool_size,
                "timeout_seconds": self.config.timeout_seconds,
                "deriver_settings": settings.DERIVER.model_dump(),
                "summary_settings": settings.SUMMARY.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
            },
        )


def main() -> int:
    """Main entry point for the LoCoMo summary evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Evaluate Honcho summary quality using LoCoMo questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data-file tests/bench/locomo_data/locomo10.json
  %(prog)s --data-file locomo10.json --pool-size 4
  %(prog)s --data-file locomo10.json --sample-id "conv-26"
  %(prog)s --data-file locomo10.json --model anthropic/claude-sonnet-4.5
  %(prog)s --data-file locomo10.json --test-count 5 --question-count 20
        """,
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to LoCoMo JSON file (required)",
    )

    # Add common arguments shared across all runners
    add_common_arguments(parser)

    # Summary-eval-specific arguments
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

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for answering questions given summary context (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()

    # Validate common arguments
    error = validate_common_arguments(args)
    if error:
        print(error)
        return 1

    # Validate locomo-specific arguments
    if not args.data_file.exists():
        print(f"Error: Data file {args.data_file} does not exist")
        return 1

    # Create config and runner
    config = RunnerConfig.from_args(args, default_timeout=600)

    runner = LoCoMoSummaryRunner(
        config=config,
        data_file=args.data_file,
        sample_id=args.sample_id,
        test_count=args.test_count,
        question_count=args.question_count,
        model=args.model,
    )

    return runner.run_and_summarize()


if __name__ == "__main__":
    exit(main())
