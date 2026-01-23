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
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from honcho.api_types import MessageCreateParams
from honcho.session import SessionPeerConfig
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
    create_anthropic_client,
    create_openai_client,
    validate_common_arguments,
)

# Load .env from bench directory
bench_dir = Path(__file__).parent
load_dotenv(bench_dir / ".env")


def format_message_with_image(msg: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """
    Format a LoCoMo message with optional image caption appended.

    Args:
        msg: LoCoMo message dict with 'text', optional 'img_url', 'blip_caption', 'query'

    Returns:
        Tuple of (formatted_content, metadata_dict or None)
    """
    text = msg.get("text", "")
    blip_caption = msg.get("blip_caption")
    img_urls = msg.get("img_url", [])
    query = msg.get("query")

    # Append caption to content so deriver can see it
    content = f"{text}\n\n[Image shared: {blip_caption}]" if blip_caption else text

    # Build metadata if image data exists
    metadata: dict[str, Any] | None = None
    if img_urls or blip_caption or query:
        metadata = {}
        if img_urls:
            metadata["img_urls"] = img_urls
        if blip_caption:
            metadata["blip_caption"] = blip_caption
        if query:
            metadata["image_query"] = query

    return content, metadata


def determine_question_target(question: str, speaker_a: str, speaker_b: str) -> str:
    """
    Determine which speaker a question is asking about based on the question text.

    Args:
        question: The question text
        speaker_a: Name of speaker A (e.g., "Caroline")
        speaker_b: Name of speaker B (e.g., "Melanie")

    Returns:
        The name of the speaker the question is about (speaker_a or speaker_b)
    """
    question_lower = question.lower()
    speaker_a_lower = speaker_a.lower()
    speaker_b_lower = speaker_b.lower()

    # Check for possessive forms too (e.g., "Melanie's kids")
    a_in_question = (
        speaker_a_lower in question_lower or f"{speaker_a_lower}'s" in question_lower
    )
    b_in_question = (
        speaker_b_lower in question_lower or f"{speaker_b_lower}'s" in question_lower
    )

    if a_in_question and not b_in_question:
        return speaker_a
    elif b_in_question and not a_in_question:
        return speaker_b
    else:
        # Question mentions both or neither - default to speaker_a
        return speaker_a


class LoCoMoRunner(BaseRunner[ConversationResult]):
    """
    Executes LoCoMo benchmark tests against a Honcho instance.
    """

    def __init__(
        self,
        config: RunnerConfig,
        data_file: Path,
        anthropic_api_key: str | None = None,
        sample_id: str | None = None,
        test_count: int | None = None,
        question_count: int | None = None,
    ):
        """
        Initialize the LoCoMo test runner.

        Args:
            config: Common runner configuration
            data_file: Path to the LoCoMo JSON file
            anthropic_api_key: Anthropic API key for judging responses
            sample_id: Optional sample_id to run only that conversation
            test_count: Optional number of conversations to run
            question_count: Optional limit on questions per conversation
        """
        self.data_file: Path = data_file
        self.sample_id_filter: str | None = sample_id
        self.test_count: int | None = test_count
        self.question_count: int | None = question_count

        # Initialize base class
        super().__init__(config)

        # Initialize LLM clients
        self.anthropic_client: AsyncAnthropic = create_anthropic_client(
            anthropic_api_key
        )
        self.openai_client: AsyncOpenAI = create_openai_client()

    def get_metrics_prefix(self) -> str:
        return "locomo"

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
        return f"locomo_{sample_id}"

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

        ctx.session = await ctx.honcho_client.aio.session(id=ctx.session_id)

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

    async def execute_questions(
        self, ctx: ItemContext, item: Any
    ) -> ConversationResult:
        """Execute all questions for the conversation."""
        sample_id = item.get("sample_id", "unknown")
        conversation = item.get("conversation", {})
        qa_list = item.get("qa", [])
        workspace_id = ctx.workspace_id

        speaker_a = ctx.peers["_speaker_a_name"]
        speaker_b = ctx.peers["_speaker_b_name"]
        peer_a = ctx.peers["speaker_a"]
        peer_b = ctx.peers["speaker_b"]

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
            "start_time": 0.0,
            "end_time": 0.0,
            "duration_seconds": 0.0,
        }

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

            # Determine which peer the question is about
            target_speaker = determine_question_target(question, speaker_a, speaker_b)
            target_peer = peer_a if target_speaker == speaker_a else peer_b

            print(
                f"  Q{q_idx + 1} [{category_name}] (asking {target_speaker}): {question}"
            )

            try:
                if self.config.use_get_context:
                    # Use get_context + LLM
                    context = await ctx.session.aio.context(
                        summary=True,
                        peer_target=target_speaker,
                        search_query=question,
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

                    actual_response = getattr(response.content[0], "text", "")
                else:
                    # Use dialectic .chat endpoint
                    actual_response = await target_peer.aio.chat(
                        question,
                        session=ctx.session_id,
                        reasoning_level=self.config.reasoning_level,
                    )
                    actual_response = (
                        actual_response if isinstance(actual_response, str) else ""
                    )

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
                f"tests/bench/eval_results/locomo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        generate_json_summary(
            results,
            total_duration,
            output_file,
            metadata_extra={
                "data_file": str(self.data_file),
                "base_api_port": self.config.base_api_port,
                "pool_size": self.config.pool_size,
                "timeout_seconds": self.config.timeout_seconds,
                "reasoning_level": self.config.reasoning_level,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
                "summary_settings": settings.SUMMARY.model_dump(),
            },
        )


def main() -> int:
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
  %(prog)s --data-file locomo10.json --reasoning-level high
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

    # LoCoMo-specific arguments
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key for response judging (optional)",
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

    runner = LoCoMoRunner(
        config=config,
        data_file=args.data_file,
        anthropic_api_key=args.anthropic_api_key,
        sample_id=args.sample_id,
        test_count=args.test_count,
        question_count=args.question_count,
    )

    return runner.run_and_summarize()


if __name__ == "__main__":
    exit(main())
