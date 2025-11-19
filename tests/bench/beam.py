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
--anthropic-api-key: Anthropic API key for response judging (can be set in .env as LLM_ANTHROPIC_API_KEY)
--timeout: Timeout for deriver queue to empty in seconds (default: 10 minutes)
--base-api-port: Base port for Honcho API instances (default: 8000)
--pool-size: Number of Honcho instances in the pool (default: 1)
--batch-size: Number of conversations to run concurrently in each batch (default: 1)
--json-output: Path to write JSON summary results for analytics
--cleanup-workspace: Delete workspace after executing each conversation (default: False)
--use-get-context: Use get_context + judge LLM instead of dialectic .chat endpoint (default: False)
```

## Other notes
- Judge is Claude Sonnet 4.5
- Evaluation follows the paper's nugget-based methodology with 0/0.5/1 scoring
- Event ordering uses Kendall tau-b coefficient
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)
from typing_extensions import TypedDict

try:
    from scipy.stats import (
        kendalltau,  # type: ignore  # pyright: ignore[reportUnknownVariableType]
    )
except ImportError:
    kendalltau = None  # type: ignore

from src.config import settings
from src.utils.metrics_collector import MetricsCollector

load_dotenv()


class QuestionResult(TypedDict):
    """Type definition for question evaluation results."""

    question: str
    answer: str | None
    actual_response: str
    memory_ability: str
    rubric: list[str]
    nugget_scores: list[dict[str, Any]] | None
    score: float
    passed: bool
    reasoning: str


class ConversationResult(TypedDict):
    """Type definition for conversation execution results."""

    conversation_id: str
    context_length: str
    workspace_id: str
    total_turns: int
    total_messages: int
    question_results: list[QuestionResult]
    ability_scores: dict[str, float]
    overall_score: float
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float


class BEAMRunner:
    """
    Executes BEAM benchmark tests against a Honcho instance.
    """

    def __init__(
        self,
        data_dir: Path,
        base_api_port: int = 8000,
        pool_size: int = 1,
        anthropic_api_key: str | None = None,
        timeout_seconds: int | None = None,
        cleanup_workspace: bool = False,
        use_get_context: bool = False,
    ):
        """
        Initialize the BEAM test runner.

        Args:
            data_dir: Path to the BEAM data directory
            base_api_port: Base port for Honcho API instances (default: 8000)
            pool_size: Number of Honcho instances in the pool (default: 1)
            anthropic_api_key: Anthropic API key for judging responses
            timeout_seconds: Timeout for deriver queue in seconds
            cleanup_workspace: If True, delete workspace after executing conversation
            use_get_context: If True, use get_context + judge LLM instead of dialectic .chat endpoint
        """
        self.data_dir: Path = data_dir
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.anthropic_api_key: str | None = anthropic_api_key
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 10000
        )
        self.cleanup_workspace: bool = cleanup_workspace
        self.use_get_context: bool = use_get_context

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

        if self.anthropic_api_key:
            self.anthropic_client: AsyncAnthropic = AsyncAnthropic(
                api_key=self.anthropic_api_key
            )
        else:
            api_key = os.getenv("LLM_ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("LLM_ANTHROPIC_API_KEY is not set")
            self.anthropic_client = AsyncAnthropic(api_key=api_key)

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

    def _format_duration(self, total_seconds: float) -> str:
        """Format a duration in seconds into a human-readable string."""
        minutes = int(total_seconds // 60)
        if minutes > 0:
            seconds_rounded = int(round(total_seconds - minutes * 60))
            if seconds_rounded == 60:
                minutes += 1
                seconds_rounded = 0
            return f"{minutes}m{seconds_rounded:02d}s"
        return f"{total_seconds:.2f}s"

    def _calculate_tokens(self, text: str) -> int:
        """Calculate tokens for a given text."""
        tokenizer = tiktoken.get_encoding("cl100k_base")
        try:
            return len(
                tokenizer.encode(
                    text,
                    disallowed_special=(
                        tokenizer.special_tokens_set - {"<|endoftext|>"}
                    ),
                )
            )
        except Exception:
            return len(text) // 4

    def load_conversation(
        self, context_length: str, conversation_id: str
    ) -> dict[str, Any]:
        """
        Load a BEAM conversation from the data directory.

        Args:
            context_length: Context length (100K, 500K, 1M, 10M)
            conversation_id: Conversation ID

        Returns:
            Dictionary containing conversation data and probing questions
        """
        conv_dir = self.data_dir / context_length / conversation_id

        # Load chat data
        chat_file = conv_dir / "chat.json"
        with open(chat_file) as f:
            chat_data = json.load(f)

        # Load probing questions
        questions_file = conv_dir / "probing_questions" / "probing_questions.json"
        with open(questions_file) as f:
            questions_data = json.load(f)

        return {"chat": chat_data, "questions": questions_data}

    def list_conversations(self, context_length: str) -> list[str]:
        """
        List all conversation IDs for a given context length.

        Args:
            context_length: Context length (100K, 500K, 1M, 10M)

        Returns:
            List of conversation ID strings
        """
        context_dir = self.data_dir / context_length
        return [
            d.name
            for d in sorted(context_dir.iterdir())
            if d.is_dir() and d.name.isdigit()
        ]

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

    async def judge_nugget_based(
        self,
        question: str,
        rubric: list[str],
        actual_response: str,
        memory_ability: str,
    ) -> dict[str, Any]:
        """
        Use an LLM to judge a response using nugget-based evaluation.

        Args:
            question: The question asked
            rubric: List of nuggets (atomic criteria) to check
            actual_response: Actual response from Honcho
            memory_ability: The memory ability being tested

        Returns:
            Judgment result with nugget scores and overall score
        """
        try:
            # Build the nugget evaluation prompt
            nuggets_formatted = "\n".join(
                [f"{i + 1}. {nugget}" for i, nugget in enumerate(rubric)]
            )

            system_prompt = f"""You are an expert judge evaluating AI responses for the {memory_ability} memory ability in the BEAM benchmark.

Your task is to evaluate whether the AI's response satisfies each atomic criterion (nugget) from the rubric.

SCORING INSTRUCTIONS:
For each nugget, assign a score:
- 1.0: The response fully satisfies this criterion
- 0.5: The response partially satisfies this criterion
- 0.0: The response does not satisfy this criterion

Be strict but fair in your evaluation. Focus on whether the response contains the required information or demonstrates the required behavior.

Always respond with valid JSON in this exact format:
{{
    "nugget_scores": [
        {{"nugget_index": 1, "score": 1.0, "reasoning": "brief explanation"}},
        {{"nugget_index": 2, "score": 0.5, "reasoning": "brief explanation"}}
    ],
    "overall_score": 0.75,
    "overall_reasoning": "brief summary of the evaluation"
}}"""

            user_prompt = f"""Question: "{question}"

Rubric (atomic criteria to check):
{nuggets_formatted}

Actual Response: "{actual_response}"

Evaluate the response against each nugget in the rubric. Provide a score for each nugget and calculate the overall score as the average of all nugget scores."""

            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2000,
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            if not response.content:
                raise ValueError("Anthropic returned empty response")

            content_block = response.content[0]
            judgment_text = getattr(content_block, "text", None)
            if judgment_text is None:
                raise ValueError(
                    f"No text content in response block: {type(content_block)}"
                )

            # Extract JSON from the response if it's wrapped in markdown
            if "```json" in judgment_text:
                json_start = judgment_text.find("```json") + 7
                json_end = judgment_text.find("```", json_start)
                judgment_text = judgment_text[json_start:json_end].strip()
            elif "```" in judgment_text:
                json_start = judgment_text.find("```") + 3
                json_end = judgment_text.find("```", json_start)
                judgment_text = judgment_text[json_start:json_end].strip()

            judgment = json.loads(judgment_text)
            return judgment

        except Exception as e:
            self.logger.error(f"Error judging response: {e}")
            # Fallback to simple 0 score
            return {
                "nugget_scores": [
                    {"nugget_index": i + 1, "score": 0.0, "reasoning": f"Error: {e}"}
                    for i in range(len(rubric))
                ],
                "overall_score": 0.0,
                "overall_reasoning": f"Evaluation failed due to error: {e}",
            }

    async def judge_event_ordering(
        self, question: str, rubric: list[str], actual_response: str
    ) -> dict[str, Any]:
        """
        Judge event ordering questions using Kendall tau-b coefficient.

        Args:
            question: The question asked
            rubric: List of expected events in correct order
            actual_response: Actual response from Honcho

        Returns:
            Judgment with Kendall tau-b score
        """
        try:
            # First, extract the events mentioned in the response
            system_prompt = """You are an expert at extracting ordered lists of events or items from text.

Your task is to extract the ordered list of events/items mentioned in a response.

Return valid JSON in this format:
{
    "extracted_events": ["first event", "second event", "third event"]
}"""

            user_prompt = f"""Question: "{question}"

Response: "{actual_response}"

Extract the ordered list of events or items mentioned in the response. Preserve the order as stated in the response."""

            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            if not response.content:
                raise ValueError("Anthropic returned empty response")

            content_block = response.content[0]
            extraction_text = getattr(content_block, "text", None)
            if extraction_text is None:
                raise ValueError(
                    f"No text content in response block: {type(content_block)}"
                )

            # Extract JSON
            if "```json" in extraction_text:
                json_start = extraction_text.find("```json") + 7
                json_end = extraction_text.find("```", json_start)
                extraction_text = extraction_text[json_start:json_end].strip()
            elif "```" in extraction_text:
                json_start = extraction_text.find("```") + 3
                json_end = extraction_text.find("```", json_start)
                extraction_text = extraction_text[json_start:json_end].strip()

            extracted = json.loads(extraction_text)
            extracted_events = extracted.get("extracted_events", [])

            # Now compute alignment and Kendall tau-b
            # Match extracted events to rubric events using LLM equivalence
            alignment = await self._align_events(rubric, extracted_events)

            # Compute Kendall tau-b
            tau: float
            if kendalltau is None:
                self.logger.warning(
                    "scipy not installed, cannot compute Kendall tau-b. Install with: uv pip install scipy"
                )
                tau = 0.0
            elif len(alignment) < 2:
                tau = 0.0
            else:
                # Create rank lists
                expected_ranks = list(range(len(alignment)))
                actual_ranks = [alignment[i] for i in range(len(alignment))]
                result_tuple: Any = kendalltau(expected_ranks, actual_ranks)
                # kendalltau returns a tuple, first element is the tau coefficient
                tau_value: Any = result_tuple[0]
                # Handle the return type properly - convert to float
                try:
                    tau = float(tau_value)
                    if tau != tau:  # Check for NaN
                        tau = 0.0
                except (TypeError, ValueError):
                    tau = 0.0

            return {
                "kendall_tau_b": tau,
                "extracted_events": extracted_events,
                "alignment": alignment,
                "overall_score": (tau + 1) / 2,  # Normalize to [0, 1]
                "overall_reasoning": f"Kendall tau-b coefficient: {tau:.3f}. Extracted {len(extracted_events)} events from response.",
            }

        except Exception as e:
            self.logger.error(f"Error in event ordering evaluation: {e}")
            return {
                "kendall_tau_b": 0.0,
                "extracted_events": [],
                "alignment": [],
                "overall_score": 0.0,
                "overall_reasoning": f"Evaluation failed due to error: {e}",
            }

    async def _align_events(
        self, expected_events: list[str], extracted_events: list[str]
    ) -> list[int]:
        """
        Align extracted events with expected events using LLM equivalence detection.

        Returns a list of indices mapping extracted events to expected events.
        """
        # For each extracted event, find the best match in expected events
        alignment: list[int] = []
        for extracted in extracted_events:
            best_match_idx: int = -1
            for i, expected in enumerate(expected_events):
                # Use simple string matching for now (can be enhanced with LLM)
                if (
                    expected.lower() in extracted.lower()
                    or extracted.lower() in expected.lower()
                ):
                    best_match_idx = i
                    break
            if best_match_idx >= 0:
                alignment.append(best_match_idx)
        return alignment

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
            conv_data = self.load_conversation(context_length, conversation_id)
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
                                                messages.append(user_peer.message(chunk))
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
                return result

            print(f"[{workspace_id}] Deriver queue empty. Executing questions...")

            # Execute questions for each memory ability
            for ability, questions in questions_data.items():
                print(
                    f"\n[{workspace_id}] Testing {ability} ({len(questions)} questions)"
                )

                for q_idx, q_data in enumerate(questions):
                    question = q_data["question"]
                    rubric = q_data.get("rubric", [])
                    answer = (
                        q_data.get("answer")
                        or q_data.get("ideal_response")
                        or q_data.get("ideal_answer")
                    )

                    print(f"  Q{q_idx + 1}: {question[:100]}...")

                    # Execute question using dialectic
                    if self.use_get_context:
                        context = await session.get_context(
                            summary=True,
                            peer_target="user",
                            last_user_message=question,
                        )
                        context_messages = context.to_anthropic(assistant="assistant")
                        context_messages.append({"role": "user", "content": question})

                        from typing import cast

                        response = await self.anthropic_client.messages.create(
                            model="claude-sonnet-4-5",
                            max_tokens=2048,
                            messages=cast(list[MessageParam], context_messages),
                        )

                        if not response.content:
                            actual_response = ""
                        else:
                            content_block = response.content[0]
                            actual_response = getattr(content_block, "text", "")
                    else:
                        actual_response = await user_peer.chat(question)
                        actual_response = (
                            actual_response if isinstance(actual_response, str) else ""
                        )

                    # Judge response based on memory ability
                    if ability == "event_ordering":
                        judgment = await self.judge_event_ordering(
                            question, rubric, actual_response
                        )
                        nugget_scores = None
                    else:
                        judgment = await self.judge_nugget_based(
                            question, rubric, actual_response, ability
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

                    result["question_results"].append(question_result)

                    status = "PASS" if score >= 0.5 else "FAIL"
                    print(f"    Score: {score:.2f} [{status}]")

            # Calculate ability scores
            ability_totals: dict[str, list[float]] = {}
            for qr in result["question_results"]:
                ability = qr["memory_ability"]
                if ability not in ability_totals:
                    ability_totals[ability] = []
                ability_totals[ability].append(qr["score"])

            for ability, scores in ability_totals.items():
                result["ability_scores"][ability] = sum(scores) / len(scores)

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
                f"\n[{workspace_id}] Completed in {self._format_duration(result['duration_seconds'])}"
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

    def print_summary(
        self, results: list[ConversationResult], total_elapsed_seconds: float
    ) -> None:
        """Print a summary of all test results."""
        print(f"\n{'=' * 80}")
        print("BEAM BENCHMARK EXECUTION SUMMARY")
        print(f"{'=' * 80}")

        total_conversations = len(results)
        total_questions = sum(len(r["question_results"]) for r in results)

        print(f"Total Conversations: {total_conversations}")
        print(f"Total Questions: {total_questions}")
        print(f"Total Test Time: {self._format_duration(total_elapsed_seconds)}")

        # Calculate average scores by ability
        ability_scores: dict[str, list[float]] = {}
        for result in results:
            for ability, score in result["ability_scores"].items():
                if ability not in ability_scores:
                    ability_scores[ability] = []
                ability_scores[ability].append(score)

        print("\nAverage Scores by Memory Ability:")
        for ability, scores in sorted(ability_scores.items()):
            avg_score = sum(scores) / len(scores)
            print(f"  {ability:30s}: {avg_score:.3f}")

        # Overall average
        overall_scores = [r["overall_score"] for r in results]
        overall_avg = (
            sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        )
        print(f"\n{'Overall Average Score':30s}: {overall_avg:.3f}")

        print(f"{'=' * 80}")

    def generate_json_summary(
        self,
        results: list[ConversationResult],
        context_length: str,
        total_elapsed_seconds: float,
        output_file: Path,
    ) -> None:
        """Generate a comprehensive JSON summary of test results."""
        # Calculate summary statistics
        total_conversations = len(results)
        total_questions = sum(len(r["question_results"]) for r in results)

        # Calculate average scores by ability
        ability_scores: dict[str, list[float]] = {}
        for result in results:
            for ability, score in result["ability_scores"].items():
                if ability not in ability_scores:
                    ability_scores[ability] = []
                ability_scores[ability].append(score)

        ability_averages = {
            ability: sum(scores) / len(scores)
            for ability, scores in ability_scores.items()
        }

        # Overall average
        overall_scores = [r["overall_score"] for r in results]
        overall_avg = (
            sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        )

        summary = {
            "metadata": {
                "context_length": context_length,
                "execution_timestamp": datetime.now().isoformat(),
                "runner_version": "1.0.0",
                "base_api_port": self.base_api_port,
                "pool_size": self.pool_size,
                "timeout_seconds": self.timeout_seconds,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
            },
            "summary_statistics": {
                "total_conversations": total_conversations,
                "total_questions": total_questions,
                "overall_average_score": overall_avg,
                "ability_averages": ability_averages,
            },
            "timing": {
                "total_duration_seconds": total_elapsed_seconds,
            },
            "detailed_results": results,
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nJSON summary written to: {output_file}")


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
        choices=["100K", "500K", "1M", "10M"],
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
        anthropic_api_key=args.anthropic_api_key,
        timeout_seconds=args.timeout,
        cleanup_workspace=args.cleanup_workspace,
        use_get_context=args.use_get_context,
    )

    try:
        # Determine which conversations to run
        if args.conversation_ids:
            conversation_ids = args.conversation_ids.split(",")
        else:
            conversation_ids = runner.list_conversations(args.context_length)

        # Run conversations
        results, total_elapsed = await runner.run_conversations(
            args.context_length, conversation_ids, args.batch_size
        )

        runner.print_summary(results, total_elapsed)

        # Generate JSON output
        if args.json_output:
            output_file = args.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/beam_{args.context_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        runner.generate_json_summary(
            results, args.context_length, total_elapsed, output_file
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
