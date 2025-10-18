"""
Honcho ImplexConv Test Runner

A script that executes ImplexConv tests against a running Honcho instance.
ImplexConv tests implicit reasoning by embedding evidence in semantically distant conversations.

Key differences from LongMemEval:
1. All conversations for a persona are in one JSON object
2. Questions test either "opposed" or "supportive" implicit reasoning
3. Retrieved_conv_ids indicate which conversations contain evidence
4. Opposed questions have explicit implicit_reasoning field

## Usage

python -m tests.bench.implexconv --test-file tests/bench/implexconv_data/opposed.json --reasoning-type opposed
python -m tests.bench.implexconv --test-file tests/bench/implexconv_data/supportive.json --reasoning-type supportive
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import tiktoken
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from honcho import AsyncHoncho
from honcho.async_client.session import SessionPeerConfig
from honcho_core.types.workspaces.sessions.message_create_param import (
    MessageCreateParam,
)
from typing_extensions import TypedDict

from src.config import settings
from src.utils.metrics_collector import MetricsCollector

load_dotenv()


class QuestionData(TypedDict):
    """Type definition for ImplexConv question."""

    question: str
    answer: str
    opposed_implicit_reasoning: str | None  # Only for opposed type
    retrieved_conv_ids: list[str]


class ConversationExample(TypedDict):
    """Type definition for ImplexConv conversation example."""

    conversation: dict[str, str]  # Keys are conv_ids, values are conversation text
    qa: list[QuestionData]


class SessionResult(TypedDict):
    """Type definition for session creation results."""

    conv_id: str
    message_count: int
    is_evidence: bool


class QueryResult(TypedDict):
    """Type definition for query execution results."""

    question: str
    expected_answer: str
    actual_response: str
    judgment: dict[str, Any]
    token_efficiency: dict[str, Any] | None
    retrieval_analysis: dict[str, Any] | None


class TestResult(TypedDict):
    """Type definition for test execution results."""

    example_id: str
    question_index: int
    reasoning_type: Literal["opposed", "supportive"]
    workspace_id: str
    sessions_created: list[SessionResult]
    query_executed: QueryResult | None
    passed: bool
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float
    output_lines: list[str]


class ImplexConvRunner:
    """
    Executes ImplexConv tests against a Honcho instance.

    ImplexConv tests implicit reasoning where evidence is embedded in
    semantically distant conversations that require inference rather
    than direct retrieval.
    """

    def __init__(
        self,
        base_api_port: int = 8000,
        pool_size: int = 1,
        anthropic_api_key: str | None = None,
        timeout_seconds: int | None = None,
        reasoning_type: Literal["opposed", "supportive"] = "opposed",
        cleanup_workspace: bool = False,
        use_get_context: bool = False,
    ):
        """
        Initialize the test runner.

        Args:
            base_api_port: Base port for Honcho API instances
            pool_size: Number of Honcho instances in the pool
            anthropic_api_key: Anthropic API key for judging responses
            timeout_seconds: Timeout for deriver queue in seconds
            reasoning_type: Type of reasoning being tested (opposed or supportive)
            cleanup_workspace: If True, delete workspace after executing question
            use_get_context: If True, use get_context + judge LLM instead of dialectic
        """
        self.base_api_port: int = base_api_port
        self.pool_size: int = pool_size
        self.anthropic_api_key: str | None = anthropic_api_key
        self.timeout_seconds: int = (
            timeout_seconds if timeout_seconds is not None else 10000
        )
        self.reasoning_type: Literal["opposed", "supportive"] = reasoning_type
        self.cleanup_workspace: bool = cleanup_workspace
        self.use_get_context: bool = use_get_context

        # Initialize metrics collector
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.metrics_collector.start_collection(
            f"implexconv_{reasoning_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Configure logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Suppress HTTP request logs
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

    def get_honcho_url_for_index(self, example_index: int) -> str:
        """Get the Honcho URL for a given example index using round-robin."""
        instance_id = example_index % self.pool_size
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

    def _calculate_total_tokens(self, conversations: dict[str, str]) -> int:
        """Calculate total tokens from all conversations."""
        tokenizer = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0

        for conv_text in conversations.values():
            try:
                total_tokens += len(
                    tokenizer.encode(
                        conv_text,
                        disallowed_special=(
                            tokenizer.special_tokens_set - {"<|endoftext|>"}
                        ),
                    )
                )
            except Exception:
                total_tokens += len(conv_text) // 4
                self.logger.warning("Error tokenizing content. Using rough estimate")

        return total_tokens

    def _get_latest_tokens_used(self) -> int | None:
        """Get the tokens_used_estimate from the most recent dialectic_chat metric."""
        metrics_file = Path(settings.LOCAL_METRICS_FILE)
        if not metrics_file.exists():
            return None

        try:
            with open(metrics_file) as f:
                lines = f.readlines()

            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_name = data.get("task_name", "")
                    if task_name.startswith("dialectic_chat_"):
                        for metric in data.get("metrics", []):
                            metric_name = metric.get("name", "")
                            if metric_name.endswith("tokens_used_estimate"):
                                return int(metric.get("value", 0))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        except Exception as e:
            self.logger.warning(f"Error reading metrics file: {e}")

        return None

    def _parse_conversation(self, conv_text: str) -> list[dict[str, str]]:
        """
        Parse a conversation string into list of message dicts.

        Format: "Speaker1: ...\nAssistant: ...\n\nSpeaker1: ..."
        Note: Double newline separates turns, single newline within a message
        """
        messages: list[dict[str, str]] = []

        # Split by double newline to get turns, but keep single newlines within messages
        turns = conv_text.split("\n\n")

        for turn in turns:
            if not turn.strip():
                continue

            # Each turn might have multiple lines from the same speaker
            lines = turn.split("\n")
            current_role = None
            current_content_parts = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this line starts with a role marker
                if line.startswith("Speaker1:"):
                    # Save previous message if exists
                    if current_role and current_content_parts:
                        content = " ".join(current_content_parts).strip()
                        if content:  # Only add if content is non-empty
                            messages.append({"role": current_role, "content": content})
                    current_role = "user"
                    current_content_parts = [line[9:].strip()]  # Remove "Speaker1:"
                elif line.startswith("Assistant:"):
                    # Save previous message if exists
                    if current_role and current_content_parts:
                        content = " ".join(current_content_parts).strip()
                        if content:  # Only add if content is non-empty
                            messages.append({"role": current_role, "content": content})
                    current_role = "assistant"
                    current_content_parts = [line[10:].strip()]  # Remove "Assistant:"
                else:
                    # Continuation of current message
                    if current_role:
                        current_content_parts.append(line)

            # Save the last message
            if current_role and current_content_parts:
                content = " ".join(current_content_parts).strip()
                if content:  # Only add if content is non-empty
                    messages.append({"role": current_role, "content": content})

        return messages

    def load_test_file(self, test_file: Path) -> list[ConversationExample]:
        """Load ImplexConv test definitions from a JSON file."""
        with open(test_file) as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, list):
            return cast(list[ConversationExample], data)
        elif isinstance(data, dict):
            # If it's a dict, wrap it in a list
            return cast(list[ConversationExample], [data])
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")

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

    async def judge_implicit_reasoning(
        self,
        question: str,
        expected_answer: str,
        actual_response: str,
        implicit_reasoning: str | None = None,
    ) -> dict[str, Any]:
        """
        Use an LLM to judge if the response demonstrates implicit reasoning.

        For opposed: Must show evidence of finding the blocking scenario
        For supportive: Must show evidence of confirming the trait
        """
        try:
            if self.reasoning_type == "opposed":
                system_prompt = f"""
You are evaluating implicit reasoning in long-term conversations for OPPOSED reasoning.

OPPOSED REASONING means: There is a scenario that PREVENTS or BLOCKS a persona trait.

IMPLICIT REASONING: "{implicit_reasoning}"

This implicit reasoning should PREVENT certain activities or suggestions in the response.

CRITICAL EVALUATION CRITERIA:
1. Did the AI find and use the implicit blocking scenario?
2. Does the response appropriately avoid suggestions that would be blocked by this scenario?
3. Is the response logically consistent with the implicit constraint?

Example:
- Implicit: "broke my leg"
- Question: "What sports should I do?"
- CORRECT: Suggests low-impact activities, acknowledges injury
- INCORRECT: Suggests basketball, running (ignores the injury)

SEMANTIC FLEXIBILITY: The response doesn't need to quote the implicit reasoning verbatim,
but must demonstrate awareness of the constraint through its suggestions and tone.

Respond with JSON: {{"passed": bool, "found_implicit": bool, "reasoning": "1-3 sentence explanation"}}
"""
            else:  # supportive
                system_prompt = """
You are evaluating implicit reasoning in long-term conversations for SUPPORTIVE reasoning.

SUPPORTIVE REASONING means: There is evidence that CONFIRMS or REINFORCES a persona trait.

CRITICAL EVALUATION CRITERIA:
1. Does the response confirm the persona trait exists?
2. Is the confirmation based on evidence from the conversation history (not just assumption)?
3. Does the response show the AI found implicit supporting evidence?

Example:
- Trait: "shares daily facts on blog"
- Question: "Do I actually post facts daily?"
- Evidence: "newsletter section takes forever to draft" (implies regular posting)
- CORRECT: "Yes, you've been consistently posting"
- INCORRECT: "I'm not sure" or "No" (missed the evidence)

SEMANTIC FLEXIBILITY: Accept responses that clearly confirm the trait even if phrased
differently than the expected answer.

Respond with JSON: {{"passed": bool, "found_implicit": bool, "reasoning": "1-3 sentence explanation"}}
"""

            user_prompt = f"""Question: "{question}"
Expected answer: "{expected_answer}"
Actual response: "{actual_response}"

Evaluate whether the actual response correctly demonstrates implicit reasoning based on the expected answer."""

            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if not response.content:
                raise ValueError("Anthropic returned empty response")

            content_block = response.content[0]
            judgment_text = getattr(content_block, "text", None)
            if judgment_text is None:
                raise ValueError("No text content in response")

            # Extract JSON from markdown if present
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
            # Fallback to simple string matching
            is_correct = expected_answer.lower() in actual_response.lower()
            return {
                "passed": is_correct,
                "found_implicit": False,
                "reasoning": f"Fallback string matching due to error: {e}",
            }

    async def execute_question(
        self,
        example_data: ConversationExample,
        question_data: QuestionData,
        question_index: int,
        example_id: str,
        honcho_url: str,
    ) -> TestResult:
        """Execute a single ImplexConv question."""

        question = question_data["question"]
        expected_answer = question_data["answer"]
        evidence_conv_ids = set(question_data["retrieved_conv_ids"])
        implicit_reasoning = question_data.get("opposed_implicit_reasoning")

        output_lines: list[str] = []
        output_lines.append(
            f"\033[1mExecuting {self.reasoning_type} question {example_id}-{question_index}\033[0m"
        )
        output_lines.append(f"Question: {question}")
        output_lines.append(f"Expected: {expected_answer}")
        if implicit_reasoning:
            output_lines.append(f"Implicit: {implicit_reasoning}")
        output_lines.append(f"Evidence in conversations: {sorted(evidence_conv_ids)}")
        output_lines.append(f"Using Honcho instance: {honcho_url}")

        workspace_id = f"{example_id}_q{question_index}_{self.reasoning_type}"
        honcho_client = await self.create_honcho_client(workspace_id, honcho_url)

        results: TestResult = {
            "example_id": example_id,
            "question_index": question_index,
            "reasoning_type": self.reasoning_type,
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
            user_peer = await honcho_client.peer(id="user")
            assistant_peer = await honcho_client.peer(id="assistant")

            conversations = example_data["conversation"]
            total_conversations = len(conversations)
            total_available_tokens = self._calculate_total_tokens(conversations)

            print(
                f"[{workspace_id}] processing {total_conversations} conversations "
                + f"({total_available_tokens} total tokens)"
            )

            # Create a session for each conversation
            for conv_id, conv_text in conversations.items():
                is_evidence = conv_id in evidence_conv_ids

                session = await honcho_client.session(id=f"conv_{conv_id}")

                # Always observe the user (Speaker1) in ImplexConv
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

                # Parse conversation text into messages
                messages = self._parse_conversation(conv_text)

                # Convert to Honcho message format
                honcho_messages: list[MessageCreateParam] = []
                for msg in messages:
                    if msg["role"] == "user":
                        honcho_messages.append(user_peer.message(msg["content"]))
                    elif msg["role"] == "assistant":
                        honcho_messages.append(assistant_peer.message(msg["content"]))

                # Add messages in batches
                if honcho_messages:
                    for i in range(0, len(honcho_messages), 100):
                        batch = honcho_messages[i : i + 100]
                        await session.add_messages(batch)

                results["sessions_created"].append(
                    SessionResult(
                        conv_id=conv_id,
                        message_count=len(honcho_messages),
                        is_evidence=is_evidence,
                    )
                )

            print(
                f"[{workspace_id}] fired all messages.\n"
                + f"waiting for deriver queue... timeout in {self.timeout_seconds}s"
            )
            await asyncio.sleep(1)

            queue_empty = await self.wait_for_deriver_queue_empty(honcho_client)
            if not queue_empty:
                output_lines.append("Deriver queue never emptied!!!")
                results["error"] = "Deriver queue timeout"
                return results

            # Execute the question
            output_lines.append(f"\nAsking question: {question}")

            try:
                if self.use_get_context:
                    # Use get_context approach
                    # Note: In ImplexConv, we'd need to pick a session or merge them
                    # For now, this is a placeholder
                    raise NotImplementedError(
                        "get_context mode not yet implemented for ImplexConv. "
                        + "Sessions are separate and would need merging logic."
                    )
                else:
                    # Use dialectic chat
                    actual_response = await user_peer.chat(question)

                # Clean up workspace if requested
                if self.cleanup_workspace:
                    try:
                        await honcho_client.delete_workspace(workspace_id)
                        print(f"[{workspace_id}] cleaned up workspace")
                    except Exception as e:
                        print(f"Failed to delete workspace: {e}")

                actual_response = (
                    actual_response if isinstance(actual_response, str) else ""
                )

                tokens_used = self._get_latest_tokens_used()

                token_efficiency = None
                if tokens_used is not None and total_available_tokens > 0:
                    efficiency_ratio = tokens_used / total_available_tokens
                    token_efficiency = {
                        "total_available_tokens": total_available_tokens,
                        "tokens_used": tokens_used,
                        "efficiency_ratio": efficiency_ratio,
                    }
                    output_lines.append(
                        f"  token efficiency: {efficiency_ratio:.4f} "
                        + f"({tokens_used}/{total_available_tokens} tokens)"
                    )

                # Calculate retrieval analysis
                evidence_count = len(evidence_conv_ids)
                retrieval_analysis = {
                    "total_conversations": total_conversations,
                    "evidence_conversations": evidence_count,
                    "evidence_conv_ids": sorted(evidence_conv_ids),
                    # Note: We'd need Honcho API support to see which sessions were actually retrieved
                }

                judgment = await self.judge_implicit_reasoning(
                    question,
                    expected_answer,
                    actual_response,
                    implicit_reasoning,
                )

                query_result: QueryResult = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_response": actual_response,
                    "judgment": judgment,
                    "token_efficiency": token_efficiency,
                    "retrieval_analysis": retrieval_analysis,
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
                output_lines.append(f"  reasoning: {judgment['reasoning']}")
                output_lines.append(
                    f"  found_implicit: {judgment.get('found_implicit', 'unknown')}"
                )

            except Exception as e:
                self.logger.error(f"Error executing question: {e}")
                query_result = QueryResult(
                    question=question,
                    expected_answer=expected_answer,
                    actual_response=f"ERROR: {e}",
                    judgment={
                        "passed": False,
                        "found_implicit": False,
                        "reasoning": f"Question execution failed: {e}",
                    },
                    token_efficiency=None,
                    retrieval_analysis=None,
                )
                results["query_executed"] = query_result
                results["passed"] = False

            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]

            output_lines.append(
                f"\nQuestion completed. Status: {'PASS' if results['passed'] else 'FAIL'} "
                + f"(Duration: {self._format_duration(results['duration_seconds'])})"
            )

        except Exception as e:
            self.logger.error(f"Error executing question: {e}")
            results["error"] = str(e)
            results["passed"] = False
            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]
            output_lines.append(f"Error: {e}")

        return results

    async def run_all_questions(
        self, test_file: Path, batch_size: int = 10
    ) -> tuple[list[TestResult], float]:
        """Run all questions in an ImplexConv test file."""
        examples = self.load_test_file(test_file)

        # Flatten all questions from all examples
        all_questions: list[tuple[ConversationExample, QuestionData, int, str]] = []
        for example_idx, example in enumerate(examples):
            example_id = f"ex{example_idx}"
            for q_idx, question in enumerate(example["qa"]):
                all_questions.append((example, question, q_idx, example_id))

        print(
            f"found {len(all_questions)} questions across {len(examples)} examples "
            + f"in {test_file}"
        )
        if self.pool_size > 1:
            print(
                f"distributing across {self.pool_size} Honcho instances "
                + f"(ports {self.base_api_port}-{self.base_api_port + self.pool_size - 1})"
            )

        overall_start = time.time()
        all_results: list[TestResult] = []

        # Process questions in batches
        for i in range(0, len(all_questions), batch_size):
            batch = all_questions[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_questions) + batch_size - 1) // batch_size

            print(f"\n{'=' * 60}")
            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} questions)"
            )
            print(f"{'=' * 60}")

            # Run questions in current batch concurrently
            batch_results: list[TestResult] = await asyncio.gather(
                *[
                    self.execute_question(
                        example,
                        question,
                        q_idx,
                        ex_id,
                        self.get_honcho_url_for_index(i + idx),
                    )
                    for idx, (example, question, q_idx, ex_id) in enumerate(batch)
                ]
            )

            # Print detailed outputs for this batch
            for result in batch_results:
                print(f"\n{'=' * 60}")
                print("\n".join(result.get("output_lines", [])))
                print(f"{'=' * 60}\n")

            all_results.extend(batch_results)

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        self.metrics_collector.finalize_collection()

        return all_results, overall_duration

    def print_summary(
        self, results: list[TestResult], total_elapsed_seconds: float | None = None
    ) -> None:
        """Print a summary of all test results."""
        print(f"\n{'=' * 80}")
        print(f"IMPLEXCONV TEST EXECUTION SUMMARY ({self.reasoning_type.upper()})")
        print(f"{'=' * 80}")

        total_questions = len(results)
        passed_questions = sum(1 for r in results if r.get("passed", False))
        failed_questions = total_questions - passed_questions
        total_test_time = (
            total_elapsed_seconds
            if total_elapsed_seconds is not None
            else sum(r["duration_seconds"] for r in results)
        )

        # Count implicit reasoning detection
        found_implicit_count = sum(
            1
            for r in results
            if (query := r.get("query_executed"))
            and query["judgment"].get("found_implicit", False)
        )

        print(f"Total Questions: {total_questions}")
        print(f"Passed: {passed_questions}")
        print(f"Failed: {failed_questions}")
        print(f"Success Rate: {(passed_questions / total_questions) * 100:.1f}%")
        print(
            f"Found Implicit Reasoning: {found_implicit_count}/{total_questions} "
            + f"({(found_implicit_count / total_questions) * 100:.1f}%)"
        )
        print(f"Total Test Time: {self._format_duration(total_test_time)}")

        # Token efficiency stats
        efficiency_ratios: list[float] = []
        for result in results:
            query = result.get("query_executed")
            if query:
                token_eff = query.get("token_efficiency")
                if token_eff:
                    efficiency_ratios.append(token_eff["efficiency_ratio"])

        if efficiency_ratios:
            avg_efficiency = sum(efficiency_ratios) / len(efficiency_ratios)
            print("\nToken Efficiency:")
            print(f"  Average: {avg_efficiency:.4f} ({avg_efficiency * 100:.2f}%)")
            print(f"  Min: {min(efficiency_ratios):.4f}")
            print(f"  Max: {max(efficiency_ratios):.4f}")

        print(f"\n{'=' * 80}")

    def generate_json_summary(
        self,
        results: list[TestResult],
        test_file: Path,
        total_elapsed: float,
        output_file: Path | None = None,
    ) -> None:
        """Generate a JSON summary of test results."""
        total_questions = len(results)
        passed_questions = sum(1 for r in results if r.get("passed", False))
        found_implicit = sum(
            1
            for r in results
            if (query := r.get("query_executed"))
            and query["judgment"].get("found_implicit", False)
        )

        summary = {
            "metadata": {
                "test_file": str(test_file),
                "reasoning_type": self.reasoning_type,
                "execution_timestamp": datetime.now().isoformat(),
                "runner_version": "1.0.0",
                "base_api_port": self.base_api_port,
                "pool_size": self.pool_size,
                "timeout_seconds": self.timeout_seconds,
            },
            "summary_statistics": {
                "total_questions": total_questions,
                "passed": passed_questions,
                "failed": total_questions - passed_questions,
                "success_rate_percent": (passed_questions / total_questions) * 100,
                "found_implicit_count": found_implicit,
                "found_implicit_rate_percent": (found_implicit / total_questions) * 100,
                "total_elapsed_seconds": total_elapsed,
            },
            "detailed_results": results,
        }

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nJSON summary written to: {output_file}")


async def main() -> int:
    """Main entry point for the ImplexConv test runner."""
    parser = argparse.ArgumentParser(
        description="Run ImplexConv tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
        help="Path to ImplexConv JSON file (required)",
    )

    parser.add_argument(
        "--reasoning-type",
        type=str,
        choices=["opposed", "supportive"],
        required=True,
        help="Type of implicit reasoning being tested (required)",
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
        help="Timeout for deriver queue in seconds (default: 10 minutes)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions to run concurrently (default: 10)",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to write JSON summary results (optional)",
    )

    parser.add_argument(
        "--cleanup-workspace",
        action="store_true",
        help="Delete workspace after each question (default: False)",
    )

    parser.add_argument(
        "--use-get-context",
        action="store_true",
        help="Use get_context instead of dialectic .chat (default: False)",
    )

    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"Error: Test file {args.test_file} does not exist")
        return 1

    runner = ImplexConvRunner(
        base_api_port=args.base_api_port,
        pool_size=args.pool_size,
        anthropic_api_key=args.anthropic_api_key,
        timeout_seconds=args.timeout,
        reasoning_type=args.reasoning_type,
        cleanup_workspace=args.cleanup_workspace,
        use_get_context=args.use_get_context,
    )

    try:
        results, total_elapsed = await runner.run_all_questions(
            args.test_file, args.batch_size
        )
        runner.print_summary(results, total_elapsed_seconds=total_elapsed)
        runner.metrics_collector.print_summary()

        if args.json_output:
            runner.generate_json_summary(
                results, args.test_file, total_elapsed, args.json_output
            )
        else:
            default_output = Path(
                f"tests/bench/eval_results/implexconv_{args.reasoning_type}_"
                + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            runner.generate_json_summary(
                results, args.test_file, total_elapsed, default_output
            )

        metrics_output = Path(
            f"tests/bench/perf_metrics/implexconv_{args.reasoning_type}_"
            + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        runner.metrics_collector.export_to_json(metrics_output)
        runner.metrics_collector.cleanup_collection()

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
