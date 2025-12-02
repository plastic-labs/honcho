"""
Common utilities for BEAM benchmark test runners.

Shared functionality between the Honcho benchmark and baseline benchmark.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import tiktoken
from openai import AsyncOpenAI
from scipy.stats import kendalltau  # pyright: ignore[reportUnknownVariableType]
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


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


def format_duration(total_seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    minutes = int(total_seconds // 60)
    if minutes > 0:
        seconds_rounded = int(round(total_seconds - minutes * 60))
        if seconds_rounded == 60:
            minutes += 1
            seconds_rounded = 0
        return f"{minutes}m{seconds_rounded:02d}s"
    return f"{total_seconds:.2f}s"


def calculate_tokens(text: str) -> int:
    """Calculate tokens for a given text."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    try:
        return len(
            tokenizer.encode(
                text,
                disallowed_special=(tokenizer.special_tokens_set - {"<|endoftext|>"}),
            )
        )
    except Exception:
        return len(text) // 4


def load_conversation(
    data_dir: Path, context_length: str, conversation_id: str
) -> dict[str, Any]:
    """
    Load a BEAM conversation from the data directory.

    Args:
        data_dir: Path to the BEAM data directory
        context_length: Context length (100K, 500K, 1M, 10M)
        conversation_id: Conversation ID

    Returns:
        Dictionary containing conversation data and probing questions
    """
    conv_dir = data_dir / context_length / conversation_id

    # Load chat data
    chat_file = conv_dir / "chat.json"
    with open(chat_file) as f:
        chat_data = json.load(f)

    # Load probing questions
    questions_file = conv_dir / "probing_questions" / "probing_questions.json"
    with open(questions_file) as f:
        questions_data = json.load(f)

    return {"chat": chat_data, "questions": questions_data}


def list_conversations(data_dir: Path, context_length: str) -> list[str]:
    """
    List all conversation IDs for a given context length.

    Args:
        data_dir: Path to the BEAM data directory
        context_length: Context length (100K, 500K, 1M, 10M)

    Returns:
        List of conversation ID strings
    """
    context_dir = data_dir / context_length
    return [
        d.name for d in sorted(context_dir.iterdir()) if d.is_dir() and d.name.isdigit()
    ]


def extract_messages_from_chat_data(chat_data: list[Any]) -> list[dict[str, str]]:
    """
    Extract all messages from BEAM chat data structure.

    Handles both standard structure (100K, 500K, 1M) and 10M plan-based structure.

    Args:
        chat_data: Raw chat data from BEAM JSON

    Returns:
        List of messages with 'role' and 'content' keys
    """
    messages: list[dict[str, str]] = []

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
                            messages.append(
                                {
                                    "role": turn["role"],
                                    "content": turn["content"],
                                }
                            )
        else:
            # Standard structure for 100K, 500K, 1M
            for turn_group in batch.get("turns", []):
                for turn in turn_group:
                    messages.append(
                        {
                            "role": turn["role"],
                            "content": turn["content"],
                        }
                    )

    return messages


async def judge_nugget_based(
    openrouter_client: AsyncOpenAI,
    judge_model: str,
    question: str,
    rubric: list[str],
    actual_response: str,
) -> dict[str, Any]:
    """
    Use an LLM to judge a response using nugget-based evaluation.

    Args:
        openrouter_client: OpenAI-compatible client for API calls
        judge_model: Model ID to use for judging
        question: The question asked
        rubric: List of nuggets (atomic criteria) to check
        actual_response: Actual response to evaluate

    Returns:
        Judgment result with nugget scores and overall score
    """
    try:
        # Build the nugget evaluation prompt
        nuggets_formatted = "\n".join(
            [f"{i + 1}. {nugget}" for i, nugget in enumerate(rubric)]
        )

        system_prompt = """You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERIA.

## EVALUATION RUBRIC:

The rubric defines specific requirements, constraints, or expected behaviors that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether each rubric criterion specifies:

- **Positive requirements** (things the response SHOULD include/do)

- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)

A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.

- If the response does not address the QUESTION, score **0.0** for all criteria and stop.

- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:

Judge by meaning, not exact wording.

- Accept **paraphrases** and **synonyms** that preserve intent.

- **Case/punctuation/whitespace** differences must be ignored.

- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.

- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):

Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").

- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.

- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:

- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.

  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).

  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.

  - Positive: element present but minor inaccuracies/incomplete execution.

  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.

  - Positive: required element missing or incorrect.

  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:

1. **Understand the Requirement**: For each rubric criterion, determine if it is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If a rubric criterion contains multiple elements connected by "and" or commas, evaluate whether:

   - **All elements** must be present for full compliance (1.0)

   - **Some elements** present indicates partial compliance (0.5)

   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**: For each criterion:

   - For positive requirements: Look for the presence and quality of the required element

   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with each specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: For each criterion, explain whether it was satisfied and justify the score.

Use the `evaluate_response` tool to submit your evaluation with scores and reasoning for each rubric criterion."""

        user_prompt = f"""## EVALUATION INPUTS

- QUESTION (what the user asked): {question}

- RUBRIC CRITERIA (what to check):
{nuggets_formatted}

- RESPONSE TO EVALUATE: {actual_response}

Evaluate the response against each rubric criterion. Provide a score and reasoning for each criterion, and calculate the overall score as the average of all criterion scores."""

        tool_definition = {
            "type": "function",
            "function": {
                "name": "evaluate_response",
                "description": "Submit the evaluation results for the response based on the rubric.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nugget_scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "nugget_index": {"type": "integer"},
                                    "score": {"type": "number"},
                                    "reasoning": {"type": "string"},
                                },
                                "required": ["nugget_index", "score", "reasoning"],
                            },
                        },
                        "overall_score": {"type": "number"},
                        "overall_reasoning": {"type": "string"},
                    },
                    "required": [
                        "nugget_scores",
                        "overall_score",
                        "overall_reasoning",
                    ],
                },
            },
        }

        messages = cast(
            list[dict[str, Any]],
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        response = await openrouter_client.chat.completions.create(
            model=judge_model,
            max_tokens=2000,
            temperature=0.0,
            messages=cast(Any, messages),  # type: ignore[arg-type]
            tools=cast(Any, [tool_definition]),  # type: ignore[arg-type]
            tool_choice={
                "type": "function",
                "function": {"name": "evaluate_response"},
            },
        )

        if not response.choices or not response.choices[0].message:
            raise ValueError("OpenRouter returned empty response")

        message = response.choices[0].message

        if not message.tool_calls or len(message.tool_calls) == 0:
            raise ValueError("No tool calls found in response")

        tool_call = message.tool_calls[0]
        # Access function tool call attributes (OpenAI format)
        if tool_call.function.name != "evaluate_response":  # pyright: ignore
            raise ValueError(f"Unexpected tool call: {tool_call.function.name}")  # pyright: ignore

        # Parse the JSON arguments
        judgment = json.loads(tool_call.function.arguments)  # pyright: ignore
        if not isinstance(judgment, dict):
            raise ValueError(f"Tool arguments is not a dictionary: {type(judgment)}")

        return cast(dict[str, Any], judgment)

    except Exception as e:
        logger.error(f"Error judging response: {e}")
        # Fallback to simple 0 score
        return {
            "nugget_scores": [
                {"nugget_index": i + 1, "score": 0.0, "reasoning": f"Error: {e}"}
                for i in range(len(rubric))
            ],
            "overall_score": 0.0,
            "overall_reasoning": f"Evaluation failed due to error: {e}",
        }


def align_events(expected_events: list[str], extracted_events: list[str]) -> list[int]:
    """
    Align extracted events with expected events using string matching.

    Returns a list of indices mapping extracted events to expected events.
    """
    alignment: list[int] = []
    for extracted in extracted_events:
        best_match_idx: int = -1
        for i, expected in enumerate(expected_events):
            # Use simple string matching
            if (
                expected.lower() in extracted.lower()
                or extracted.lower() in expected.lower()
            ):
                best_match_idx = i
                break
        if best_match_idx >= 0:
            alignment.append(best_match_idx)
    return alignment


async def judge_event_ordering(
    openrouter_client: AsyncOpenAI,
    judge_model: str,
    question: str,
    rubric: list[str],
    actual_response: str,
) -> dict[str, Any]:
    """
    Judge event ordering questions using Kendall tau-b coefficient.

    Args:
        openrouter_client: OpenAI-compatible client for API calls
        judge_model: Model ID to use for judging
        question: The question asked
        rubric: List of expected events in correct order
        actual_response: Actual response from the system

    Returns:
        Judgment with Kendall tau-b score
    """
    try:
        # First, extract the events mentioned in the response
        system_prompt = """You are an expert at extracting ordered lists of events or items from text.

Your task is to extract the ordered list of events/items mentioned in a response.

Use the `extract_ordered_events` tool to submit the extracted list."""

        user_prompt = f"""Question: "{question}"

Response: "{actual_response}"

Extract the ordered list of events or items mentioned in the response. Preserve the order as stated in the response."""

        tool_definition = {
            "type": "function",
            "function": {
                "name": "extract_ordered_events",
                "description": "Submit the ordered list of events extracted from the response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "extracted_events": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["extracted_events"],
                },
            },
        }

        messages = cast(
            list[dict[str, Any]],
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        response = await openrouter_client.chat.completions.create(
            model=judge_model,
            max_tokens=1000,
            temperature=0.0,
            messages=cast(Any, messages),  # type: ignore[arg-type]
            tools=cast(Any, [tool_definition]),  # type: ignore[arg-type]
            tool_choice={
                "type": "function",
                "function": {"name": "extract_ordered_events"},
            },
        )

        if not response.choices or not response.choices[0].message:
            raise ValueError("OpenRouter returned empty response")

        message = response.choices[0].message

        if not message.tool_calls or len(message.tool_calls) == 0:
            raise ValueError("No tool calls found in response")

        tool_call = message.tool_calls[0]
        # Access function tool call attributes (OpenAI format)
        if tool_call.function.name != "extract_ordered_events":  # pyright: ignore
            raise ValueError(f"Unexpected tool call: {tool_call.function.name}")  # pyright: ignore

        # Parse the JSON arguments
        extracted_dict = json.loads(tool_call.function.arguments)  # pyright: ignore
        if not isinstance(extracted_dict, dict):
            raise ValueError(
                f"Tool arguments is not a dictionary: {type(extracted_dict)}"
            )

        extracted_dict = cast(dict[str, Any], extracted_dict)
        raw_events: list[str] = extracted_dict.get("extracted_events", [])
        extracted_events = [str(e) for e in raw_events]

        # Now compute alignment and Kendall tau-b
        alignment = align_events(rubric, extracted_events)

        # Compute Kendall tau-b
        tau: float
        if kendalltau is None:
            logger.warning(
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
        logger.error(f"Error in event ordering evaluation: {e}")
        return {
            "kendall_tau_b": 0.0,
            "extracted_events": [],
            "alignment": [],
            "overall_score": 0.0,
            "overall_reasoning": f"Evaluation failed due to error: {e}",
        }


def calculate_ability_scores(
    question_results: list[QuestionResult],
) -> dict[str, float]:
    """Calculate average scores by memory ability."""
    ability_totals: dict[str, list[float]] = {}
    for qr in question_results:
        ability = qr["memory_ability"]
        if ability not in ability_totals:
            ability_totals[ability] = []
        ability_totals[ability].append(qr["score"])

    return {
        ability: sum(scores) / len(scores) for ability, scores in ability_totals.items()
    }


def print_summary(
    results: list[ConversationResult], total_elapsed_seconds: float
) -> None:
    """Print a summary of all test results."""
    print(f"\n{'=' * 80}")
    print("BEAM BENCHMARK EXECUTION SUMMARY")
    print(f"{'=' * 80}")

    total_conversations = len(results)
    total_questions = sum(len(r["question_results"]) for r in results)

    print(f"Total Conversations: {total_conversations}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Test Time: {format_duration(total_elapsed_seconds)}")

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
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    print(f"\n{'Overall Average Score':30s}: {overall_avg:.3f}")

    print(f"{'=' * 80}")


def generate_json_summary(
    results: list[ConversationResult],
    context_length: str,
    total_elapsed_seconds: float,
    output_file: Path,
    metadata_extra: dict[str, Any] | None = None,
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
        ability: sum(scores) / len(scores) for ability, scores in ability_scores.items()
    }

    # Overall average
    overall_scores = [r["overall_score"] for r in results]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    metadata = {
        "context_length": context_length,
        "execution_timestamp": datetime.now().isoformat(),
        "runner_version": "1.0.0",
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    summary = {
        "metadata": metadata,
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
