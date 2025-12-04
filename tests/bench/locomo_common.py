"""
Common utilities for LoCoMo benchmark test runners.

Shared functionality between the Honcho benchmark and baseline benchmark.

LoCoMo evaluates very long-term conversational memory of LLM agents across
five question categories:
1. Single-hop - Direct factual recall from conversations
2. Multi-hop - Reasoning across multiple pieces of information
3. Temporal - Understanding time-based relationships and sequences
4. Commonsense/World knowledge - Applying broader contextual understanding
5. Adversarial - Challenging questions that cannot be answered from the conversation

Reference: https://github.com/snap-research/locomo
Paper: https://arxiv.org/abs/2402.17753
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# Category ID to name mapping
CATEGORY_NAMES: dict[int, str] = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "commonsense",
    5: "adversarial",  # Should be filtered out during evaluation
}


class QuestionResult(TypedDict):
    """Type definition for question evaluation results."""

    question_id: int
    question: str
    expected_answer: str
    actual_response: str
    category: int
    category_name: str
    evidence: list[str]
    judgment: dict[str, Any]
    passed: bool


class ConversationResult(TypedDict):
    """Type definition for conversation execution results."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    total_sessions: int
    total_turns: int
    total_tokens: int
    question_results: list[QuestionResult]
    category_scores: dict[str, dict[str, Any]]
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
    tokenizer = tiktoken.get_encoding("o200k_base")
    try:
        return len(
            tokenizer.encode(
                text,
                disallowed_special=(tokenizer.special_tokens_set - {"<|endoftext|>"}),
            )
        )
    except Exception:
        return len(text) // 4


def load_locomo_data(data_file: Path) -> list[dict[str, Any]]:
    """
    Load LoCoMo data from a JSON file.

    Args:
        data_file: Path to the LoCoMo JSON file

    Returns:
        List of conversation dictionaries
    """
    with open(data_file) as f:
        return json.load(f)


def parse_locomo_date(date_str: str) -> datetime:
    """
    Parse LoCoMo date format to datetime.

    Args:
        date_str: Date string in format "H:MM am/pm on D Month, YYYY"
                  e.g., "1:56 pm on 8 May, 2023"

    Returns:
        Parsed datetime object
    """
    try:
        # Handle formats like "1:56 pm on 8 May, 2023"
        # Remove "on " and parse
        date_str = date_str.replace(" on ", " ")
        # Try parsing with different formats
        for fmt in ["%I:%M %p %d %B, %Y", "%I:%M %p %d %B %Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        # If all fail, return a default datetime
        logger.warning(f"Could not parse date '{date_str}', using current time")
        return datetime.now()
    except Exception as e:
        logger.warning(f"Error parsing date '{date_str}': {e}")
        return datetime.now()


def extract_sessions(
    conversation: dict[str, Any],
) -> list[tuple[str, list[dict[str, str]]]]:
    """
    Extract sessions from a LoCoMo conversation.

    Args:
        conversation: The conversation dict containing session_N and session_N_date_time

    Returns:
        List of tuples (date_time_str, messages) where messages have 'speaker' and 'text'
    """
    sessions: list[tuple[str, list[dict[str, str]]]] = []

    # Find all session keys
    session_keys = sorted(
        [k for k in conversation if re.match(r"session_\d+$", k)],
        key=lambda x: int(x.split("_")[1]),
    )

    for session_key in session_keys:
        session_num = session_key.split("_")[1]
        date_key = f"session_{session_num}_date_time"
        date_str = conversation.get(date_key, "")
        messages = conversation.get(session_key, [])
        sessions.append((date_str, messages))

    return sessions


def extract_all_messages(
    conversation: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Extract all messages from all sessions in a conversation.

    Args:
        conversation: The conversation dict

    Returns:
        List of message dicts with 'speaker', 'text', and 'dia_id'
    """
    all_messages: list[dict[str, str]] = []
    sessions = extract_sessions(conversation)

    for _date_str, messages in sessions:
        for msg in messages:
            all_messages.append(
                {
                    "speaker": msg.get("speaker", ""),
                    "text": msg.get("text", ""),
                    "dia_id": msg.get("dia_id", ""),
                }
            )

    return all_messages


def filter_questions(
    qa_list: list[dict[str, Any]],
    exclude_adversarial: bool = False,
    question_ids: list[int] | None = None,
    test_count: int | None = None,
) -> list[dict[str, Any]]:
    """
    Filter questions based on criteria.

    Args:
        qa_list: List of QA dictionaries
        exclude_adversarial: If True, exclude category 5 (adversarial) questions
        question_ids: Optional list of specific question indices to include
        test_count: Optional limit on number of questions

    Returns:
        Filtered list of questions
    """
    filtered = qa_list

    # Filter out adversarial questions (category 5)
    if exclude_adversarial:
        filtered = [q for q in filtered if q.get("category") != 5]

    # Filter by specific question IDs
    if question_ids is not None:
        filtered = [
            q
            for i, q in enumerate(filtered)
            if i in question_ids or (i + 1) in question_ids
        ]

    # Limit to first N questions
    if test_count is not None and test_count > 0:
        filtered = filtered[:test_count]

    return filtered


def _build_judge_prompt(
    category: int,
    question: str,
    answer: str,
    response: str,
) -> str:
    """Build the judge prompt for LoCoMo evaluation.

    Args:
        category: Question category (1-5)
        question: The question asked
        answer: Expected answer from the test
        response: Actual response from the system under test

    Returns:
        The complete prompt for the judge model
    """
    # Adversarial questions (category 5) - should identify as unanswerable
    if category == 5:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the "
            "question as unanswerable. The model could say that the information is "
            "incomplete, or some other information is given but the asked information "
            f"is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\n"
            f"Model Response: {response}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        )

    # Temporal questions (category 3) - allow off-by-one errors
    if category == 3:
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also answer "
            "yes. If the response only contains a subset of the information required by "
            "the answer, answer no.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
            "\n\nIs the model response correct? Answer yes or no only."
        )

    # Default prompt for single-hop, multi-hop, commonsense
    return (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response is equivalent to the correct answer or contains "
        "all the intermediate steps to get the correct answer, you should also answer "
        "yes. If the response only contains a subset of the information required by "
        f"the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}"
        f"\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
    )


async def judge_response(
    openai_client: AsyncOpenAI,
    question: str,
    expected_answer: str,
    actual_response: str,
    category: int = 1,
) -> dict[str, Any]:
    """Use GPT-4o to judge if the actual response matches the expected answer.

    Args:
        openai_client: OpenAI client instance
        question: The question asked
        expected_answer: Expected answer from the test
        actual_response: Actual response from the system under test
        category: Question category (1-5)

    Returns:
        Judgment result with pass/fail and reasoning
    """
    try:
        prompt = _build_judge_prompt(
            category, question, expected_answer, actual_response
        )

        response = await openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            max_tokens=10,
            temperature=0,
            n=1,
            messages=[{"role": "user", "content": prompt}],
        )

        if not response.choices:
            raise ValueError("OpenAI returned empty response")

        eval_response = response.choices[0].message.content
        if eval_response is None:
            raise ValueError("No text content in response")

        # Check if "yes" appears in lowercased response
        passed = "yes" in eval_response.lower()

        return {
            "passed": passed,
            "reasoning": eval_response.strip(),
        }

    except Exception as e:
        logger.error(f"Error judging response: {e}")
        # Fallback to simple string matching
        is_correct = expected_answer.lower() in actual_response.lower()
        return {
            "passed": is_correct,
            "reasoning": f"Fallback string matching due to error: {'Match found' if is_correct else 'No match found'}",
        }


def calculate_category_scores(
    question_results: list[QuestionResult],
) -> dict[str, dict[str, Any]]:
    """
    Calculate scores grouped by question category.

    Args:
        question_results: List of question results

    Returns:
        Dictionary mapping category name to statistics
    """
    category_stats: dict[str, dict[str, Any]] = {}

    for qr in question_results:
        cat_name = qr["category_name"]
        if cat_name not in category_stats:
            category_stats[cat_name] = {
                "total": 0,
                "passed": 0,
            }

        category_stats[cat_name]["total"] += 1
        if qr["passed"]:
            category_stats[cat_name]["passed"] += 1

    # Calculate success rates
    for cat_name in category_stats:
        stats = category_stats[cat_name]
        stats["success_rate"] = (
            (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        )

    return category_stats


def print_summary(
    results: list[ConversationResult], total_elapsed_seconds: float
) -> None:
    """Print a summary of all test results."""
    print(f"\n{'=' * 80}")
    print("LOCOMO BENCHMARK EXECUTION SUMMARY")
    print(f"{'=' * 80}")

    total_conversations = len(results)
    total_questions = sum(len(r["question_results"]) for r in results)

    print(f"Total Conversations: {total_conversations}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Test Time: {format_duration(total_elapsed_seconds)}")

    # Aggregate category scores across all conversations
    category_totals: dict[str, dict[str, Any]] = {}
    for result in results:
        for cat_name, stats in result["category_scores"].items():
            if cat_name not in category_totals:
                category_totals[cat_name] = {
                    "total": 0,
                    "passed": 0,
                }
            category_totals[cat_name]["total"] += stats["total"]
            category_totals[cat_name]["passed"] += stats["passed"]

    print("\nScores by Question Category:")
    print(f"{'Category':<20} {'Total':<8} {'Passed':<8} {'Rate':<10}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10}")

    for cat_name in sorted(category_totals.keys()):
        stats = category_totals[cat_name]
        rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{cat_name:<20} {stats['total']:<8} {stats['passed']:<8} {rate:<10.1f}%")

    # Overall averages
    overall_scores = [r["overall_score"] for r in results]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    print(f"\n{'Overall Average Score':<30}: {overall_avg:.3f}")

    print(f"{'=' * 80}")


def generate_json_summary(
    results: list[ConversationResult],
    total_elapsed_seconds: float,
    output_file: Path,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Generate a comprehensive JSON summary of test results."""
    total_conversations = len(results)
    total_questions = sum(len(r["question_results"]) for r in results)

    # Aggregate category scores
    category_totals: dict[str, dict[str, Any]] = {}
    for result in results:
        for cat_name, stats in result["category_scores"].items():
            if cat_name not in category_totals:
                category_totals[cat_name] = {
                    "total": 0,
                    "passed": 0,
                }
            category_totals[cat_name]["total"] += stats["total"]
            category_totals[cat_name]["passed"] += stats["passed"]

    category_averages = {
        cat: {
            "total": stats["total"],
            "passed": stats["passed"],
            "success_rate": (stats["passed"] / stats["total"]) * 100
            if stats["total"] > 0
            else 0,
        }
        for cat, stats in category_totals.items()
    }

    # Overall averages
    overall_scores = [r["overall_score"] for r in results]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    metadata = {
        "benchmark": "LoCoMo",
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
            "category_statistics": category_averages,
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
