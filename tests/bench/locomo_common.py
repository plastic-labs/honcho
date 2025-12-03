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
from anthropic import AsyncAnthropic
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
    overall_f1: float
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


def normalize_answer(answer: Any) -> str:
    """
    Normalize an answer for comparison.

    Args:
        answer: The answer (can be string, int, etc.)

    Returns:
        Normalized lowercase string
    """
    if answer is None:
        return ""
    return str(answer).lower().strip()


def compute_f1_score(prediction: str, ground_truth: str) -> dict[str, float]:
    """
    Compute F1 score between prediction and ground truth.

    This uses token-level F1, which is standard for QA evaluation.

    Args:
        prediction: The predicted answer
        ground_truth: The ground truth answer

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    # Normalize and tokenize
    pred_tokens = set(normalize_answer(prediction).split())
    truth_tokens = set(normalize_answer(ground_truth).split())

    # Handle edge cases
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Compute overlap
    common = pred_tokens & truth_tokens

    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


async def judge_response(
    anthropic_client: AsyncAnthropic,
    question: str,
    expected_answer: str,
    actual_response: str,
) -> dict[str, Any]:
    """
    Use an LLM to judge if the actual response matches the expected answer.

    Args:
        anthropic_client: Anthropic client instance
        question: The question asked
        expected_answer: Expected answer from the test
        actual_response: Actual response from the system under test

    Returns:
        Judgment result with pass/fail, reasoning, and F1 score
    """
    # First compute F1 score
    f1_result = compute_f1_score(actual_response, str(expected_answer))

    try:
        system_prompt = """You are an expert judge evaluating AI responses to memory questions about past conversations. Your task is to determine if an actual response contains the correct answer.

CRITICAL JUDGING PRINCIPLES:
1. SEMANTIC UNDERSTANDING: Focus on whether the actual response conveys the same core factual information as expected, even if expressed differently
2. FLEXIBLE INTERPRETATION: Accept responses that are longer, more detailed, or use different phrasing as long as they contain the correct answer
3. MEMORY ACCURACY: The key is whether the AI correctly recalled and stated the factual information from memory
4. IMPLICIT vs EXPLICIT: Accept responses that clearly imply the correct answer through context
5. PARTIAL CREDIT: If the response contains some but not all of the expected information, consider it partially correct

SCORING:
- 1.0 (PASS): Response contains the correct answer or equivalent
- 0.5 (PARTIAL): Response contains some correct information but is incomplete or has minor errors
- 0.0 (FAIL): Response is wrong, doesn't contain the expected information, or is a refusal

Always respond with valid JSON: {"score": number, "passed": boolean, "reasoning": "short explanation"}"""

        user_prompt = f"""Question: "{question}"
Expected answer: "{expected_answer}"
Actual response: "{actual_response}"

Evaluate whether the actual response correctly answers the question based on the expected answer. Focus on factual accuracy and evidence that the AI accessed the correct memory."""

        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
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

        # Add F1 metrics
        judgment["f1"] = f1_result["f1"]
        judgment["precision"] = f1_result["precision"]
        judgment["recall"] = f1_result["recall"]

        return judgment

    except Exception as e:
        logger.error(f"Error judging response: {e}")
        # Fallback to F1-based judgment
        is_correct = f1_result["f1"] >= 0.5
        return {
            "score": f1_result["f1"],
            "passed": is_correct,
            "reasoning": f"Fallback F1-based judgment: F1={f1_result['f1']:.2f}",
            "f1": f1_result["f1"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
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
                "scores": [],
                "f1_scores": [],
            }

        category_stats[cat_name]["total"] += 1
        if qr["passed"]:
            category_stats[cat_name]["passed"] += 1

        judgment = qr.get("judgment", {})
        score = judgment.get("score", 0.0)
        f1 = judgment.get("f1", 0.0)
        category_stats[cat_name]["scores"].append(score)
        category_stats[cat_name]["f1_scores"].append(f1)

    # Calculate averages
    for cat_name in category_stats:
        stats = category_stats[cat_name]
        stats["success_rate"] = (
            (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        )
        stats["avg_score"] = (
            sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        )
        stats["avg_f1"] = (
            sum(stats["f1_scores"]) / len(stats["f1_scores"])
            if stats["f1_scores"]
            else 0
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
                    "scores": [],
                    "f1_scores": [],
                }
            category_totals[cat_name]["total"] += stats["total"]
            category_totals[cat_name]["passed"] += stats["passed"]
            category_totals[cat_name]["scores"].extend(stats.get("scores", []))
            category_totals[cat_name]["f1_scores"].extend(stats.get("f1_scores", []))

    print("\nScores by Question Category:")
    print(f"{'Category':<20} {'Total':<8} {'Passed':<8} {'Rate':<10} {'Avg F1':<10}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10}")

    for cat_name in sorted(category_totals.keys()):
        stats = category_totals[cat_name]
        rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        avg_f1 = (
            sum(stats["f1_scores"]) / len(stats["f1_scores"])
            if stats["f1_scores"]
            else 0
        )
        print(
            f"{cat_name:<20} {stats['total']:<8} {stats['passed']:<8} {rate:<10.1f}% {avg_f1:<10.3f}"
        )

    # Overall averages
    overall_scores = [r["overall_score"] for r in results]
    overall_f1s = [r["overall_f1"] for r in results]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    overall_f1_avg = sum(overall_f1s) / len(overall_f1s) if overall_f1s else 0.0

    print(f"\n{'Overall Average Score':<30}: {overall_avg:.3f}")
    print(f"{'Overall Average F1':<30}: {overall_f1_avg:.3f}")

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
                    "f1_scores": [],
                }
            category_totals[cat_name]["total"] += stats["total"]
            category_totals[cat_name]["passed"] += stats["passed"]
            category_totals[cat_name]["f1_scores"].extend(stats.get("f1_scores", []))

    category_averages = {
        cat: {
            "total": stats["total"],
            "passed": stats["passed"],
            "success_rate": (stats["passed"] / stats["total"]) * 100
            if stats["total"] > 0
            else 0,
            "avg_f1": sum(stats["f1_scores"]) / len(stats["f1_scores"])
            if stats["f1_scores"]
            else 0,
        }
        for cat, stats in category_totals.items()
    }

    # Overall averages
    overall_scores = [r["overall_score"] for r in results]
    overall_f1s = [r["overall_f1"] for r in results]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    overall_f1_avg = sum(overall_f1s) / len(overall_f1s) if overall_f1s else 0.0

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
            "overall_average_f1": overall_f1_avg,
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
