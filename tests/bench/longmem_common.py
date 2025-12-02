"""
Common utilities for LongMemEval test runners.

Shared functionality between the Honcho benchmark and baseline benchmark.
"""

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from anthropic import AsyncAnthropic
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class BaseQueryResult(TypedDict):
    """Base type definition for query execution results."""

    question: str
    expected_answer: str
    actual_response: str
    judgment: dict[str, Any]


class BaseTestResult(TypedDict):
    """Base type definition for test execution results."""

    question_id: str
    question_type: str
    passed: bool
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float
    output_lines: list[str]


def format_duration(total_seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    If the duration is at least one minute, this returns a string in the
    form "XmYYs" with zero-padded seconds. Otherwise, it returns the
    duration in seconds with two decimal places, e.g., "12.34s".

    Args:
        total_seconds: The duration in seconds.

    Returns:
        A formatted duration string.
    """
    minutes = int(total_seconds // 60)
    if minutes > 0:
        seconds_rounded = int(round(total_seconds - minutes * 60))
        if seconds_rounded == 60:
            minutes += 1
            seconds_rounded = 0
        return f"{minutes}m{seconds_rounded:02d}s"
    return f"{total_seconds:.2f}s"


def calculate_total_tokens(haystack_sessions: list[list[dict[str, str]]]) -> int:
    """Calculate total tokens from all messages in all sessions.

    Args:
        haystack_sessions: List of sessions, each containing messages

    Returns:
        Total number of tokens across all messages
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0

    for session_messages in haystack_sessions:
        for msg in session_messages:
            content = msg.get("content", "")
            try:
                total_tokens += len(
                    tokenizer.encode(
                        content,
                        disallowed_special=(
                            tokenizer.special_tokens_set - {"<|endoftext|>"}
                        ),
                    )
                )
            except Exception:
                total_tokens += len(content) // 4
                logger.warning(
                    f"Error tokenizing content. Using rough estimate of {len(content) // 4} tokens"
                )

    return total_tokens


def parse_longmemeval_date(date_str: str) -> datetime:
    """Parse longmemeval date format to datetime.

    Args:
        date_str: Date string in format "YYYY/MM/DD (Day) HH:MM"

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If date format is invalid
    """
    try:
        # Extract the date and time parts, ignoring the day name in parentheses
        # Format: "2023/05/20 (Sat) 02:21"
        parts = date_str.split(") ")
        if len(parts) != 2:
            raise ValueError(f"Invalid date format: {date_str}")

        date_part = parts[0].split(" (")[0]  # "2023/05/20"
        time_part = parts[1]  # "02:21"

        # Combine and parse
        datetime_str = f"{date_part} {time_part}"
        return datetime.strptime(datetime_str, "%Y/%m/%d %H:%M")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse date '{date_str}': {e}") from e


def load_test_file(test_file: Path) -> list[dict[str, Any]]:
    """Load longmemeval test definitions from a JSON file.

    Args:
        test_file: Path to the JSON test file

    Returns:
        List of test question dictionaries
    """
    with open(test_file) as f:
        return json.load(f)


async def judge_response(
    anthropic_client: AsyncAnthropic,
    question: str,
    expected_answer: str,
    actual_response: str,
) -> dict[str, Any]:
    """Use an LLM to judge if the actual response matches the expected answer.

    Args:
        anthropic_client: Anthropic client instance
        question: The question asked
        expected_answer: Expected answer from the test
        actual_response: Actual response from the system under test

    Returns:
        Judgment result with pass/fail and reasoning
    """
    try:
        system_prompt = """
You are an expert judge evaluating AI responses to memory questions. Your task is to determine if an actual response contains the correct answer from long-term memory.

CRITICAL JUDGING PRINCIPLES:
1. SEMANTIC UNDERSTANDING: Focus on whether the actual response conveys the same core factual information as expected, even if expressed differently (i.e. 1 hour is the same as 60 minutes, and if today's date is 2025-12-02, then 'yesterday' and '2025-12-01' are equivalent)
2. FLEXIBLE INTERPRETATION: Accept responses that are longer, more detailed, or use different phrasing as long as they contain the correct answer
3. MEMORY ACCURACY: The key is whether the AI correctly recalled and stated the factual information from memory
4. IMPLICIT vs EXPLICIT: Accept responses that clearly imply the correct answer through context

ONLY FAIL when:
- The core factual answer is demonstrably wrong
- The response shows no evidence of accessing the relevant memory
- The AI explicitly states incorrect information that contradicts the expected answer

Always respond with valid JSON: {"passed": boolean, "reasoning": "short (1-3 sentences) explanation of why the response is correct or incorrect"}"""

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
        return judgment

    except Exception as e:
        logger.error(f"Error judging response: {e}")
        # Fallback to simple string matching
        is_correct = expected_answer.lower() in actual_response.lower()
        return {
            "passed": is_correct,
            "reasoning": f"Fallback string matching due to error: {'Match found' if is_correct else 'No match found'}",
        }


def filter_questions(
    questions: list[dict[str, Any]],
    test_file: Path,
    question_id: str | None = None,
    test_count: int | None = None,
) -> list[dict[str, Any]]:
    """Filter questions by question_id and/or test_count.

    Args:
        questions: List of question dictionaries
        test_file: Path to test file (for logging)
        question_id: Optional question_id to filter to
        test_count: Optional limit on number of questions

    Returns:
        Filtered list of questions
    """
    # Filter by question_id if specified
    if question_id is not None:
        original_count = len(questions)
        questions = [q for q in questions if q.get("question_id") == question_id]
        if not questions:
            print(
                f"Error: No question found with question_id '{question_id}' in {test_file}"
            )
            return []
        print(
            f"filtering to question_id '{question_id}' ({len(questions)}/{original_count} {'question' if len(questions) == 1 else 'questions'})"
        )

    # Limit to first N questions if test_count is specified
    if test_count is not None and test_count > 0:
        questions = questions[:test_count]
        print(
            f"limiting to first {len(questions)} {'question' if len(questions) == 1 else 'questions'} from {test_file}"
        )

    return questions


def calculate_type_statistics(
    results: Sequence[Any],
) -> dict[str, dict[str, int | float]]:
    """Calculate pass/fail statistics grouped by question type.

    Args:
        results: List of test results

    Returns:
        Dictionary mapping question type to statistics
    """
    type_stats: dict[str, dict[str, int | float]] = {}
    for result in results:
        q_type = result["question_type"]
        if q_type not in type_stats:
            type_stats[q_type] = {"total": 0, "passed": 0, "failed": 0}
        type_stats[q_type]["total"] += 1
        if result.get("passed", False):
            type_stats[q_type]["passed"] += 1
        else:
            type_stats[q_type]["failed"] += 1

    # Add success rates
    for q_type in type_stats:
        stats = type_stats[q_type]
        stats["success_rate"] = (
            (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        )

    return type_stats


def calculate_timing_statistics(
    results: Sequence[Any], total_elapsed_seconds: float
) -> dict[str, Any]:
    """Calculate timing statistics from test results.

    Args:
        results: List of test results
        total_elapsed_seconds: Total elapsed time for all tests

    Returns:
        Dictionary of timing statistics
    """
    durations = [r["duration_seconds"] for r in results]
    return {
        "total_duration_seconds": total_elapsed_seconds,
        "individual_test_durations": {
            "min_seconds": min(durations) if durations else 0,
            "max_seconds": max(durations) if durations else 0,
            "mean_seconds": sum(durations) / len(durations) if durations else 0,
            "median_seconds": sorted(durations)[len(durations) // 2]
            if durations
            else 0,
        },
    }


def write_json_summary(summary: dict[str, Any], output_file: Path) -> None:
    """Write a JSON summary to a file.

    Args:
        summary: Summary dictionary to write
        output_file: Path to output file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON summary written to: {output_file}")
