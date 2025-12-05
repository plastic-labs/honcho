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
from openai import AsyncOpenAI
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
    tokenizer = tiktoken.get_encoding("o200k_base")
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


def _build_judge_prompt(
    question_type: str,
    question: str,
    answer: str,
    response: str,
    question_id: str,
) -> str:
    """Build the judge prompt matching the official LongMemEval evaluation code.

    Based on get_anscheck_prompt() from the official LongMemEval repository.

    Args:
        question_type: Type of question being evaluated
        question: The question asked
        answer: Expected answer from the test
        response: Actual response from the system under test
        question_id: Question ID (used to detect abstention questions)

    Returns:
        The complete prompt for the judge model
    """
    # Check for abstention questions (have '_abs' in question_id)
    if "_abs" in question_id:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the "
            "question as unanswerable. The model could say that the information is "
            "incomplete, or some other information is given but the asked information "
            f"is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\n"
            f"Model Response: {response}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        )

    # Standard prompts by question type
    if question_type in (
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    ):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also answer "
            "yes. If the response only contains a subset of the information required by "
            f"the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}"
            f"\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also answer "
            "yes. If the response only contains a subset of the information required by "
            "the answer, answer no. In addition, do not penalize off-by-one errors for "
            "the number of days. If the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
            "the answer is 18), the model's response is still correct. \n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
            "\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response contains some previous information along with an "
            "updated answer, the response should be considered as correct as long as the "
            f"updated answer is the required answer.\n\nQuestion: {question}\n\n"
            f"Correct Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, "
            "and a response from a model. Please answer yes if the response satisfies "
            "the desired response. Otherwise, answer no. The model does not need to "
            "reflect all the points in the rubric. The response is correct as long as "
            "it recalls and utilizes the user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}"
            "\n\nIs the model response correct? Answer yes or no only."
        )
    else:
        # Default case (same as multi-session)
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
    question_type: str = "default",
    question_id: str = "",
) -> dict[str, Any]:
    """Use GPT-4o to judge if the actual response matches the expected answer.

    Uses the exact prompt format from the official LongMemEval evaluation code
    (evaluate_qa.py) to ensure consistent evaluation.

    Args:
        openai_client: OpenAI client instance
        question: The question asked
        expected_answer: Expected answer from the test
        actual_response: Actual response from the system under test
        question_type: Type of question (temporal-reasoning, knowledge-update,
                       single-session-preference, single-session-user,
                       single-session-assistant, multi-session)
        question_id: Question ID (used to detect abstention questions with '_abs')

    Returns:
        Judgment result with pass/fail and reasoning
    """
    try:
        prompt = _build_judge_prompt(
            question_type, question, expected_answer, actual_response, question_id
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

        # Match official evaluation: check if "yes" appears in lowercased response
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
