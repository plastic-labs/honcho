"""
Common utilities for RefusalBench benchmark test runners.

Shared functionality between the Honcho benchmark and baseline benchmark.
Based on the RefusalBench methodology (arXiv:2510.10390).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# Valid refusal codes from the RefusalBench paper
VALID_REFUSAL_CODES: list[str] = [
    "REFUSE_AMBIGUOUS_QUERY",
    "REFUSE_CONTRADICTORY_CONTEXT",
    "REFUSE_INFO_MISSING_IN_CONTEXT",
    "REFUSE_FALSE_PREMISE_IN_QUERY",
    "REFUSE_GRANULARITY_MISMATCH",
    "REFUSE_NONFACTUAL_QUERY",
]

# Perturbation classes and their corresponding refusal codes
PERTURBATION_CLASSES: dict[str, str] = {
    "P-Ambiguity": "REFUSE_AMBIGUOUS_QUERY",
    "P-Contradiction": "REFUSE_CONTRADICTORY_CONTEXT",
    "P-MissingInfo": "REFUSE_INFO_MISSING_IN_CONTEXT",
    "P-FalsePremise": "REFUSE_FALSE_PREMISE_IN_QUERY",
    "P-GranularityMismatch": "REFUSE_GRANULARITY_MISMATCH",
    "P-EpistemicMismatch": "REFUSE_NONFACTUAL_QUERY",
}

INTENSITY_LEVELS: list[str] = ["LOW", "MEDIUM", "HIGH"]


class RefusalBenchEntry(TypedDict):
    """Type definition for a single RefusalBench dataset entry."""

    unique_id: str
    perturbation_class: str
    intensity: str
    original_query: str
    perturbed_query: str
    original_context: str
    perturbed_context: str
    original_answers: list[str]
    expected_rag_behavior: str


class ItemResult(TypedDict):
    """Type definition for a single item evaluation result."""

    unique_id: str
    perturbation_class: str
    intensity: str
    expected_behavior: str
    predicted_type: str
    answer_quality_score: float | None
    refusal_match_correct: bool
    is_correct: bool
    judge_explanation: str
    actual_response: str
    query: str
    workspace_id: str
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float


def load_refusalbench_data(data_file: str | Path) -> list[RefusalBenchEntry]:
    """
    Load RefusalBench dataset from a JSONL file.

    Args:
        data_file: Path to the JSONL dataset file.

    Returns:
        List of RefusalBenchEntry dictionaries.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If any entry is missing required fields.
    """
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"RefusalBench data file not found: {data_path}")

    required_fields = {
        "unique_id",
        "perturbation_class",
        "intensity",
        "perturbed_query",
        "perturbed_context",
        "original_answers",
        "expected_rag_behavior",
    }

    entries: list[RefusalBenchEntry] = []
    with open(data_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at line %d: %s", line_num, e)
                continue

            missing = required_fields - set(entry.keys())
            if missing:
                raise ValueError(
                    f"Entry at line {line_num} missing required fields: {missing}"
                )

            entries.append(cast(RefusalBenchEntry, entry))

    return entries


async def judge_refusal_response(
    client: AsyncOpenAI,
    model: str,
    query: str,
    response: str,
    correct_answers: list[str],
) -> tuple[str, float | None, str]:
    """
    Use an LLM judge to classify a response as answering or refusing.

    The judge detects natural language refusals (hedging, uncertainty, explicit
    refusal) from dialectic responses and maps them to REFUSE_* codes.

    Args:
        client: OpenAI-compatible client for API calls.
        model: Model ID to use for judging.
        query: The question that was asked.
        response: The system's actual response.
        correct_answers: List of correct answers for the question.

    Returns:
        Tuple of (predicted_type, answer_quality_score, explanation).
        predicted_type is "ANSWER_CORRECTLY" or one of the REFUSE_* codes.
        answer_quality_score is 0-5 for answers, None for refusals.
    """
    try:
        system_prompt = """You are an expert evaluator for a memory system benchmark called RefusalBench.

Your task is to analyze a system's response to determine whether it:
1. ANSWERED the question (attempted to provide information), or
2. REFUSED to answer (indicated inability, uncertainty, or inappropriateness)

## Refusal Detection

A response counts as a REFUSAL if the system:
- Explicitly states it cannot answer or doesn't have enough information
- Hedges heavily with phrases like "I'm not sure", "the context doesn't specify", "there's conflicting information"
- Points out problems with the question (false premise, ambiguity, missing info)
- Declines to speculate or provide opinion on non-factual questions
- Notes contradictions in the provided context
- Says the requested level of detail is not available

A response counts as an ANSWER if the system:
- Provides a direct response to the question
- Gives information even if partially incorrect
- Attempts to answer despite acknowledging limitations

## Refusal Classification

If the response is a refusal, classify it as one of:
- REFUSE_AMBIGUOUS_QUERY: The system noted the question is ambiguous or could refer to multiple things
- REFUSE_CONTRADICTORY_CONTEXT: The system noted conflicting or contradictory information
- REFUSE_INFO_MISSING_IN_CONTEXT: The system noted the needed information is not available
- REFUSE_FALSE_PREMISE_IN_QUERY: The system noted the question contains a false assumption
- REFUSE_GRANULARITY_MISMATCH: The system noted it cannot provide the level of detail requested
- REFUSE_NONFACTUAL_QUERY: The system noted the question asks for opinion, speculation, or prediction

## Answer Quality Scoring (only if answered)

If the system answered, score the quality from 0-5:
- 5: Perfect answer matching the correct answers
- 4: Good answer with minor omissions
- 3: Partially correct answer
- 2: Answer with significant errors
- 1: Mostly incorrect answer
- 0: Completely wrong answer

Use the `classify_response` tool to submit your evaluation."""

        answers_formatted = "\n".join(
            [f"- {ans}" for ans in correct_answers]
        )

        user_prompt = f"""## Question Asked
{query}

## System's Response
{response}

## Correct Answers (for reference)
{answers_formatted}

Analyze the system's response. Determine if it answered or refused, classify accordingly, and provide an explanation."""

        tool_definition = {
            "type": "function",
            "function": {
                "name": "classify_response",
                "description": "Submit the classification of the system's response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response_type": {
                            "type": "string",
                            "enum": [
                                "ANSWER_CORRECTLY",
                                *VALID_REFUSAL_CODES,
                            ],
                            "description": "Whether the system answered or which refusal type it used.",
                        },
                        "answer_quality_score": {
                            "type": "number",
                            "description": "Quality score 0-5 if the system answered. Omit or set to null if refused.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation of the classification decision.",
                        },
                    },
                    "required": ["response_type", "explanation"],
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

        api_response = await client.chat.completions.create(
            model=model,
            max_tokens=1000,
            temperature=0.0,
            messages=cast(Any, messages),  # type: ignore[arg-type]
            tools=cast(Any, [tool_definition]),  # type: ignore[arg-type]
            tool_choice={
                "type": "function",
                "function": {"name": "classify_response"},
            },
        )

        if not api_response.choices or not api_response.choices[0].message:
            raise ValueError("Judge returned empty response")

        message = api_response.choices[0].message

        if not message.tool_calls or len(message.tool_calls) == 0:
            raise ValueError("No tool calls found in judge response")

        tool_call = message.tool_calls[0]
        if tool_call.function.name != "classify_response":  # pyright: ignore
            raise ValueError(f"Unexpected tool call: {tool_call.function.name}")  # pyright: ignore

        judgment = json.loads(tool_call.function.arguments)  # pyright: ignore
        if not isinstance(judgment, dict):
            raise ValueError(f"Tool arguments is not a dictionary: {type(judgment)}")

        judgment = cast(dict[str, Any], judgment)
        predicted_type: str = judgment.get("response_type", "ANSWER_CORRECTLY")
        quality_score_raw = judgment.get("answer_quality_score")
        quality_score: float | None = None
        if quality_score_raw is not None and quality_score_raw != "null":
            try:
                quality_score = float(quality_score_raw)
            except (TypeError, ValueError):
                quality_score = None
        explanation: str = judgment.get("explanation", "")

        return predicted_type, quality_score, explanation

    except Exception as e:
        logger.exception("Error in judge_refusal_response: %s", e)
        return "ANSWER_CORRECTLY", 0.0, f"Judge error: {e}"


def evaluate_single_item(
    entry: RefusalBenchEntry,
    predicted_type: str,
    quality_score: float | None,
    explanation: str,  # pyright: ignore[reportUnusedParameter]
) -> bool:
    """
    Evaluate whether the prediction is correct for a single item.

    Args:
        entry: The dataset entry with expected behavior.
        predicted_type: The judge's predicted behavior type.
        quality_score: Answer quality score (0-5) if answered, None if refused.
        explanation: Judge's explanation.

    Returns:
        True if the prediction matches the expected behavior.
    """
    expected = entry["expected_rag_behavior"]

    if expected == "ANSWER_CORRECTLY":
        # For answerable items: correct if the system answered AND quality >= 4
        if predicted_type == "ANSWER_CORRECTLY":
            return quality_score is not None and quality_score >= 4.0
        return False
    else:
        # For unanswerable items: correct if the system refused with the right code
        return predicted_type == expected


def compute_metrics(results: list[ItemResult]) -> dict[str, Any]:
    """
    Compute overall metrics from evaluation results.

    Args:
        results: List of per-item evaluation results.

    Returns:
        Dictionary with accuracy, false_refusal_rate, refusal_accuracy,
        missed_refusal_rate, and calibrated_refusal_score (CRS).
    """
    if not results:
        return {
            "total": 0,
            "accuracy": 0.0,
            "false_refusal_rate": 0.0,
            "refusal_accuracy": 0.0,
            "missed_refusal_rate": 0.0,
            "calibrated_refusal_score": 0.0,
        }

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])

    # Answerable items (expected ANSWER_CORRECTLY)
    answerable = [r for r in results if r["expected_behavior"] == "ANSWER_CORRECTLY"]
    answerable_total = len(answerable)
    false_refusals = sum(
        1 for r in answerable if r["predicted_type"] != "ANSWER_CORRECTLY"
    )

    # Unanswerable items (expected REFUSE_*)
    unanswerable = [r for r in results if r["expected_behavior"] != "ANSWER_CORRECTLY"]
    unanswerable_total = len(unanswerable)
    correct_refusals = sum(1 for r in unanswerable if r["refusal_match_correct"])
    missed_refusals = sum(
        1 for r in unanswerable if r["predicted_type"] == "ANSWER_CORRECTLY"
    )

    accuracy = correct / total if total > 0 else 0.0
    false_refusal_rate = (
        false_refusals / answerable_total if answerable_total > 0 else 0.0
    )
    refusal_accuracy = (
        correct_refusals / unanswerable_total if unanswerable_total > 0 else 0.0
    )
    missed_refusal_rate = (
        missed_refusals / unanswerable_total if unanswerable_total > 0 else 0.0
    )

    # CRS = 0.5 * accuracy + 0.5 * refusal_accuracy
    calibrated_refusal_score = 0.5 * accuracy + 0.5 * refusal_accuracy

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "answerable_total": answerable_total,
        "false_refusals": false_refusals,
        "false_refusal_rate": false_refusal_rate,
        "unanswerable_total": unanswerable_total,
        "correct_refusals": correct_refusals,
        "missed_refusals": missed_refusals,
        "refusal_accuracy": refusal_accuracy,
        "missed_refusal_rate": missed_refusal_rate,
        "calibrated_refusal_score": calibrated_refusal_score,
    }


def compute_metrics_by_class(
    results: list[ItemResult],
) -> dict[str, dict[str, Any]]:
    """
    Compute metrics broken down by perturbation class.

    Args:
        results: List of per-item evaluation results.

    Returns:
        Dictionary mapping perturbation class to its metrics.
    """
    by_class: dict[str, list[ItemResult]] = {}
    for r in results:
        cls = r["perturbation_class"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(r)

    return {cls: compute_metrics(items) for cls, items in sorted(by_class.items())}


def compute_metrics_by_intensity(
    results: list[ItemResult],
) -> dict[str, dict[str, Any]]:
    """
    Compute metrics broken down by intensity level.

    Args:
        results: List of per-item evaluation results.

    Returns:
        Dictionary mapping intensity level to its metrics.
    """
    by_intensity: dict[str, list[ItemResult]] = {}
    for r in results:
        intensity = r["intensity"]
        if intensity not in by_intensity:
            by_intensity[intensity] = []
        by_intensity[intensity].append(r)

    return {
        level: compute_metrics(items) for level, items in sorted(by_intensity.items())
    }


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


def print_summary(results: list[ItemResult], total_duration: float) -> None:
    """Print a formatted summary of RefusalBench results."""
    print(f"\n{'=' * 80}")
    print("REFUSALBENCH EXECUTION SUMMARY")
    print(f"{'=' * 80}")

    metrics = compute_metrics(results)

    print(f"Total Items: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
    print(f"Calibrated Refusal Score (CRS): {metrics['calibrated_refusal_score']:.3f}")
    print(f"Total Time: {format_duration(total_duration)}")

    print(f"\n{'─' * 40}")
    print("Answerable Items:")
    print(f"  Total: {metrics['answerable_total']}")
    print(f"  False Refusal Rate: {metrics['false_refusal_rate']:.3f}")

    print(f"\nUnanswerable Items:")
    print(f"  Total: {metrics['unanswerable_total']}")
    print(f"  Refusal Accuracy: {metrics['refusal_accuracy']:.3f}")
    print(f"  Missed Refusal Rate: {metrics['missed_refusal_rate']:.3f}")

    # By perturbation class
    by_class = compute_metrics_by_class(results)
    if by_class:
        print(f"\n{'─' * 40}")
        print("By Perturbation Class:")
        for cls, cls_metrics in by_class.items():
            acc = cls_metrics["accuracy"]
            total = cls_metrics["total"]
            print(f"  {cls:30s}: {acc:.3f} ({total} items)")

    # By intensity
    by_intensity = compute_metrics_by_intensity(results)
    if by_intensity:
        print(f"\n{'─' * 40}")
        print("By Intensity Level:")
        for level, level_metrics in by_intensity.items():
            acc = level_metrics["accuracy"]
            total = level_metrics["total"]
            print(f"  {level:30s}: {acc:.3f} ({total} items)")

    print(f"{'=' * 80}")


def generate_json_summary(
    results: list[ItemResult],
    total_duration: float,
    output_file: Path,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """
    Generate a comprehensive JSON summary of RefusalBench results.

    Args:
        results: List of per-item evaluation results.
        total_duration: Total execution time in seconds.
        output_file: Path to write the JSON summary.
        metadata_extra: Optional additional metadata to include.
    """
    metrics = compute_metrics(results)
    by_class = compute_metrics_by_class(results)
    by_intensity = compute_metrics_by_intensity(results)

    metadata: dict[str, Any] = {
        "benchmark": "refusalbench",
        "execution_timestamp": datetime.now().isoformat(),
        "runner_version": "1.0.0",
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    summary = {
        "metadata": metadata,
        "summary_statistics": metrics,
        "by_perturbation_class": by_class,
        "by_intensity": by_intensity,
        "timing": {
            "total_duration_seconds": total_duration,
        },
        "detailed_results": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON summary written to: {output_file}")
