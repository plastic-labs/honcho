"""
Common utilities for explicit derivation benchmark tests.

Shared functionality for testing Honcho's explicit fact extraction capabilities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class Proposition(TypedDict):
    """A single atomic proposition extracted from a message."""

    text: str
    source_quote: str


class ExtractionResult(TypedDict):
    """Result from extracting propositions from a message."""

    propositions: list[Proposition]


class PropositionJudgment(TypedDict):
    """Judgment result for a single proposition."""

    text: str
    is_valid: bool
    quality_rating: str  # "excellent", "good", "fair", "poor", "invalid"
    quality_justification: str  # Explanation for the rating
    error_type: str | None
    error_reason: str | None
    stage_results: dict[str, Any]


class StageMetrics(TypedDict):
    """Metrics for each validation stage."""

    stage_1_total: int  # Total props evaluated in Stage 1
    stage_1_passed: int  # Props that passed Stage 1
    stage_1_pass_rate: float  # Pass rate for Stage 1
    stage_2_total: int  # Total props evaluated in Stage 2
    stage_2_passed: int  # Props that passed Stage 2
    stage_2_pass_rate: float  # Pass rate for Stage 2
    stage_2_escalated: int  # Props escalated to Stage 3
    stage_3_total: int  # Total props evaluated in Stage 3
    stage_3_validated: int  # Props validated by Stage 3
    stage_3_invalidated: int  # Props invalidated by Stage 3
    final_valid: int  # Final count of valid propositions
    final_invalid: int  # Final count of invalid propositions


class JudgeResult(TypedDict):
    """Result from judging a set of extracted propositions."""

    judgments: list[PropositionJudgment]
    precision: float
    coverage: float
    error_breakdown: dict[str, int]
    missing_propositions: list[str]
    stage_metrics: StageMetrics


class ConversationResult(TypedDict):
    """Result from executing explicit benchmark on a conversation."""

    conversation_id: str
    workspace_id: str
    total_messages: int
    extracted_propositions: list[Proposition]
    ground_truth_propositions: list[str]
    judgment: JudgeResult
    matched_extracted_count: int  # Number of unique valid extracted propositions that matched any ground truth
    matched_ground_truth_count: int  # Number of unique ground truth propositions that were matched
    hallucination_rate: float
    stage_metrics: StageMetrics
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


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score (balanced F-score with β=1).

    Args:
        precision: Precision value (0-1)
        recall: Recall value (0-1)

    Returns:
        F1 score (0-1)
    """
    if precision == 0 and recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_f2_score(precision: float, recall: float) -> float:
    """
    Calculate F2 score (recall-weighted F-score with β=2).

    Args:
        precision: Precision value (0-1)
        recall: Recall value (0-1)

    Returns:
        F2 score (0-1)
    """
    if precision == 0 and recall == 0:
        return 0.0
    beta = 2
    beta_squared = beta**2
    return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)


def load_dataset(data_file: Path) -> list[dict[str, Any]]:
    """
    Load explicit derivation test dataset from JSON file.

    Args:
        data_file: Path to dataset JSON file

    Returns:
        List of conversation test cases with messages and ground truth

    Expected format:
    [
        {
            "conversation_id": "conv_001",
            "peer_name": "Sarah",
            "messages": [
                {"text": "...", "timestamp": "2025-06-26T10:30:00Z"}
            ],
            "ground_truth": ["Sarah lives in NYC", "Sarah has a dog"]
        }
    ]
    """
    with open(data_file) as f:
        return json.load(f)


def print_summary(results: list[ConversationResult], total_elapsed_seconds: float) -> None:
    """Print a summary of all test results."""
    print(f"\n{'=' * 80}")
    print("EXPLICIT DERIVATION BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    total_conversations = len(results)
    successful_results = [r for r in results if r["error"] is None]

    print(f"Total Conversations: {total_conversations}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {total_conversations - len(successful_results)}")
    print(f"Total Test Time: {format_duration(total_elapsed_seconds)}")

    if not successful_results:
        print("\nNo successful results to analyze.")
        return

    # Calculate aggregate metrics (aggregated across all propositions)
    total_extracted = sum(len(r["extracted_propositions"]) for r in successful_results)
    total_valid = sum(r["stage_metrics"]["final_valid"] for r in successful_results)
    total_ground_truth = sum(len(r["ground_truth_propositions"]) for r in successful_results)

    # Calculate matched propositions
    # For precision: count of unique valid extracted propositions that matched any ground truth
    # For recall: count of unique ground truth propositions that were matched
    total_matched_extracted = sum(r["matched_extracted_count"] for r in successful_results)
    total_matched_ground_truth = sum(r["matched_ground_truth_count"] for r in successful_results)

    # Precision: Of the valid propositions, what fraction matched ground truth?
    precision = total_matched_extracted / total_valid if total_valid > 0 else 0.0
    # Recall: Of the ground truth propositions, what fraction were successfully extracted?
    recall = total_matched_ground_truth / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = calculate_f1_score(precision, recall)
    f2 = calculate_f2_score(precision, recall)

    # Calculate hallucination rate
    avg_hallucination = sum(r["hallucination_rate"] for r in successful_results) / len(
        successful_results
    )

    # Calculate average stage metrics
    avg_stage_1_pass_rate = sum(r["stage_metrics"]["stage_1_pass_rate"] for r in successful_results) / len(successful_results)
    avg_stage_2_pass_rate = sum(r["stage_metrics"]["stage_2_pass_rate"] for r in successful_results) / len(successful_results)
    avg_stage_3_validation_rate = (
        sum(
            r["stage_metrics"]["stage_3_validated"] / r["stage_metrics"]["stage_3_total"]
            if r["stage_metrics"]["stage_3_total"] > 0
            else 0.0
            for r in successful_results
        )
        / len(successful_results)
    )

    print("\nMetrics:")
    print(f"  Total Extracted:             {total_extracted}")
    print(f"  Total Valid:                 {total_valid}")
    print(f"  Total Ground Truth:          {total_ground_truth}")
    print(f"  Matched (Valid):             {total_matched_extracted}")
    print(f"  Matched (Ground Truth):      {total_matched_ground_truth}")
    print(f"  Precision:                   {precision:.3f}")
    print(f"  Recall:                      {recall:.3f}")
    print(f"  F1 Score:                    {f1:.3f}")
    print(f"  F2 Score:                    {f2:.3f}")
    print(f"  Average Hallucination Rate:  {avg_hallucination:.3f}")

    # Determine which stages ran by checking if any result has non-zero totals
    stage_1_ran = any(r["stage_metrics"]["stage_1_total"] > 0 for r in successful_results)
    stage_2_ran = any(r["stage_metrics"]["stage_2_total"] > 0 for r in successful_results)
    stage_3_ran = any(r["stage_metrics"]["stage_3_total"] > 0 for r in successful_results)

    print("\nAverage Stage Metrics:")
    if stage_1_ran:
        print(f"  Stage 1 Pass Rate (Structural):  {avg_stage_1_pass_rate:.1%}")
    else:
        print(f"  Stage 1 (Structural):            SKIPPED")

    if stage_2_ran:
        print(f"  Stage 2 Pass Rate (NLI):         {avg_stage_2_pass_rate:.1%}")
    else:
        print(f"  Stage 2 (NLI):                   SKIPPED")

    if stage_3_ran:
        print(f"  Stage 3 Validation Rate (LLM):   {avg_stage_3_validation_rate:.1%}")
    else:
        print(f"  Stage 3 (LLM Judge):             SKIPPED")

    # Error breakdown
    all_errors: dict[str, int] = {}
    for result in successful_results:
        for error_type, count in result["judgment"]["error_breakdown"].items():
            all_errors[error_type] = all_errors.get(error_type, 0) + count

    if all_errors:
        print("\nError Breakdown (total across all conversations):")
        for error_type, count in sorted(all_errors.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type:30s}: {count}")

    print(f"{'=' * 80}")


def generate_json_summary(
    results: list[ConversationResult],
    total_elapsed_seconds: float,
    output_file: Path,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Generate a comprehensive JSON summary of test results."""
    successful_results = [r for r in results if r["error"] is None]

    # Calculate summary statistics
    summary_stats: dict[str, Any] = {
        "total_conversations": len(results),
        "successful_conversations": len(successful_results),
        "failed_conversations": len(results) - len(successful_results),
    }

    if successful_results:
        summary_stats.update(
            {
                "average_hallucination_rate": sum(
                    r["hallucination_rate"] for r in successful_results
                )
                / len(successful_results),
            }
        )

        # Calculate aggregate metrics (aggregated across all propositions)
        total_extracted = sum(len(r["extracted_propositions"]) for r in successful_results)
        total_valid = sum(r["stage_metrics"]["final_valid"] for r in successful_results)
        total_ground_truth = sum(len(r["ground_truth_propositions"]) for r in successful_results)
        total_matched_extracted = sum(r["matched_extracted_count"] for r in successful_results)
        total_matched_ground_truth = sum(r["matched_ground_truth_count"] for r in successful_results)

        precision = total_matched_extracted / total_valid if total_valid > 0 else 0.0
        recall = total_matched_ground_truth / total_ground_truth if total_ground_truth > 0 else 0.0
        f1 = calculate_f1_score(precision, recall)
        f2 = calculate_f2_score(precision, recall)

        summary_stats["metrics"] = {
            "total_extracted": total_extracted,
            "total_valid": total_valid,
            "total_ground_truth": total_ground_truth,
            "matched_extracted": total_matched_extracted,
            "matched_ground_truth": total_matched_ground_truth,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f2_score": f2,
        }

        # Calculate average stage metrics
        avg_stage_metrics = {
            "stage_1_pass_rate": sum(r["stage_metrics"]["stage_1_pass_rate"] for r in successful_results) / len(successful_results),
            "stage_2_pass_rate": sum(r["stage_metrics"]["stage_2_pass_rate"] for r in successful_results) / len(successful_results),
            "stage_3_validation_rate": sum(
                r["stage_metrics"]["stage_3_validated"] / r["stage_metrics"]["stage_3_total"]
                if r["stage_metrics"]["stage_3_total"] > 0
                else 0.0
                for r in successful_results
            ) / len(successful_results),
            "total_propositions": sum(r["stage_metrics"]["stage_1_total"] for r in successful_results),
            "total_stage_1_passed": sum(r["stage_metrics"]["stage_1_passed"] for r in successful_results),
            "total_stage_2_passed": sum(r["stage_metrics"]["stage_2_passed"] for r in successful_results),
            "total_stage_2_escalated": sum(r["stage_metrics"]["stage_2_escalated"] for r in successful_results),
            "total_stage_3_validated": sum(r["stage_metrics"]["stage_3_validated"] for r in successful_results),
            "total_stage_3_invalidated": sum(r["stage_metrics"]["stage_3_invalidated"] for r in successful_results),
            "total_final_valid": sum(r["stage_metrics"]["final_valid"] for r in successful_results),
            "total_final_invalid": sum(r["stage_metrics"]["final_invalid"] for r in successful_results),
        }
        summary_stats["average_stage_metrics"] = avg_stage_metrics

        # Aggregate error breakdown
        all_errors: dict[str, int] = {}
        for result in successful_results:
            for error_type, count in result["judgment"]["error_breakdown"].items():
                all_errors[error_type] = all_errors.get(error_type, 0) + count
        summary_stats["total_error_breakdown"] = all_errors

    metadata = {
        "execution_timestamp": datetime.now().isoformat(),
        "runner_version": "1.0.0",
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    summary = {
        "metadata": metadata,
        "summary_statistics": summary_stats,
        "timing": {
            "total_duration_seconds": total_elapsed_seconds,
        },
        "detailed_results": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON summary written to: {output_file}")
