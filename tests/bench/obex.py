"""
Deriver Observation Extraction Benchmark.

Evaluates the deriver's ability to extract observations from messages by comparing
extracted observations against ground truth using embedding similarity.

Test data sources (915 total cases):
- LoCoMo: 543 cases with human-curated observations
- LongMem: 170 cases with Sonnet-generated observations
- BEAM: 202 cases with Sonnet-generated observations

Usage:
    uv run python tests/bench/obex.py                    # Run full evaluation
    uv run python tests/bench/obex.py --limit 10         # Test with limited cases
    uv run python tests/bench/obex.py --source locomo    # Evaluate specific source
    uv run python tests/bench/obex.py --threshold 0.80   # Custom similarity threshold
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings
from src.deriver.non_agent.prompts import minimal_deriver_prompt
from src.embedding_client import EmbeddingClient
from src.utils.clients import honcho_llm_call
from src.utils.representation import PromptRepresentation

CANDIDATES_DIR = Path(__file__).parent / "obexeval_data" / "candidates"
EVAL_RESULTS_DIR = Path(__file__).parent / "eval_results"


# =============================================================================
# Data Classes
# =============================================================================


def f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


@dataclass
class CaseResult:
    """Result for a single test case."""

    case_id: str
    source: str
    difficulty: str
    explicit_precision: float
    explicit_recall: float
    # deductive_precision: float
    # deductive_recall: float
    num_extracted: int
    num_expected: int
    # Actual model output
    extracted_explicit: list[str] = field(default_factory=list)
    # extracted_deductive: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def explicit_f1(self) -> float:
        return f1_score(self.explicit_precision, self.explicit_recall)

    # @property
    # def deductive_f1(self) -> float:
    #     return f1_score(self.deductive_precision, self.deductive_recall)


@dataclass
class SourceStats:
    """Statistics for a source or difficulty grouping."""

    count: int
    explicit_precision: float
    explicit_recall: float
    explicit_f1: float
    # deductive_precision: float
    # deductive_recall: float
    # deductive_f1: float


@dataclass
class AggregateResults:
    """Aggregate results across all cases."""

    total_cases: int
    explicit_precision: float
    explicit_recall: float
    explicit_f1: float
    # deductive_precision: float
    # deductive_recall: float
    # deductive_f1: float
    by_source: dict[str, SourceStats] = field(default_factory=dict)
    by_difficulty: dict[str, SourceStats] = field(default_factory=dict)


# =============================================================================
# Data Loading
# =============================================================================


TestCase = dict[str, Any]


def load_test_cases(source: str | None = None) -> list[TestCase]:
    """Load test cases from candidate files."""
    all_cases: list[TestCase] = []
    for json_file in CANDIDATES_DIR.glob("*_candidates.json"):
        with open(json_file) as f:
            data: dict[str, Any] = json.load(f)
            all_cases.extend(data.get("candidates", []))

    if source:
        all_cases = [tc for tc in all_cases if tc.get("source") == source]

    return all_cases


# =============================================================================
# Deriver Execution
# =============================================================================


def format_messages(messages: list[dict[str, Any]]) -> str:
    """Format messages for the deriver prompt."""
    formatted: list[str] = []
    for msg in messages:
        author: str = msg.get("author", "unknown")
        content: str = msg.get("content", "")
        timestamp: str = msg.get("timestamp", "")

        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = dt.strftime("%Y-%m-%d %H:%M")
                formatted.append(f"[{time_str}] {author}: {content}")
            except ValueError:
                formatted.append(f"{author}: {content}")
        else:
            formatted.append(f"{author}: {content}")

    return "\n".join(formatted)


async def run_deriver(
    messages: list[dict[str, Any]], target_peer: str
) -> PromptRepresentation:
    """Run the deriver on messages and return extracted observations."""
    formatted = format_messages(messages)
    prompt = minimal_deriver_prompt(
        peer_id=target_peer,
        messages=formatted,
    )

    response = await honcho_llm_call(
        llm_settings=settings.DERIVER,
        prompt=prompt,
        max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS or 2000,
        track_name="Deriver Eval",
        response_model=PromptRepresentation,
        json_mode=True,
        stop_seqs=["   \n", "\n\n\n\n"],
        enable_retry=True,
        retry_attempts=3,
    )

    return response.content


# =============================================================================
# Embedding-based Matching
# =============================================================================


class ObservationMatcher:
    """Matches observations using embedding similarity."""

    threshold: float
    client: EmbeddingClient

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.client = EmbeddingClient()
        self._cache: dict[str, list[float]] = {}

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings with caching."""
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            embeddings = await self.client.simple_batch_embed(uncached)
            for text, emb in zip(uncached, embeddings, strict=True):
                self._cache[text] = emb
        return [self._cache[t] for t in texts]

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        a_np, b_np = np.array(a), np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

    async def match(
        self, extracted: list[str], expected: list[str]
    ) -> tuple[float, float]:
        """
        Match extracted to expected using greedy similarity matching.
        Returns: (precision, recall)
        """
        if not extracted and not expected:
            return 1.0, 1.0
        if not extracted:
            return 0.0, 0.0
        if not expected:
            return 0.0, 1.0

        # Get embeddings
        all_texts = extracted + expected
        all_embs = await self._get_embeddings(all_texts)
        ext_embs = all_embs[: len(extracted)]
        exp_embs = all_embs[len(extracted) :]

        # Build similarity matrix
        sim_matrix = np.zeros((len(extracted), len(expected)))
        for i, e1 in enumerate(ext_embs):
            for j, e2 in enumerate(exp_embs):
                sim_matrix[i, j] = self._cosine_sim(e1, e2)

        # Greedy matching
        matches = 0
        used_ext: set[int] = set()
        used_exp: set[int] = set()
        pairs = sorted(
            [
                (sim_matrix[i, j], i, j)
                for i in range(len(extracted))
                for j in range(len(expected))
            ],
            reverse=True,
        )

        for sim, i, j in pairs:
            if sim < self.threshold:
                break
            if i in used_ext or j in used_exp:
                continue
            matches += 1
            used_ext.add(i)
            used_exp.add(j)

        precision = matches / len(extracted)
        recall = matches / len(expected)
        return precision, recall


# =============================================================================
# Evaluation
# =============================================================================


async def evaluate_case(case: TestCase, matcher: ObservationMatcher) -> CaseResult:
    """Evaluate a single test case."""
    case_id: str = case["id"]
    source: str = case.get("source", "unknown")
    difficulty: str = case.get("difficulty", "unknown")
    messages: list[dict[str, Any]] = case.get("messages", [])
    target_peer: str = case.get("target_peer", "user")
    expected: dict[str, Any] = case.get("expected_observations", {})

    exp_explicit: list[str] = [o["content"] for o in expected.get("explicit", [])]
    # exp_deductive: list[str] = [o["conclusion"] for o in expected.get("deductive", [])]

    ext_explicit: list[str]
    # ext_deductive: list[str]
    error: str | None
    try:
        result = await run_deriver(messages, target_peer)
        ext_explicit = [o.content for o in result.explicit]
        # ext_deductive = [o.conclusion for o in result.deductive]
        error = None
    except Exception as e:
        ext_explicit = []  # , ext_deductive = [], []
        error = str(e)

    # Match explicit
    exp_p, exp_r = await matcher.match(ext_explicit, exp_explicit)

    # Match deductive
    # ded_p, ded_r = await matcher.match(ext_deductive, exp_deductive)

    return CaseResult(
        case_id=case_id,
        source=source,
        difficulty=difficulty,
        explicit_precision=exp_p,
        explicit_recall=exp_r,
        # deductive_precision=ded_p,
        # deductive_recall=ded_r,
        num_extracted=len(ext_explicit),  # + len(ext_deductive),
        num_expected=len(exp_explicit),  # + len(exp_deductive),
        extracted_explicit=ext_explicit,
        # extracted_deductive=ext_deductive,
        error=error,
    )


def aggregate_results(results: list[CaseResult]) -> AggregateResults:
    """Compute aggregate statistics."""
    if not results:
        return AggregateResults(0, 0, 0, 0)  # , 0, 0, 0)

    n = len(results)
    exp_p = sum(r.explicit_precision for r in results) / n
    exp_r = sum(r.explicit_recall for r in results) / n
    # ded_p = sum(r.deductive_precision for r in results) / n
    # ded_r = sum(r.deductive_recall for r in results) / n
    agg = AggregateResults(
        total_cases=n,
        explicit_precision=exp_p,
        explicit_recall=exp_r,
        explicit_f1=f1_score(exp_p, exp_r),
        # deductive_precision=ded_p,
        # deductive_recall=ded_r,
        # deductive_f1=f1_score(ded_p, ded_r),
    )

    # By source
    by_source: dict[str, list[CaseResult]] = {}
    for r in results:
        by_source.setdefault(r.source, []).append(r)
    for source, rs in by_source.items():
        src_exp_p = sum(r.explicit_precision for r in rs) / len(rs)
        src_exp_r = sum(r.explicit_recall for r in rs) / len(rs)
        # src_ded_p = sum(r.deductive_precision for r in rs) / len(rs)
        # src_ded_r = sum(r.deductive_recall for r in rs) / len(rs)
        agg.by_source[source] = SourceStats(
            count=len(rs),
            explicit_precision=src_exp_p,
            explicit_recall=src_exp_r,
            explicit_f1=f1_score(src_exp_p, src_exp_r),
            # deductive_precision=src_ded_p,
            # deductive_recall=src_ded_r,
            # deductive_f1=f1_score(src_ded_p, src_ded_r),
        )

    # By difficulty
    by_diff: dict[str, list[CaseResult]] = {}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r)
    for diff, rs in by_diff.items():
        diff_exp_p = sum(r.explicit_precision for r in rs) / len(rs)
        diff_exp_r = sum(r.explicit_recall for r in rs) / len(rs)
        # diff_ded_p = sum(r.deductive_precision for r in rs) / len(rs)
        # diff_ded_r = sum(r.deductive_recall for r in rs) / len(rs)
        agg.by_difficulty[diff] = SourceStats(
            count=len(rs),
            explicit_precision=diff_exp_p,
            explicit_recall=diff_exp_r,
            explicit_f1=f1_score(diff_exp_p, diff_exp_r),
            # deductive_precision=diff_ded_p,
            # deductive_recall=diff_ded_r,
            # deductive_f1=f1_score(diff_ded_p, diff_ded_r),
        )

    return agg


def print_results(agg: AggregateResults, results: list[CaseResult]) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("DERIVER OBSERVATION EXTRACTION EVALUATION")
    print("=" * 70)

    print(f"\nTotal test cases: {agg.total_cases}")

    print(f"\n{'EXPLICIT OBSERVATIONS':^35}")
    print(f"  Precision: {agg.explicit_precision:.3f}")
    print(f"  Recall:    {agg.explicit_recall:.3f}")
    print(f"  F1:        {agg.explicit_f1:.3f}")

    # print(f"\n{'DEDUCTIVE OBSERVATIONS':^35}")
    # print(f"  Precision: {agg.deductive_precision:.3f}")
    # print(f"  Recall:    {agg.deductive_recall:.3f}")
    # print(f"  F1:        {agg.deductive_f1:.3f}")

    if agg.by_source:
        print(f"\n{'BY SOURCE':^35}")
        for source, data in sorted(agg.by_source.items()):
            print(
                f"  {source:12} n={data.count:3}  exp_f1={data.explicit_f1:.3f}"  #  ded_f1={data.deductive_f1:.3f}"
            )

    if agg.by_difficulty:
        print(f"\n{'BY DIFFICULTY':^35}")
        for diff, data in sorted(agg.by_difficulty.items()):
            print(
                f"  {diff:12} n={data.count:3}  exp_f1={data.explicit_f1:.3f}"  #  ded_f1={data.deductive_f1:.3f}"
            )

    # Worst cases (by average of precision and recall)
    worst = sorted(
        results,
        key=lambda r: (
            r.explicit_precision + r.explicit_recall
            # + r.deductive_precision
            # + r.deductive_recall
        )
        / 2,  # / 4,
    )[:5]
    print(f"\n{'WORST PERFORMING CASES':^35}")
    for r in worst:
        avg = (
            (
                r.explicit_precision + r.explicit_recall
                # + r.deductive_precision
                # + r.deductive_recall
            )
            / 2,
        )  # / 4,
        print(f"  {r.case_id[:45]:45} avg={avg:.3f}")

    # Errors
    errors = [r for r in results if r.error]
    if errors:
        print(f"\nErrors: {len(errors)} cases failed")


def generate_json_summary(
    results: list[CaseResult],
    agg: AggregateResults,
    source_filter: str | None,
    threshold: float,
    concurrency: int,
    total_elapsed_seconds: float,
    output_file: Path,
) -> None:
    """
    Generate a comprehensive JSON summary of test results for analytics.

    Args:
        results: List of case results
        agg: Aggregate results
        source_filter: Source filter used (if any)
        threshold: Similarity threshold used
        concurrency: Concurrency level used
        total_elapsed_seconds: Total elapsed time for all tests
        output_file: Path to write JSON output to
    """
    errors = [r for r in results if r.error]

    # Calculate per-case average scores
    case_scores: list[float] = []
    for r in results:
        avg = (r.explicit_precision + r.explicit_recall) / 2
        # + r.deductive_precision
        # + r.deductive_recall
        # / 4,
        case_scores.append(avg)

    output_data: dict[str, Any] = {
        "metadata": {
            "benchmark": "obex",
            "description": "Observation Extraction Benchmark - evaluates deriver's ability to extract observations from messages",
            "execution_timestamp": datetime.now().isoformat(),
            "runner_version": "1.0.0",
            "deriver_settings": settings.DERIVER.model_dump(),
        },
        "config": {
            "source_filter": source_filter,
            "threshold": threshold,
            "concurrency": concurrency,
            "total_cases": len(results),
        },
        "timing": {
            "total_elapsed_seconds": total_elapsed_seconds,
            "average_seconds_per_case": total_elapsed_seconds / len(results)
            if results
            else 0,
        },
        "summary_statistics": {
            "total_cases": agg.total_cases,
            "cases_with_errors": len(errors),
            "explicit_precision": agg.explicit_precision,
            "explicit_recall": agg.explicit_recall,
            "explicit_f1": agg.explicit_f1,
            # "deductive_precision": agg.deductive_precision,
            # "deductive_recall": agg.deductive_recall,
            # "deductive_f1": agg.deductive_f1,
            "mean_case_score": sum(case_scores) / len(case_scores)
            if case_scores
            else 0,
            "min_case_score": min(case_scores) if case_scores else 0,
            "max_case_score": max(case_scores) if case_scores else 0,
        },
        "statistics_by_source": {k: asdict(v) for k, v in agg.by_source.items()},
        "statistics_by_difficulty": {
            k: asdict(v) for k, v in agg.by_difficulty.items()
        },
        "detailed_results": [
            {
                "case_id": r.case_id,
                "source": r.source,
                "difficulty": r.difficulty,
                "explicit_precision": r.explicit_precision,
                "explicit_recall": r.explicit_recall,
                "explicit_f1": r.explicit_f1,
                # "deductive_precision": r.deductive_precision,
                # "deductive_recall": r.deductive_recall,
                # "deductive_f1": r.deductive_f1,
                "num_extracted": r.num_extracted,
                "num_expected": r.num_expected,
                "average_f1": r.explicit_f1,  # (r.explicit_f1 + r.deductive_f1) / 2,
                "model_output": {
                    "explicit": r.extracted_explicit,
                    # "deductive": r.extracted_deductive,
                },
                "error": r.error,
            }
            for r in results
        ],
    }

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_file}")


async def run_evaluation(
    source: str | None = None,
    limit: int | None = None,
    threshold: float = 0.85,
    output: str | None = None,
    concurrency: int = 10,
) -> None:
    """Run the evaluation pipeline."""
    start_time = time.time()

    print("Loading test cases...")
    cases = load_test_cases(source)

    if limit:
        cases = cases[:limit]

    if not cases:
        print("No test cases found!")
        return

    print(
        f"Evaluating {len(cases)} cases (model={settings.DERIVER.MODEL}, threshold={threshold}, concurrency={concurrency})..."
    )

    matcher = ObservationMatcher(threshold=threshold)
    completed = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_with_semaphore(case: TestCase) -> CaseResult:
        nonlocal completed
        async with semaphore:
            result = await evaluate_case(case, matcher)
            completed += 1
            if completed % 10 == 0 or completed == 1:
                print(f"  Completed {completed}/{len(cases)}...")
            return result

    # Run all evaluations concurrently with semaphore limiting
    tasks = [evaluate_with_semaphore(case) for case in cases]
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_elapsed = end_time - start_time

    agg = aggregate_results(list(results))
    print_results(agg, list(results))

    print(
        f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed / len(cases):.2f}s per case)"
    )

    # Determine output file path
    if output:
        output_file = Path(output)
    else:
        # Generate default timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_suffix = f"_{source}" if source else ""
        output_file = EVAL_RESULTS_DIR / f"obex{source_suffix}_{timestamp}.json"

    # Always save results
    generate_json_summary(
        list(results),
        agg,
        source,
        threshold,
        concurrency,
        total_elapsed,
        output_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deriver observation extraction benchmark"
    )
    parser.add_argument(
        "--source", type=str, help="Filter by source (locomo, longmem, beam)"
    )
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold (default: 0.85)",
    )
    parser.add_argument("--output", type=str, help="Output path for results JSON")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent LLM calls (default: 10)",
    )
    args = parser.parse_args()
    asyncio.run(
        run_evaluation(
            args.source, args.limit, args.threshold, args.output, args.concurrency
        )
    )


if __name__ == "__main__":
    main()
