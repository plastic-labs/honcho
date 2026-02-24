"""
Honcho OOLONG benchmark runner.

Evaluates long-context reasoning and aggregation on:
- OOLONG-synth: synthetic ICL aggregation tasks
- OOLONG-real: D&D transcript aggregation tasks
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from dotenv import load_dotenv
from honcho.api_types import MessageCreateParams
from honcho.session import Session, SessionPeerConfig

from src.config import settings

from .oolong_common import (
    calculate_context_length,
    calculate_task_statistics,
    calculate_timing_statistics,
    format_duration,
    load_oolong_real_dataset,
    load_oolong_synth_dataset,
    parse_real_answer,
    parse_real_context_messages,
    parse_synth_answer,
    parse_synth_context_messages,
    score_real_response,
    score_synth_response,
    write_json_summary,
)
from .runner_common import (
    BaseRunner,
    ItemContext,
    RunnerConfig,
    add_common_arguments,
    validate_common_arguments,
)

load_dotenv()

CONTEXT_SIZE_MAP: dict[str, int] = {
    "1K": 1024,
    "2K": 2 * 1024,
    "4K": 4 * 1024,
    "8K": 8 * 1024,
    "16K": 16 * 1024,
    "32K": 32 * 1024,
    "64K": 64 * 1024,
    "128K": 128 * 1024,
    "256K": 256 * 1024,
    "512K": 512 * 1024,
    "1M": 1024 * 1024,
    "2M": 2 * 1024 * 1024,
    "4M": 4 * 1024 * 1024,
}


def parse_context_size(size_str: str) -> int:
    """Parse a context-size string into an exact token count."""
    normalized = size_str.strip().upper()

    if normalized.isdigit():
        value = int(normalized)
        if value <= 0:
            raise ValueError("Context size must be positive")
        return value

    if normalized in CONTEXT_SIZE_MAP:
        return CONTEXT_SIZE_MAP[normalized]

    valid_sizes = ", ".join(CONTEXT_SIZE_MAP)
    raise ValueError(
        f"Invalid context size '{size_str}'. Use one of [{valid_sizes}] or a positive integer token count."
    )


class QueryResult(TypedDict):
    """Query execution result for one OOLONG example."""

    question: str
    expected_answer: str
    actual_response: str
    score: float
    context_length_tokens: int


class TestResult(TypedDict):
    """Single OOLONG example result."""

    question_id: str
    context_window_id: str
    task_group: str
    dataset: str
    answer_type: str
    passed: bool
    score: float
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float
    query_executed: QueryResult | None
    output_lines: list[str]


class OolongRunner(BaseRunner[TestResult]):
    """Execute OOLONG benchmark examples through the shared runner framework."""

    variant: str
    data_dir: Path
    split: str
    merge_sessions: bool
    max_examples: int | None
    min_context_len: int | None
    max_context_len: int | None
    context_window_id: str | None
    use_labels: bool

    def __init__(
        self,
        config: RunnerConfig,
        variant: str,
        data_dir: Path,
        split: str,
        merge_sessions: bool,
        max_examples: int | None = None,
        min_context_len: int | None = None,
        max_context_len: int | None = None,
        context_window_id: str | None = None,
        use_labels: bool = False,
    ):
        self.variant = variant
        self.data_dir = data_dir
        self.split = split
        self.merge_sessions = merge_sessions
        self.max_examples = max_examples
        self.min_context_len = min_context_len
        self.max_context_len = max_context_len
        self.context_window_id = context_window_id
        self.use_labels = use_labels
        super().__init__(config)

    def get_metrics_prefix(self) -> str:
        return "oolong"

    def load_items(self) -> list[Any]:
        if self.variant == "synth":
            dataset = load_oolong_synth_dataset(
                split=self.split,
                data_dir=self.data_dir,
                max_context_len=self.max_context_len,
                min_context_len=self.min_context_len,
                max_examples=self.max_examples,
                context_window_id=self.context_window_id,
            )
        else:
            dataset = load_oolong_real_dataset(
                split=self.split,
                data_dir=self.data_dir,
                max_context_len=self.max_context_len,
                min_context_len=self.min_context_len,
                max_examples=self.max_examples,
                context_window_id=self.context_window_id,
            )
        return [dataset[i] for i in range(len(dataset))]

    def get_workspace_id(self, item: Any) -> str:
        return f"oolong_{self.variant}_{item['id']}"

    def get_session_id(self, item: Any, workspace_id: str) -> str:
        return f"{workspace_id}_session"

    async def setup_peers(self, ctx: ItemContext, item: Any) -> None:
        ctx.peers["user"] = await ctx.honcho_client.aio.peer(id="user")

    async def setup_session(self, ctx: ItemContext, item: Any) -> None:
        if not self.merge_sessions:
            ctx.session = None
            return

        user_peer = ctx.peers["user"]
        ctx.session = await ctx.honcho_client.aio.session(
            id=ctx.session_id, configuration=self._get_session_configuration()
        )
        await ctx.session.aio.add_peers(
            [(user_peer, SessionPeerConfig(observe_me=True, observe_others=False))]
        )

    async def _add_messages_to_session(
        self, session: Session, user_peer: Any, messages: list[dict[str, Any]]
    ) -> None:
        honcho_messages: list[MessageCreateParams] = []
        for msg in messages:
            honcho_messages.append(
                user_peer.message(
                    content=msg["content"],
                    metadata=msg.get("metadata"),
                )
            )

        for i in range(0, len(honcho_messages), 100):
            batch = honcho_messages[i : i + 100]
            await session.aio.add_messages(batch)

    async def ingest_messages(self, ctx: ItemContext, item: Any) -> int:
        context_text = item["context_window_text"]
        if self.variant == "synth":
            if self.use_labels:
                context_text = item.get("context_window_text_with_labels", context_text)
            messages = parse_synth_context_messages(context_text)
        else:
            messages = parse_real_context_messages(context_text)

        user_peer = ctx.peers["user"]

        if self.merge_sessions:
            if ctx.session is None:
                raise ValueError("Merged mode requires a configured session")
            await self._add_messages_to_session(ctx.session, user_peer, messages)
            return len(messages)

        chunk_size = 200
        session_ids: list[str] = []
        for idx, start in enumerate(range(0, len(messages), chunk_size)):
            chunk = messages[start : start + chunk_size]
            session_id = f"{ctx.workspace_id}_session_{idx + 1}"
            session = await ctx.honcho_client.aio.session(
                id=session_id, configuration=self._get_session_configuration()
            )
            await session.aio.add_peers(
                [(user_peer, SessionPeerConfig(observe_me=True, observe_others=False))]
            )
            await self._add_messages_to_session(session, user_peer, chunk)
            session_ids.append(session_id)

        ctx.peers["_session_ids"] = session_ids
        return len(messages)

    def get_dream_observers(self, item: Any) -> list[str]:
        return ["user"]

    def get_dream_session_ids(self, ctx: ItemContext, _item: Any) -> list[str]:
        if self.merge_sessions:
            return [ctx.session_id]

        session_ids = ctx.peers.get("_session_ids")
        if not isinstance(session_ids, list) or not session_ids:
            raise ValueError(
                "Non-merged OOLONG mode requires at least one chunk session ID for dreams"
            )

        session_ids_typed = cast(list[object], session_ids)
        cleaned_session_ids: list[str] = []
        for maybe_session_id in session_ids_typed:
            if isinstance(maybe_session_id, str) and maybe_session_id:
                cleaned_session_ids.append(maybe_session_id)
        if not cleaned_session_ids:
            raise ValueError(
                "Non-merged OOLONG mode has no valid chunk session IDs for dreams"
            )
        return cleaned_session_ids

    async def execute_questions(self, ctx: ItemContext, item: Any) -> TestResult:
        start_time = time.time()
        question_id = item["id"]
        context_window_id = item["context_window_id"]
        question = item["question"]
        answer_str = item["answer"]

        if self.variant == "synth":
            task_group = item.get("task_group", "unknown")
            dataset_name = item.get("dataset", "oolong-synth")
            answer_type = item.get("answer_type", "unknown")
            gold_answer = parse_synth_answer(answer_str)
        else:
            task_group = item.get("question_type", "unknown")
            dataset_name = "oolong-real"
            answer_type = "varied"
            gold_answer = parse_real_answer(answer_str)

        context_text = item["context_window_text"]
        if self.variant == "synth" and self.use_labels:
            context_text = item.get("context_window_text_with_labels", context_text)
        context_length = calculate_context_length(context_text)

        result: TestResult = {
            "question_id": question_id,
            "context_window_id": context_window_id,
            "task_group": task_group,
            "dataset": dataset_name,
            "answer_type": answer_type,
            "passed": False,
            "score": 0.0,
            "error": None,
            "start_time": start_time,
            "end_time": 0.0,
            "duration_seconds": 0.0,
            "query_executed": None,
            "output_lines": [],
        }

        user_peer = ctx.peers["user"]
        try:
            chat_kwargs: dict[str, Any] = {}
            if self.config.reasoning_level:
                chat_kwargs["reasoning_level"] = self.config.reasoning_level
            if self.merge_sessions and ctx.session is not None:
                chat_kwargs["session"] = ctx.session

            response = await user_peer.aio.chat(question, **chat_kwargs)
            actual_response = response if isinstance(response, str) else ""

            if self.variant == "synth":
                score = score_synth_response(gold_answer, actual_response, answer_type)
            else:
                score = score_real_response(gold_answer, actual_response)

            result["query_executed"] = QueryResult(
                question=question,
                expected_answer=str(gold_answer),
                actual_response=actual_response,
                score=score,
                context_length_tokens=context_length,
            )
            result["score"] = score
            result["passed"] = score >= 0.99
            result["output_lines"] = [
                f"Question: {question}",
                f"Expected: {gold_answer}",
                f"Score: {score:.3f}",
            ]
        except Exception as e:
            result["error"] = str(e)
            result["query_executed"] = QueryResult(
                question=question,
                expected_answer=str(gold_answer),
                actual_response=f"ERROR: {e}",
                score=0.0,
                context_length_tokens=context_length,
            )

        result["end_time"] = time.time()
        result["duration_seconds"] = result["end_time"] - result["start_time"]
        return result

    def print_summary(self, results: list[TestResult], total_duration: float) -> None:
        total_examples = len(results)
        perfect_scores = sum(1 for r in results if r["score"] >= 0.99)
        average_score = (
            sum(result["score"] for result in results) / total_examples
            if total_examples
            else 0.0
        )

        print(f"\n{'=' * 80}")
        print(f"OOLONG-{self.variant.upper()} BENCHMARK SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total examples: {total_examples}")
        print(f"Average score: {average_score:.3f}")
        perfect_rate = (
            (perfect_scores / total_examples) * 100 if total_examples else 0.0
        )
        print(f"Perfect scores (>=0.99): {perfect_scores} ({perfect_rate:.1f}%)")
        print(f"Total test time: {format_duration(total_duration)}")

        task_stats = calculate_task_statistics(results)
        if task_stats:
            print("\nTask group statistics:")
            for task_name, stats in sorted(task_stats.items()):
                print(
                    f"  {task_name}: avg={stats['average_score']:.3f}, perfect={stats['perfect_score_rate']:.1f}% ({stats['total']})"
                )
        print(f"{'=' * 80}")

    def generate_output(self, results: list[TestResult], total_duration: float) -> None:
        total_examples = len(results)
        perfect_scores = sum(1 for r in results if r["score"] >= 0.99)
        average_score = (
            sum(result["score"] for result in results) / total_examples
            if total_examples
            else 0.0
        )
        task_stats = calculate_task_statistics(results)
        timing_stats = calculate_timing_statistics(results, total_duration)

        summary: dict[str, Any] = {
            "metadata": {
                "benchmark": "oolong",
                "variant": self.variant,
                "split": self.split,
                "data_dir": str(self.data_dir),
                "execution_timestamp": datetime.now().isoformat(),
                "runner_version": "2.0.0",
                "base_api_port": self.config.base_api_port,
                "pool_size": self.config.pool_size,
                "timeout_seconds": self.config.timeout_seconds,
                "merge_sessions": self.merge_sessions,
                "labels": self.use_labels,
                "reasoning_level": self.config.reasoning_level,
                "deriver_settings": settings.DERIVER.model_dump(),
                "dialectic_settings": settings.DIALECTIC.model_dump(),
                "dream_settings": settings.DREAM.model_dump(),
            },
            "summary_statistics": {
                "total_examples": total_examples,
                "perfect_scores": perfect_scores,
                "perfect_score_rate": perfect_scores / total_examples
                if total_examples
                else 0.0,
                "average_score": average_score,
                "statistics_by_task_group": task_stats,
            },
            "timing": timing_stats,
            "detailed_results": [
                {
                    "question_id": result["question_id"],
                    "context_window_id": result["context_window_id"],
                    "task_group": result["task_group"],
                    "dataset": result["dataset"],
                    "answer_type": result["answer_type"],
                    "score": result["score"],
                    "passed": result["passed"],
                    "error": result["error"],
                    "duration_seconds": result["duration_seconds"],
                    "query_executed": result["query_executed"],
                }
                for result in results
            ],
        }

        if self.config.json_output:
            output_file = self.config.json_output
        else:
            output_file = Path(
                f"tests/bench/eval_results/oolong_{self.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        write_json_summary(summary, output_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OOLONG benchmark tests against a Honcho instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --variant synth --data-dir /datasets/oolong-synth
  %(prog)s --variant real --data-dir /datasets/oolong-real --split validation
  %(prog)s --variant synth --data-dir /datasets/oolong-synth --context-size 16K
  %(prog)s --variant synth --data-dir /datasets/oolong-synth --no-merge-sessions
        """,
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="synth",
        choices=["synth", "real"],
        help="Which OOLONG variant to run (default: synth)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to run (default: all)",
    )
    parser.add_argument(
        "--context-size",
        type=str,
        default=None,
        help=(
            "Context-size bucket cap, e.g. 8K, 16K, 1M, or exact token count like 16384. "
            "Sets --max-context-len; --min-context-len is only kept when explicitly passed."
        ),
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Maximum context length in tokens",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        default=False,
        help="Use context_window_text_with_labels for synth examples",
    )
    parser.add_argument(
        "--min-context-len",
        type=int,
        default=1024,
        help="Minimum context length in tokens (default: 1024, upstream behavior)",
    )
    parser.add_argument(
        "--context-window-id",
        type=str,
        default=None,
        help="Run only examples with this context_window_id",
    )
    parser.add_argument(
        "--no-merge-sessions",
        action="store_false",
        dest="merge_sessions",
        default=True,
        help="Store context across multiple sessions instead of a merged session",
    )

    add_common_arguments(parser)
    args = parser.parse_args()
    min_context_len_explicit = "--min-context-len" in sys.argv

    error = validate_common_arguments(args)
    if error:
        print(error)
        return 1

    if args.use_get_context:
        print("Error: --use-get-context is not supported by the OOLONG runner")
        return 1

    if not args.data_dir.exists():
        print(f"Error: data directory does not exist: {args.data_dir}")
        return 1

    if args.max_examples is not None and args.max_examples <= 0:
        print(f"Error: max examples must be positive, got {args.max_examples}")
        return 1

    if args.context_size:
        try:
            exact_size = parse_context_size(args.context_size)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        # OOLONG-style behavior: context-size is a bucket cap. Preserve
        # an explicit lower bound only when the user provides one.
        args.max_context_len = exact_size
        if not min_context_len_explicit:
            args.min_context_len = None
        print(
            f"Using context-size cap: <= {exact_size} tokens"
            + (
                f" (min: {args.min_context_len})"
                if args.min_context_len is not None
                else ""
            )
        )

    if args.min_context_len is not None and args.min_context_len < 0:
        print(f"Error: min context len must be >= 0, got {args.min_context_len}")
        return 1

    if args.max_context_len is not None and args.max_context_len <= 0:
        print(f"Error: max context len must be positive, got {args.max_context_len}")
        return 1

    if (
        args.max_context_len is not None
        and args.min_context_len is not None
        and args.max_context_len < args.min_context_len
    ):
        print(
            f"Error: max context len must be >= min context len ({args.max_context_len} < {args.min_context_len})"
        )
        return 1

    config = RunnerConfig.from_args(args, default_timeout=600)
    runner = OolongRunner(
        config=config,
        variant=args.variant,
        data_dir=args.data_dir,
        split=args.split,
        merge_sessions=args.merge_sessions,
        max_examples=args.max_examples,
        min_context_len=args.min_context_len,
        max_context_len=args.max_context_len,
        context_window_id=args.context_window_id,
        use_labels=args.labels,
    )
    return runner.run_and_summarize()


if __name__ == "__main__":
    exit(main())
