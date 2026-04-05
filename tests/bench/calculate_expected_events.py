#!/usr/bin/env python3
"""
Calculate expected CloudEvents and input tokens from longmemeval test case files.

This script parses longmemeval test files (like longmemeval_sanity.json, longmemeval_oracle.json)
and calculates the expected number of CloudEvents and input tokens that should be processed
when running the test.

Supported test files:
- longmemeval_sanity.json (5 questions, quick verification)
- longmemeval_oracle.json (full evaluation set)
- longmemeval_oracle_100.json (100-question subset)
- longmemeval_s_cleaned.json (cleaned dataset)

Event types calculated:
- representation.completed: One per (session, observed) batch processed by deriver
- dialectic.completed: One per dialectic chat query
- dream.run: One per dream consolidation trigger

Token counting:
- Uses tiktoken (o200k_base encoding) for accurate token estimation
- Falls back to character/4 estimation if tiktoken unavailable

Usage:
    python calculate_expected_events.py <test_file.json> [options]

Examples:
    # Calculate events for sanity test (default: separate sessions, dialectic chat, with dream)
    python calculate_expected_events.py longmemeval_data/longmemeval_sanity.json

    # Calculate events for full oracle test
    python calculate_expected_events.py longmemeval_data/longmemeval_oracle.json

    # Calculate with merged sessions
    python calculate_expected_events.py longmemeval_data/longmemeval_sanity.json --merge-sessions

    # Calculate without dream events
    python calculate_expected_events.py longmemeval_data/longmemeval_sanity.json --no-dream

    # Calculate with get_context instead of dialectic (no dialectic events)
    python calculate_expected_events.py longmemeval_data/longmemeval_sanity.json --use-get-context

    # Calculate for first 10 questions only
    python calculate_expected_events.py longmemeval_data/longmemeval_oracle.json --test-count 10

    # Output as JSON for programmatic use
    python calculate_expected_events.py longmemeval_data/longmemeval_sanity.json --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Token counting setup
try:
    import tiktoken

    _tokenizer = tiktoken.get_encoding("o200k_base")
    _USE_TIKTOKEN = True
except ImportError:
    _tokenizer = None
    _USE_TIKTOKEN = False  # pyright: ignore[reportConstantRedefinition]


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken, with fallback to char/4."""
    if not text:
        return 0
    if _USE_TIKTOKEN and _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text))
        except Exception:
            pass
    # Fallback: rough estimate of 4 chars per token
    return len(text) // 4


@dataclass
class QuestionEventCounts:
    """Event counts and token metrics for a single question."""

    question_id: str
    question_type: str
    sessions_count: int
    total_messages: int
    observed_peer: str
    representation_events: int
    dialectic_events: int
    dream_events: int
    # Token counts
    total_input_tokens: int = 0
    user_tokens: int = 0
    assistant_tokens: int = 0
    question_tokens: int = 0

    @property
    def total_events(self) -> int:
        return self.representation_events + self.dialectic_events + self.dream_events


@dataclass
class TestFileEventSummary:
    """Summary of expected events for a test file."""

    file_path: str
    total_questions: int
    merge_sessions: bool
    use_get_context: bool
    dream_enabled: bool
    questions: list[QuestionEventCounts] = field(default_factory=list)

    @property
    def total_representation_events(self) -> int:
        return sum(q.representation_events for q in self.questions)

    @property
    def total_dialectic_events(self) -> int:
        return sum(q.dialectic_events for q in self.questions)

    @property
    def total_dream_events(self) -> int:
        return sum(q.dream_events for q in self.questions)

    @property
    def total_events(self) -> int:
        return (
            self.total_representation_events
            + self.total_dialectic_events
            + self.total_dream_events
        )

    @property
    def total_input_tokens(self) -> int:
        return sum(q.total_input_tokens for q in self.questions)

    @property
    def total_user_tokens(self) -> int:
        return sum(q.user_tokens for q in self.questions)

    @property
    def total_assistant_tokens(self) -> int:
        return sum(q.assistant_tokens for q in self.questions)

    @property
    def total_question_tokens(self) -> int:
        return sum(q.question_tokens for q in self.questions)

    def by_question_type(self) -> dict[str, dict[str, int]]:
        """Group event counts by question type."""
        result: dict[str, dict[str, int]] = {}
        for q in self.questions:
            if q.question_type not in result:
                result[q.question_type] = {
                    "questions": 0,
                    "representation_events": 0,
                    "dialectic_events": 0,
                    "dream_events": 0,
                    "total_events": 0,
                    "total_input_tokens": 0,
                    "user_tokens": 0,
                    "assistant_tokens": 0,
                }
            result[q.question_type]["questions"] += 1
            result[q.question_type]["representation_events"] += q.representation_events
            result[q.question_type]["dialectic_events"] += q.dialectic_events
            result[q.question_type]["dream_events"] += q.dream_events
            result[q.question_type]["total_events"] += q.total_events
            result[q.question_type]["total_input_tokens"] += q.total_input_tokens
            result[q.question_type]["user_tokens"] += q.user_tokens
            result[q.question_type]["assistant_tokens"] += q.assistant_tokens
        return result


def count_messages_by_role(
    session_messages: list[dict[str, Any]],
) -> dict[str, int]:
    """Count messages by role in a session."""
    counts = {"user": 0, "assistant": 0}
    for msg in session_messages:
        role = msg.get("role", "")
        if role in counts:
            counts[role] += 1
    return counts


def calculate_question_events(
    question_data: dict[str, Any],
    merge_sessions: bool,
    use_get_context: bool,
    dream_enabled: bool,
) -> QuestionEventCounts:
    """
    Calculate expected events and tokens for a single question.

    The calculation is based on how the longmem.py runner processes test cases:
    - Creates workspace per question
    - Creates sessions (merged or separate based on config)
    - Adds messages to sessions (triggers representation events)
    - Triggers dream consolidation (if enabled)
    - Queries via dialectic chat (if not using get_context)

    Args:
        question_data: Question data from the test file
        merge_sessions: Whether sessions are merged into one
        use_get_context: Whether using get_context instead of dialectic chat
        dream_enabled: Whether dream consolidation is enabled

    Returns:
        QuestionEventCounts with calculated event counts and token metrics
    """
    question_id = question_data.get("question_id", "unknown")
    question_type = question_data.get("question_type", "unknown")
    haystack_sessions = question_data.get("haystack_sessions", [])
    question_text = question_data.get("question", "")

    # Determine which peer is observed based on question type
    # single-session-assistant observes assistant, all others observe user
    is_assistant_type = question_type == "single-session-assistant"
    observed_peer = "assistant" if is_assistant_type else "user"

    # Count total messages and tokens
    total_messages = 0
    user_tokens = 0
    assistant_tokens = 0

    for session in haystack_sessions:
        for msg in session:
            total_messages += 1
            content = msg.get("content", "")
            role = msg.get("role", "")
            tokens = estimate_tokens(content)
            if role == "user":
                user_tokens += tokens
            elif role == "assistant":
                assistant_tokens += tokens

    # Token count for the question itself (used in dialectic)
    question_tokens = estimate_tokens(question_text)

    # Total input tokens = all message content
    total_input_tokens = user_tokens + assistant_tokens

    # Calculate representation events
    # Each unique (session, observed) pair generates one representation event
    # (assuming messages fit within REPRESENTATION_BATCH_MAX_TOKENS)
    # When merge_sessions=True, all messages go into one session
    if merge_sessions:
        # One merged session = one representation event
        representation_events = 1
        sessions_count = 1
    else:
        # One representation event per session
        sessions_count = len(haystack_sessions)
        representation_events = sessions_count

    # Calculate dialectic events
    # One per chat query (if using dialectic chat)
    dialectic_events = 0 if use_get_context else 1

    # Calculate dream events
    # One per dream trigger
    dream_events = 1 if dream_enabled else 0

    return QuestionEventCounts(
        question_id=question_id,
        question_type=question_type,
        sessions_count=sessions_count,
        total_messages=total_messages,
        observed_peer=observed_peer,
        representation_events=representation_events,
        dialectic_events=dialectic_events,
        dream_events=dream_events,
        total_input_tokens=total_input_tokens,
        user_tokens=user_tokens,
        assistant_tokens=assistant_tokens,
        question_tokens=question_tokens,
    )


def calculate_test_file_events(
    test_file: Path,
    merge_sessions: bool = False,
    use_get_context: bool = False,
    dream_enabled: bool = True,
    question_ids: list[str] | None = None,
    test_count: int | None = None,
) -> TestFileEventSummary:
    """
    Calculate expected events for all questions in a test file.

    Args:
        test_file: Path to the test file (JSON)
        merge_sessions: Whether to merge all sessions into one per question
        use_get_context: Whether using get_context instead of dialectic chat
        dream_enabled: Whether dream consolidation is enabled
        question_ids: Optional list of specific question IDs to include
        test_count: Optional limit on number of questions to process

    Returns:
        TestFileEventSummary with calculated event counts
    """
    with open(test_file) as f:
        questions = json.load(f)

    # Filter by question_ids if specified
    if question_ids:
        questions = [q for q in questions if q.get("question_id") in question_ids]

    # Limit by test_count if specified
    if test_count is not None:
        questions = questions[:test_count]

    summary = TestFileEventSummary(
        file_path=str(test_file),
        total_questions=len(questions),
        merge_sessions=merge_sessions,
        use_get_context=use_get_context,
        dream_enabled=dream_enabled,
    )

    for question_data in questions:
        counts = calculate_question_events(
            question_data, merge_sessions, use_get_context, dream_enabled
        )
        summary.questions.append(counts)

    return summary


def format_tokens(tokens: int) -> str:
    """Format token count with K/M suffix for readability."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def print_summary(summary: TestFileEventSummary, verbose: bool = False) -> None:
    """Print a formatted summary of expected events and tokens."""
    print(f"\n{'=' * 60}")
    print(f"Expected CloudEvents for: {summary.file_path}")
    print(f"{'=' * 60}")
    print("\nConfiguration:")
    print(f"  - Merge sessions: {summary.merge_sessions}")
    print(f"  - Use get_context: {summary.use_get_context}")
    print(f"  - Dream enabled: {summary.dream_enabled}")
    print(
        f"  - Token counting: {'tiktoken (o200k_base)' if _USE_TIKTOKEN else 'estimate (chars/4)'}"
    )

    print(f"\nTotal Questions: {summary.total_questions}")

    print(f"\n{'─' * 40}")
    print("Expected Event Counts:")
    print(f"{'─' * 40}")
    print(f"  representation.completed: {summary.total_representation_events:>6}")
    print(f"  dialectic.completed:      {summary.total_dialectic_events:>6}")
    print(f"  dream.run:                {summary.total_dream_events:>6}")
    print(f"{'─' * 40}")
    print(f"  TOTAL:                    {summary.total_events:>6}")
    print(f"{'─' * 40}")

    print(f"\n{'─' * 40}")
    print("Expected Input Tokens:")
    print(f"{'─' * 40}")
    print(f"  User messages:      {summary.total_user_tokens:>12,}")
    print(f"  Assistant messages: {summary.total_assistant_tokens:>12,}")
    print(f"  Questions:          {summary.total_question_tokens:>12,}")
    print(f"{'─' * 40}")
    print(
        f"  TOTAL (messages):   {summary.total_input_tokens:>12,}  ({format_tokens(summary.total_input_tokens)})"
    )
    print(f"{'─' * 40}")

    # Breakdown by question type
    by_type = summary.by_question_type()
    if len(by_type) > 1:
        print("\nBreakdown by Question Type:")
        print(f"{'─' * 70}")
        for qtype, counts in sorted(by_type.items()):
            print(f"\n  {qtype}:")
            print(f"    Questions:      {counts['questions']:>4}")
            print(f"    representation: {counts['representation_events']:>4}")
            print(f"    dialectic:      {counts['dialectic_events']:>4}")
            print(f"    dream:          {counts['dream_events']:>4}")
            print(f"    total events:   {counts['total_events']:>4}")
            print(
                f"    input tokens:   {counts['total_input_tokens']:>10,}  ({format_tokens(counts['total_input_tokens'])})"
            )

    if verbose:
        print(f"\n{'─' * 70}")
        print("Per-Question Details:")
        print(f"{'─' * 70}")
        for q in summary.questions:
            print(f"\n  {q.question_id} ({q.question_type}):")
            print(f"    Sessions: {q.sessions_count}, Messages: {q.total_messages}")
            print(f"    Observed: {q.observed_peer}")
            print(
                f"    Events: repr={q.representation_events}, dial={q.dialectic_events}, dream={q.dream_events}"
            )
            print(
                f"    Tokens: user={q.user_tokens:,}, asst={q.assistant_tokens:,}, total={q.total_input_tokens:,}"
            )

    print()


def output_json(summary: TestFileEventSummary) -> str:
    """Output summary as JSON for programmatic use."""
    return json.dumps(
        {
            "file_path": summary.file_path,
            "configuration": {
                "merge_sessions": summary.merge_sessions,
                "use_get_context": summary.use_get_context,
                "dream_enabled": summary.dream_enabled,
                "token_counting": "tiktoken" if _USE_TIKTOKEN else "estimate",
            },
            "total_questions": summary.total_questions,
            "expected_events": {
                "representation_completed": summary.total_representation_events,
                "dialectic_completed": summary.total_dialectic_events,
                "dream_run": summary.total_dream_events,
                "total": summary.total_events,
            },
            "expected_tokens": {
                "user_messages": summary.total_user_tokens,
                "assistant_messages": summary.total_assistant_tokens,
                "questions": summary.total_question_tokens,
                "total_messages": summary.total_input_tokens,
            },
            "by_question_type": summary.by_question_type(),
            "questions": [
                {
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "sessions_count": q.sessions_count,
                    "total_messages": q.total_messages,
                    "observed_peer": q.observed_peer,
                    "representation_events": q.representation_events,
                    "dialectic_events": q.dialectic_events,
                    "dream_events": q.dream_events,
                    "total_events": q.total_events,
                    "total_input_tokens": q.total_input_tokens,
                    "user_tokens": q.user_tokens,
                    "assistant_tokens": q.assistant_tokens,
                    "question_tokens": q.question_tokens,
                }
                for q in summary.questions
            ],
        },
        indent=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate expected CloudEvents from longmemeval test case files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "test_file",
        type=Path,
        help="Path to the longmemeval test file (JSON)",
    )
    parser.add_argument(
        "--merge-sessions",
        action="store_true",
        help="Calculate with merged sessions (default: separate sessions)",
    )
    parser.add_argument(
        "--use-get-context",
        action="store_true",
        help="Calculate for get_context mode (no dialectic events)",
    )
    parser.add_argument(
        "--no-dream",
        action="store_true",
        help="Calculate without dream events",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        action="append",
        dest="question_ids",
        help="Only include specific question IDs (can be repeated)",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        help="Limit to first N questions",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-question details",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )

    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"Error: Test file not found: {args.test_file}")
        return

    summary = calculate_test_file_events(
        test_file=args.test_file,
        merge_sessions=args.merge_sessions,
        use_get_context=args.use_get_context,
        dream_enabled=not args.no_dream,
        question_ids=args.question_ids,
        test_count=args.test_count,
    )

    if args.json:
        print(output_json(summary))
    else:
        print_summary(summary, verbose=args.verbose)


if __name__ == "__main__":
    main()
