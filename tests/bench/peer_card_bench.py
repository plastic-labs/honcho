"""
Peer Card Benchmark

This benchmark exercises the peer card LLM call with varied inputs and compares
outputs across multiple provider/model candidates. Results are graded by an LLM
judge.

Usage example:
  python -m tests.bench.peer_card_bench --candidates anthropic:claude-3-7-sonnet-20250219 --candidates openai:gpt-4o-mini-2024-07-18

Environment variables for providers:
  - Anthropic: LLM_ANTHROPIC_API_KEY
  - OpenAI: LLM_OPENAI_API_KEY or OPENAI_API_KEY
  - Google (Gemini): LLM_GEMINI_API_KEY or GEMINI_API_KEY
  - Groq: LLM_GROQ_API_KEY or GROQ_API_KEY
"""

import argparse
import asyncio
import json
import os
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic

from src.config import settings
from src.deriver.prompts import peer_card_prompt
from src.utils.clients import honcho_llm_call
from src.utils.peer_card import PeerCardQuery
from src.utils.representation import ExplicitObservation, Representation

COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"


@dataclass(frozen=True)
class Candidate:
    """Represents a provider/model pair to benchmark."""

    provider: str
    model: str


@dataclass
class Case:
    """Represents a single benchmark case with expectations for grading.

    Attributes:
        name: Human-friendly identifier for the case.
        old_peer_card: Existing card text to update, or None to create fresh.
        new_observations: New input observations that may change the card.
        expected_facts: Facts that must be semantically present in the result.
        forbidden_facts: Facts that must NOT be present in the result.
    """

    name: str
    old_peer_card: list[str] | None
    new_observations: list[str]
    expected_facts: list[str]
    forbidden_facts: list[str]


def load_case_file(path: Path) -> Case:
    """Load a single peer-card test case from a JSON file.

    The JSON schema must include: name, old_peer_card (nullable), new_observations (list[str]), expected_facts (list[str]).
    """

    with path.open() as f:
        data = json.load(f)

    return Case(
        name=str(data["name"]),
        old_peer_card=data.get("old_peer_card"),
        new_observations=list(data.get("new_observations", [])),
        expected_facts=list(data.get("expected_facts", [])),
        forbidden_facts=list(data.get("forbidden_facts", [])),
    )


def load_cases(tests_dir: Path, test_name: str | None) -> list[Case]:
    """Load all cases from a directory, or a specific case by filename.

    Args:
        tests_dir: Directory containing JSON case files.
        test_name: Optional filename to load a single case (e.g., "create_basic_card.json").

    Returns:
        List of loaded Case objects.
    """

    if test_name:
        file_path = tests_dir / test_name
        if not file_path.exists():
            raise FileNotFoundError(f"Test file {file_path} does not exist")
        return [load_case_file(file_path)]

    files = sorted(p for p in tests_dir.glob("*.json") if p.is_file())
    return [load_case_file(p) for p in files]


def parse_candidates(values: list[str]) -> list[Candidate]:
    """Parse provider:model strings into Candidate objects."""

    result: list[Candidate] = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        if ":" not in v:
            raise ValueError(f"Invalid candidate format: {v} (expected provider:model)")
        provider, model = v.split(":", 1)
        result.append(Candidate(provider=provider.strip(), model=model.strip()))
    return result


def deduplicate_preserve_order(items: list[Candidate]) -> list[Candidate]:
    """Return a new list with duplicate provider:model pairs removed, preserving order."""

    seen: set[tuple[str, str]] = set()
    unique: list[Candidate] = []
    for item in items:
        key = (item.provider, item.model)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def build_peer_card_caller(
    candidate: Candidate,
) -> Callable[[list[str] | None, Representation], Coroutine[Any, Any, PeerCardQuery]]:
    """Create an async callable that invokes the peer card prompt with a specific provider/model."""

    resolved_provider = (
        "openai" if candidate.provider == "custom" else candidate.provider
    )

    async def call(
        old_peer_card: list[str] | None, new_observations: Representation
    ) -> PeerCardQuery:
        prompt = peer_card_prompt(
            old_peer_card=old_peer_card,
            new_observations=new_observations.str_no_timestamps(),
        )

        response = await honcho_llm_call(
            provider=cast(Any, resolved_provider),
            model=candidate.model,
            prompt=prompt,
            max_tokens=settings.PEER_CARD.MAX_OUTPUT_TOKENS,
            response_model=PeerCardQuery,
            json_mode=True,
            reasoning_effort="minimal",
            enable_retry=True,
            retry_attempts=3,
        )

        return response.content

    return call


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object from potentially noisy LLM output.

    Strategy in order:
    - Try to parse the whole text as JSON
    - Try to parse contents of any fenced code blocks (``` or ```json)
    - Try the substring from the first '{' to the last '}'
    - Scan for balanced-brace substrings and try them in order

    Raises ValueError when no valid JSON object can be found.
    """

    stripped: str = text.strip()

    candidates: list[str] = []

    # 1) Fenced code blocks
    if "```" in stripped:
        idx: int = 0
        while True:
            start = stripped.find("```", idx)
            if start == -1:
                break
            lang_line_end = stripped.find("\n", start + 3)
            if lang_line_end == -1:
                break
            end = stripped.find("```", lang_line_end + 1)
            if end == -1:
                break
            block = stripped[lang_line_end + 1 : end].strip()
            if block:
                candidates.append(block)
            idx = end + 3

    # 2) From first '{' to last '}'
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    # 3) Balanced-brace scan
    depth = 0
    start_idx = -1
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx != -1:
                    candidates.append(stripped[start_idx : i + 1])

    # Try all candidates, prefer ones containing the expected keys
    preferred_keys = {"passed", "reasoning"}
    fallback_obj: dict[str, Any] | None = None
    for cand in candidates:
        try:
            obj_candidate: object = json.loads(cand)
            if isinstance(obj_candidate, dict):
                casted_obj: dict[str, Any] = {
                    str(k): v  # pyright: ignore
                    for k, v in obj_candidate.items()  # pyright: ignore
                }
                if preferred_keys.issubset(set(casted_obj.keys())):
                    return casted_obj
                if fallback_obj is None:
                    fallback_obj = casted_obj
        except Exception:
            continue

    if fallback_obj is not None:
        return fallback_obj

    raise ValueError("Could not extract JSON from judge response")


async def judge_response(
    anthropic: AsyncAnthropic,
    case: Case,
    actual_card: list[str],
) -> dict[str, Any]:
    """Use an LLM judge to evaluate whether the card contains the expected facts.

    Returns a dict with keys: passed (bool) and reasoning (str).
    """

    system_prompt = (
        "You are an expert evaluator. Determine if a biographical card satisfies BOTH: "
        "(1) it contains all expected facts (semantic match allowed) and "
        "(2) it does NOT contain any forbidden facts (semantic match). "
        "Allow flexible phrasing and synonyms for matching. A fact is present if its semantic content is clearly stated. "
        "Fail if any expected fact is missing or any forbidden fact appears. Always return JSON: "
        '{"passed": boolean, "reasoning": string}'
    )
    expected = "\n".join(f"- {f}" for f in case.expected_facts)
    forbidden = "\n".join(f"- {f}" for f in case.forbidden_facts)
    card_text = "\n".join(actual_card) if actual_card else "- (none)"
    user_prompt = (
        f"Case: {case.name}\n\n"
        f"Expected facts (must appear, semantic):\n{expected or '- (none)'}\n\n"
        f"Forbidden facts (must NOT appear, semantic):\n{forbidden or '- (none)'}\n\n"
        f"Biographical card to evaluate:\n{card_text}\n\n"
        f"Evaluation criteria: PASS only if all expected facts are present AND all forbidden facts are absent."
    )

    judgment_text: str | None = None
    try:
        response = await anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content_block = response.content[0]
        judgment_text = getattr(content_block, "text", None)
        if not judgment_text:
            raise ValueError("Empty judge response")
        return _extract_json_from_text(judgment_text)
    except Exception as e:
        print(judgment_text)
        raise ValueError(f"!!!Error judging response for case {case.name}: {e}") from e


async def run_benchmark(candidates: list[Candidate], cases: list[Case]) -> int:
    """Execute cases against candidates and print a concise report.

    Returns non-zero when any case fails for any candidate.
    """

    anthropic_key = os.getenv("LLM_ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("LLM_ANTHROPIC_API_KEY is required for grading")
    anthropic = AsyncAnthropic(api_key=anthropic_key)

    any_fail = False

    print(f"Running {len(cases)} cases across {len(candidates)} candidates\n")

    for candidate in candidates:
        print(f"=== Candidate: {candidate.provider}:{candidate.model} ===")
        candidate_start_time = time.perf_counter()
        try:
            caller = build_peer_card_caller(candidate)
        except Exception as e:
            print(f"  SKIP: cannot initialize provider/model ({e})")
            any_fail = True
            continue

        async def run_case(
            case: Case,
            _caller: Callable[
                [list[str] | None, Representation], Coroutine[Any, Any, PeerCardQuery]
            ] = caller,
        ) -> tuple[Case, dict[str, Any]]:
            card: PeerCardQuery = await _caller(
                case.old_peer_card,
                Representation(
                    explicit=[
                        ExplicitObservation(
                            content=o,
                            created_at=datetime.now(timezone.utc),
                            message_ids=[(0, 0)],
                            session_name=case.name,
                        )
                        for o in case.new_observations
                    ]
                ),
            )
            new_card = card.card
            if new_card is None or new_card == []:
                new_card = case.old_peer_card or []
            judgment = await judge_response(anthropic, case, new_card)
            return case, {"card": card, "judgment": judgment}

        results = await asyncio.gather(
            *(run_case(c) for c in cases), return_exceptions=True
        )
        passed_count: int = 0
        for res in results:
            if isinstance(res, BaseException):
                print(f"  ERROR running case: {res}")
                any_fail = True
                continue
            case, payload = res
            judgment = payload["judgment"]
            passed = bool(judgment.get("passed"))
            status_colored = (
                f"{COLOR_GREEN}PASS{COLOR_RESET}"
                if passed
                else f"{COLOR_RED}FAIL{COLOR_RESET}"
            )
            if passed:
                print(f"  {case.name:24} {status_colored}")
                passed_count += 1
            else:
                print(
                    f"  {case.name:24} {status_colored} - {judgment.get('reasoning', '')}"
                )
                any_fail = True
                print("    expected:")
                for f in case.expected_facts:
                    print(f"      - {f}")
                if case.forbidden_facts:
                    print("    forbidden (must NOT appear):")
                    for f in case.forbidden_facts:
                        print(f"      - {f}")
                print("    got:")
                [print("      " + line) for line in payload["card"].card]
                print("    with 'notes' field:")
                if payload["card"].notes:
                    print("      " + payload["card"].notes)
                else:
                    print("      - (none)")
        total_count: int = len(cases)
        percentage: float = (passed_count / total_count * 100.0) if total_count else 0.0
        print(f"  Summary: {passed_count}/{total_count} passed ({percentage:.1f}%)")
        elapsed = time.perf_counter() - candidate_start_time
        print(f"  Time: {elapsed:.2f}s\n")

    print("Done.")
    return 1 if any_fail else 0


def main() -> int:
    """CLI entry point for running the peer card benchmark."""

    parser = argparse.ArgumentParser(
        description="Benchmark peer card LLM behavior across models"
    )
    parser.add_argument(
        "--candidates",
        action="append",
        default=None,
        help=(
            "Provider:model pairs. Repeat or comma-separate. "
            "Default: anthropic:claude-3-7-sonnet-20250219"
        ),
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path("tests/bench/peer_card_tests"),
        help=(
            "Directory containing JSON peer-card cases "
            "(default: tests/bench/peer_card_tests)"
        ),
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run a specific test file by name (e.g., 'create_basic_card.json')",
    )
    args = parser.parse_args()

    # Use the default candidate only when the flag is not provided at all
    candidate_entries: list[str] = (
        args.candidates
        if args.candidates is not None
        else ["anthropic:claude-3-7-sonnet-20250219"]
    )

    flat: list[str] = []
    for entry in candidate_entries:
        flat.extend([s.strip() for s in entry.split(",") if s.strip()])
    candidates = parse_candidates(flat)
    candidates = deduplicate_preserve_order(candidates)

    # Load cases from JSON files
    if not args.tests_dir.exists():
        raise SystemExit(f"Error: Tests directory {args.tests_dir} does not exist")
    cases = load_cases(args.tests_dir, args.test)
    if not cases:
        raise SystemExit(f"Error: No JSON test cases found in {args.tests_dir}")

    return asyncio.run(run_benchmark(candidates, cases))


if __name__ == "__main__":
    raise SystemExit(main())
