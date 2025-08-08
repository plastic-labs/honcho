#!/usr/bin/env -S uv run
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
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast

from anthropic import AsyncAnthropic

from src.config import settings
from src.deriver.prompts import peer_card_prompt
from src.utils.clients import honcho_llm_call


@dataclass(frozen=True)
class Candidate:
    """Represents a provider/model pair to benchmark."""

    provider: str
    model: str


@dataclass
class Case:
    """Represents a single benchmark case with expectations for grading."""

    name: str
    old_peer_card: str | None
    new_observations: list[str]
    expected_facts: list[str]


def build_cases() -> list[Case]:
    """Return a suite of diverse cases to evaluate peer card behavior."""

    return [
        Case(
            name="create_basic_card",
            old_peer_card=None,
            new_observations=[
                "I'm Alice, 29 years old.",
                "I live in San Francisco.",
                "I'm a product manager.",
                "I love climbing and cooking.",
            ],
            expected_facts=[
                "Name: Alice",
                "Age: 29",
                "Location: San Francisco",
                "Occupation: Product Manager",
                "Interests: climbing, cooking",
            ],
        ),
        Case(
            name="update_location",
            old_peer_card="""Name: Bob\nAge: 24\nLocation: New York\nOccupation: Software Engineer\nInterests: Programming, hiking""",
            new_observations=["I just moved to Boston for work."],
            expected_facts=["Location: Boston"],
        ),
        Case(
            name="update_age",
            old_peer_card="""Name: Carol\nAge: 34\nLocation: Seattle\nOccupation: Data Scientist""",
            new_observations=["I turned 35 last week!"],
            expected_facts=["Age: 35"],
        ),
        Case(
            name="add_nickname",
            old_peer_card="""Name: Daniel\nLocation: Austin\nOccupation: Graphic Designer""",
            new_observations=["Friends call me Dan the Man."],
            expected_facts=["Name: Daniel", "Nickname: Dan the Man"],
        ),
        Case(
            name="add_dislike",
            old_peer_card="""Name: Eve\nLocation: London\nOccupation: Teacher""",
            new_observations=["I absolutely hate cilantro."],
            expected_facts=["Dislikes: cilantro"],
        ),
        Case(
            name="no_new_key_info",
            old_peer_card="""Name: Frank\nAge: 41\nLocation: Chicago\nOccupation: Sales Manager""",
            new_observations=["Nice weather today. How are you?"],
            expected_facts=[
                "Name: Frank",
                "Age: 41",
                "Location: Chicago",
                "Occupation: Sales Manager",
            ],
        ),
        Case(
            name="multiple_changes",
            old_peer_card="""Name: Grace\nAge: 30\nLocation: Paris\nOccupation: Artist\nInterests: Painting""",
            new_observations=[
                "I'm 31 now.",
                "I relocated to Berlin.",
                "I started biking a lot.",
            ],
            expected_facts=[
                "Age: 31",
                "Location: Berlin",
                "Interests: biking",
            ],
        ),
        Case(
            name="limit_signal",
            old_peer_card=None,
            new_observations=[
                "I'm Henry.",
                "I like chess.",
                "I like chess a lot.",
                "People call me Hank.",
            ],
            expected_facts=["Name: Henry", "Nickname: Hank", "Interests: chess"],
        ),
    ]


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
) -> Callable[[str | None, list[str]], Coroutine[Any, Any, str]]:
    """Create an async callable that invokes the peer card prompt with a specific provider/model."""

    resolved_provider = (
        "openai" if candidate.provider == "custom" else candidate.provider
    )

    @honcho_llm_call(
        provider=cast(Any, resolved_provider),
        model=candidate.model,
        track_name="Peer Card Call",
        max_tokens=settings.DERIVER.PEER_CARD_MAX_OUTPUT_TOKENS
        or settings.LLM.DEFAULT_MAX_TOKENS,
        thinking_budget_tokens=settings.DERIVER.THINKING_BUDGET_TOKENS
        if settings.DERIVER.PROVIDER == "anthropic"
        else None,
        enable_retry=True,
        retry_attempts=1,  # unstructured output means we shouldn't need to retry, 1 just in case
    )
    async def call(old_peer_card: str | None, new_observations: list[str]) -> Any:
        """Return the prompt content for Mirascope to execute as a model call."""

        return peer_card_prompt(
            old_peer_card=old_peer_card, new_observations=new_observations
        )

    return call  # type: ignore[return-value]


async def judge_response(
    anthropic: AsyncAnthropic,
    case: Case,
    actual_card: str,
) -> dict[str, Any]:
    """Use an LLM judge to evaluate whether the card contains the expected facts.

    Returns a dict with keys: passed (bool) and reasoning (str).
    """

    system_prompt = (
        "You are an expert evaluator. Determine if a biographical card contains the expected facts. "
        "Allow flexible phrasing and synonyms. A fact is present if its semantic content is clearly stated. "
        "Only fail if a core expected fact is missing or contradicted. Always return JSON: "
        '{"passed": boolean, "reasoning": string}'
    )
    expected = "\n".join(f"- {f}" for f in case.expected_facts)
    user_prompt = (
        f"Case: {case.name}\n\nExpected facts (semantic):\n{expected}\n\n"
        f"Biographical card:\n{actual_card}\n\nEvaluate strictly by semantic equivalence."
    )

    try:
        response = await anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=300,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content_block = response.content[0]
        judgment_text = getattr(content_block, "text", None)
        if not judgment_text:
            raise ValueError("Empty judge response")
        if "```json" in judgment_text:
            start = judgment_text.find("```json") + 7
            end = judgment_text.find("```", start)
            judgment_text = judgment_text[start:end].strip()
        elif "```" in judgment_text:
            start = judgment_text.find("```") + 3
            end = judgment_text.find("```", start)
            judgment_text = judgment_text[start:end].strip()
        return json.loads(judgment_text)
    except Exception:
        ok = all(f.lower() in actual_card.lower() for f in case.expected_facts)
        return {
            "passed": ok,
            "reasoning": "Fallback string check used by judge.",
        }


async def run_benchmark(candidates: list[Candidate]) -> int:
    """Execute all cases against all candidates and print a concise report. Returns non-zero on failures."""

    anthropic_key = os.getenv("LLM_ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("LLM_ANTHROPIC_API_KEY is required for grading")
    anthropic = AsyncAnthropic(api_key=anthropic_key)

    cases = build_cases()
    any_fail = False

    print(f"Running {len(cases)} cases across {len(candidates)} candidates\n")

    for candidate in candidates:
        print(f"=== Candidate: {candidate.provider}:{candidate.model} ===")
        try:
            caller = build_peer_card_caller(candidate)
        except Exception as e:
            print(f"  SKIP: cannot initialize provider/model ({e})")
            any_fail = True
            continue

        async def run_case(
            case: Case,
            _caller: Callable[
                [str | None, list[str]], Coroutine[Any, Any, object]
            ] = caller,
        ) -> tuple[Case, dict[str, Any]]:
            card: object = await _caller(case.old_peer_card, case.new_observations)
            response = str(card)
            judgment = await judge_response(anthropic, case, response)
            return case, {"card": response, "judgment": judgment}

        results = await asyncio.gather(
            *(run_case(c) for c in cases), return_exceptions=True
        )
        for res in results:
            if isinstance(res, BaseException):
                print(f"  ERROR running case: {res}")
                any_fail = True
                continue
            case, payload = res
            judgment = payload["judgment"]
            status = "PASS" if judgment.get("passed") else "FAIL"
            print(f"  {case.name:24} {status:4} - {judgment.get('reasoning', '')}")
            if not judgment.get("passed"):
                any_fail = True
                print("    expected:")
                for f in case.expected_facts:
                    print(f"      - {f}")
                print("    got:")
                print("      " + payload["card"])
        print()

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
        help="Provider:model pairs. Repeat or comma-separate. Default: anthropic:claude-3-7-sonnet-20250219",
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

    return asyncio.run(run_benchmark(candidates))


if __name__ == "__main__":
    raise SystemExit(main())
