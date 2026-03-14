#!/usr/bin/env python3
"""
Compare prompt-prefix cache behavior between two Honcho worktrees.

This is intentionally a narrow probe. It does not start Honcho servers, touch the
database, or exercise Hermes/session state. Instead, it imports each worktree's
`src.utils.clients.honcho_llm_call_inner` and runs a few controlled message
patterns against live providers.

The important scenario is `change_history`: the first system block stays stable
while the second rolling system block changes. The candidate branch should retain
more cache reuse there because it preserves multiple cacheable system blocks
instead of flattening them into one blob.

Example:
  uv run python scripts/compare_prefix_cache.py \
    --baseline-worktree /path/to/honcho-main \
    --candidate-worktree /path/to/honcho-branch \
    --provider anthropic-haiku=anthropic:claude-haiku-4-5 \
    --provider openrouter-haiku=custom:anthropic/claude-haiku-4.5 \
    --provider openai-mini=openai:gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHILD_CODE = r"""
import asyncio
import json
import sys
import time

payload = json.loads(sys.argv[1])

from src.utils.clients import honcho_llm_call_inner


async def run() -> None:
    results = []
    for scenario in payload["scenarios"]:
        calls = []
        for call in scenario["calls"]:
            start = time.perf_counter()
            response = await honcho_llm_call_inner(
                provider=payload["provider"],
                model=payload["model"],
                prompt="",
                max_tokens=payload["max_tokens"],
                temperature=0,
                messages=call["messages"],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            calls.append(
                {
                    "label": call["label"],
                    "duration_ms": elapsed_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cache_creation_input_tokens": response.cache_creation_input_tokens,
                    "cache_read_input_tokens": response.cache_read_input_tokens,
                    "finish_reasons": response.finish_reasons,
                    "content_preview": (response.content or "")[:120],
                }
            )
        results.append({"name": scenario["name"], "calls": calls})

    print(json.dumps({"scenarios": results}))


asyncio.run(run())
"""


BASE_PREFIX = "\n".join(
    [
        "You are Honcho's memory-backed reasoning layer.",
        "Answer precisely, prefer explicit dates, and preserve user-specific facts.",
        "Treat the following policy statements as durable background instructions.",
    ]
    + [
        f"Policy {i}: Keep stable user preferences and constraints explicit in memory-aware answers."
        for i in range(1, 121)
    ]
)

BASE_PREFIX_VARIANT = "\n".join(
    [
        "You are Honcho's memory-backed reasoning layer.",
        "Answer precisely, prefer explicit dates, and preserve user-specific facts.",
        "Treat the following policy statements as durable background instructions.",
    ]
    + [
        f"Policy {i}: Emphasize durable preferences, deadlines, and factual constraints in every answer."
        for i in range(1, 121)
    ]
)

ROLLING_HISTORY_A = "\n".join(
    [
        "Session history snapshot A:",
        "The user usually drinks green tea on weekdays and espresso on Sundays.",
        "The user moved a product launch deadline from April 25, 2026 to April 22, 2026.",
        "The user prefers short bullet points and exact dates for updates.",
    ]
    + [
        f"History line {i}: The user mentioned project detail {i} while discussing the hermes-memory rollout."
        for i in range(1, 121)
    ]
)

ROLLING_HISTORY_B = "\n".join(
    [
        "Session history snapshot B:",
        "The user usually drinks green tea on weekdays and espresso on Sundays.",
        "The user moved a product launch deadline from April 25, 2026 to April 22, 2026.",
        "The user prefers short bullet points and exact dates for updates.",
    ]
    + [
        f"History line {i}: The user mentioned project detail {i} while discussing the prefix-cache rollout."
        for i in range(1, 121)
    ]
)


@dataclass
class ProviderSpec:
    label: str
    provider: str
    model: str


@dataclass
class VariantSpec:
    label: str
    worktree: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prompt-prefix cache behavior between two Honcho worktrees."
    )
    parser.add_argument(
        "--baseline-worktree",
        required=True,
        type=Path,
        help="Path to the baseline Honcho worktree, typically main.",
    )
    parser.add_argument(
        "--candidate-worktree",
        required=True,
        type=Path,
        help="Path to the candidate Honcho worktree.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        required=True,
        help=(
            "Provider spec in the form label=provider:model. "
            "Example: anthropic-haiku=anthropic:claude-haiku-4-5"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max output tokens for each probe call.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=["repeat_exact", "change_user", "change_history", "change_base"],
        help="Optional scenario filter. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the raw comparison output as JSON.",
    )
    return parser.parse_args()


def parse_provider_spec(raw: str) -> ProviderSpec:
    if "=" not in raw or ":" not in raw:
        raise ValueError(
            f"Invalid provider spec {raw!r}. Expected label=provider:model"
        )
    label, provider_model = raw.split("=", 1)
    provider, model = provider_model.split(":", 1)
    return ProviderSpec(label=label, provider=provider, model=model)


def build_messages(base_prefix: str, rolling_history: str, user_query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": base_prefix},
        {"role": "system", "content": rolling_history},
        {"role": "user", "content": user_query},
    ]


def build_scenarios(selected: set[str] | None) -> list[dict[str, Any]]:
    scenario_defs = [
        {
            "name": "repeat_exact",
            "calls": [
                {
                    "label": "cold",
                    "messages": build_messages(
                        BASE_PREFIX,
                        ROLLING_HISTORY_A,
                        "What is the user's preferred morning drink schedule?",
                    ),
                },
                {
                    "label": "warm_same",
                    "messages": build_messages(
                        BASE_PREFIX,
                        ROLLING_HISTORY_A,
                        "What is the user's preferred morning drink schedule?",
                    ),
                },
            ],
        },
        {
            "name": "change_user",
            "calls": [
                {
                    "label": "cold",
                    "messages": build_messages(
                        BASE_PREFIX,
                        ROLLING_HISTORY_A,
                        "What is the user's preferred morning drink schedule?",
                    ),
                },
                {
                    "label": "warm_user_changed",
                    "messages": build_messages(
                        BASE_PREFIX,
                        ROLLING_HISTORY_A,
                        "What exact launch date should be remembered for the user?",
                    ),
                },
            ],
        },
        {
            "name": "change_history",
            "calls": [
                {
                    "label": "cold",
                    "messages": build_messages(
                        BASE_PREFIX,
                        ROLLING_HISTORY_A,
                        "Summarize the user's communication preference in one sentence.",
                    ),
                },
                {
                    "label": "warm_history_changed",
                    "messages": build_messages(
                        BASE_PREFIX,
                        ROLLING_HISTORY_B,
                        "Summarize the user's communication preference in one sentence.",
                    ),
                },
            ],
        },
        {
            "name": "change_base",
            "calls": [
                {
                    "label": "cold",
                    "messages": build_messages(
                        BASE_PREFIX,
                        ROLLING_HISTORY_A,
                        "What city is the user considering for a move?",
                    ),
                },
                {
                    "label": "warm_base_changed",
                    "messages": build_messages(
                        BASE_PREFIX_VARIANT,
                        ROLLING_HISTORY_A,
                        "What city is the user considering for a move?",
                    ),
                },
            ],
        },
    ]
    if not selected:
        return scenario_defs
    return [scenario for scenario in scenario_defs if scenario["name"] in selected]


def run_probe(
    variant: VariantSpec,
    provider: ProviderSpec,
    scenarios: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "provider": provider.provider,
        "model": provider.model,
        "max_tokens": max_tokens,
        "scenarios": scenarios,
    }
    process = subprocess.run(
        [sys.executable, "-c", CHILD_CODE, json.dumps(payload)],
        cwd=variant.worktree,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{variant.label} probe failed for {provider.label}.\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
    return json.loads(process.stdout)


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_variant_result(variant: VariantSpec, result: dict[str, Any]) -> None:
    print(f"  {variant.label}")
    for scenario in result["scenarios"]:
        print(f"    [{scenario['name']}]")
        for call in scenario["calls"]:
            print(
                "      "
                f"{call['label']:<18} "
                f"read={format_metric(call['cache_read_input_tokens']):>8} "
                f"create={format_metric(call['cache_creation_input_tokens']):>8} "
                f"input={format_metric(call['input_tokens']):>8} "
                f"ms={format_metric(call['duration_ms']):>8}"
            )


def print_delta_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    baseline_by_name = {scenario["name"]: scenario for scenario in baseline["scenarios"]}
    candidate_by_name = {scenario["name"]: scenario for scenario in candidate["scenarios"]}
    print("  delta summary (candidate - baseline)")
    for name in baseline_by_name:
        base_calls = baseline_by_name[name]["calls"]
        cand_calls = candidate_by_name[name]["calls"]
        if len(base_calls) < 2 or len(cand_calls) < 2:
            continue
        base_warm = base_calls[1]
        cand_warm = cand_calls[1]
        read_delta = (
            cand_warm["cache_read_input_tokens"] - base_warm["cache_read_input_tokens"]
        )
        create_delta = (
            cand_warm["cache_creation_input_tokens"]
            - base_warm["cache_creation_input_tokens"]
        )
        latency_delta = cand_warm["duration_ms"] - base_warm["duration_ms"]
        print(
            "    "
            f"{name:<16} "
            f"read_delta={read_delta:+8.2f} "
            f"create_delta={create_delta:+8.2f} "
            f"warm_latency_delta_ms={latency_delta:+8.2f}"
        )


def main() -> None:
    args = parse_args()
    providers = [parse_provider_spec(raw) for raw in args.provider]
    baseline = VariantSpec("baseline", args.baseline_worktree.resolve())
    candidate = VariantSpec("candidate", args.candidate_worktree.resolve())
    scenarios = build_scenarios(set(args.scenario) if args.scenario else None)

    all_results: dict[str, Any] = {"providers": []}

    for provider in providers:
        print("=" * 100)
        print(f"Provider {provider.label}: {provider.provider}/{provider.model}")
        print("=" * 100)
        baseline_result = run_probe(baseline, provider, scenarios, args.max_tokens)
        candidate_result = run_probe(candidate, provider, scenarios, args.max_tokens)
        print_variant_result(baseline, baseline_result)
        print_variant_result(candidate, candidate_result)
        print_delta_summary(baseline_result, candidate_result)
        all_results["providers"].append(
            {
                "label": provider.label,
                "provider": provider.provider,
                "model": provider.model,
                "baseline": baseline_result,
                "candidate": candidate_result,
            }
        )
        print()

    if args.output_json:
        args.output_json.write_text(json.dumps(all_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
