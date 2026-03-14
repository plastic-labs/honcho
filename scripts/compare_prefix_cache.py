#!/usr/bin/env python3
"""
Compare prompt-prefix cache behavior between two Honcho worktrees.

This is an eval-only probe. It does not start Honcho servers, touch the database,
 or exercise Hermes/session state. Instead, it imports each worktree's
`src.utils.clients.honcho_llm_call_inner` and runs controlled message patterns
against live providers.

The key scenario is `change_history`: the stable base prefix stays the same while
the rolling context block changes. The candidate branch should preserve more
cache reuse there because it keeps multiple cacheable system blocks separate.
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
    calls = []
    for call in payload["calls"]:
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

    print(json.dumps({"name": payload["name"], "calls": calls}))


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


def add_namespace(namespace: str, content: str) -> str:
    return f"<cache_namespace>{namespace}</cache_namespace>\n{content}"


def build_messages(
    namespace: str, base_prefix: str, rolling_history: str, user_query: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": add_namespace(namespace, base_prefix)},
        {"role": "system", "content": add_namespace(namespace, rolling_history)},
        {"role": "user", "content": user_query},
    ]


def build_scenarios(selected: set[str] | None) -> list[dict[str, Any]]:
    scenario_defs = [
        {
            "name": "repeat_exact",
            "prime": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What is the user's preferred morning drink schedule?",
            },
            "transition": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What is the user's preferred morning drink schedule?",
            },
            "steady": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What is the user's preferred morning drink schedule?",
            },
        },
        {
            "name": "change_user",
            "prime": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What is the user's preferred morning drink schedule?",
            },
            "transition": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What exact launch date should be remembered for the user?",
            },
            "steady": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What exact launch date should be remembered for the user?",
            },
        },
        {
            "name": "change_history",
            "prime": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "Summarize the user's communication preference in one sentence.",
            },
            "transition": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_B,
                "user_query": "Summarize the user's communication preference in one sentence.",
            },
            "steady": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_B,
                "user_query": "Summarize the user's communication preference in one sentence.",
            },
        },
        {
            "name": "change_base",
            "prime": {
                "base_prefix": BASE_PREFIX,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What city is the user considering for a move?",
            },
            "transition": {
                "base_prefix": BASE_PREFIX_VARIANT,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What city is the user considering for a move?",
            },
            "steady": {
                "base_prefix": BASE_PREFIX_VARIANT,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What city is the user considering for a move?",
            },
        },
    ]
    if not selected:
        return scenario_defs
    return [scenario for scenario in scenario_defs if scenario["name"] in selected]


def build_scenario_payload(
    variant: VariantSpec, provider: ProviderSpec, scenario: dict[str, Any], max_tokens: int
) -> dict[str, Any]:
    namespace = f"{variant.label}:{provider.label}:{scenario['name']}"
    calls = []
    for label in ("prime", "transition", "steady"):
        call = scenario[label]
        calls.append(
            {
                "label": label,
                "messages": build_messages(
                    namespace,
                    call["base_prefix"],
                    call["rolling_history"],
                    call["user_query"],
                ),
            }
        )
    return {
        "name": scenario["name"],
        "provider": provider.provider,
        "model": provider.model,
        "max_tokens": max_tokens,
        "calls": calls,
    }


def run_scenario_probe(
    variant: VariantSpec,
    provider: ProviderSpec,
    scenario: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    payload = build_scenario_payload(variant, provider, scenario, max_tokens)
    process = subprocess.run(
        [sys.executable, "-c", CHILD_CODE, json.dumps(payload)],
        cwd=variant.worktree,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{variant.label} probe failed for {provider.label} ({scenario['name']}).\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
    return json.loads(process.stdout)


def run_probe(
    variant: VariantSpec,
    provider: ProviderSpec,
    scenarios: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "scenarios": [
            run_scenario_probe(variant, provider, scenario, max_tokens)
            for scenario in scenarios
        ]
    }


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def cache_ratio_pct(call: dict[str, Any]) -> float:
    input_tokens = call["input_tokens"] or 0
    if input_tokens <= 0:
        return 0.0
    return (call["cache_read_input_tokens"] / input_tokens) * 100


def print_variant_result(variant: VariantSpec, result: dict[str, Any]) -> None:
    print(f"  {variant.label}")
    for scenario in result["scenarios"]:
        print(f"    [{scenario['name']}]")
        for call in scenario["calls"]:
            print(
                "      "
                f"{call['label']:<12} "
                f"read={format_metric(call['cache_read_input_tokens']):>8} "
                f"create={format_metric(call['cache_creation_input_tokens']):>8} "
                f"input={format_metric(call['input_tokens']):>8} "
                f"cached_pct={cache_ratio_pct(call):>7.1f} "
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
        base_calls = {call["label"]: call for call in baseline_by_name[name]["calls"]}
        cand_calls = {call["label"]: call for call in candidate_by_name[name]["calls"]}
        for label in ("transition", "steady"):
            base_call = base_calls[label]
            cand_call = cand_calls[label]
            read_delta = (
                cand_call["cache_read_input_tokens"]
                - base_call["cache_read_input_tokens"]
            )
            create_delta = (
                cand_call["cache_creation_input_tokens"]
                - base_call["cache_creation_input_tokens"]
            )
            ratio_delta = cache_ratio_pct(cand_call) - cache_ratio_pct(base_call)
            latency_delta = cand_call["duration_ms"] - base_call["duration_ms"]
            print(
                "    "
                f"{name}:{label:<11} "
                f"read_delta={read_delta:+8.2f} "
                f"create_delta={create_delta:+8.2f} "
                f"cached_pct_delta={ratio_delta:+7.1f} "
                f"latency_delta_ms={latency_delta:+8.2f}"
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
