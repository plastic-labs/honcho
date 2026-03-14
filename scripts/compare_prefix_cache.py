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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHILD_CODE = r"""
import asyncio
import json
import sys
import time

payload = json.loads(sys.stdin.read())

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

PROBE_SUBPROCESS_TIMEOUT_SECONDS = 300

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
    """Provider configuration for one probe run.

    Args:
        label: Human-readable label for reports.
        provider: Provider identifier for `honcho_llm_call_inner`.
        model: Model name to evaluate.
    """

    label: str
    provider: str
    model: str


@dataclass
class VariantSpec:
    """Named worktree variant used in the comparison.

    Args:
        label: Short label for reports.
        worktree: Path to the Honcho worktree to execute in.
    """

    label: str
    worktree: Path


def parse_args() -> argparse.Namespace:
    """Parse probe command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
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
    """Parse ``label=provider:model`` into a provider spec.

    Args:
        raw: Raw provider specification.

    Returns:
        Parsed provider configuration.

    Raises:
        ValueError: If the provider specification does not match the expected format.
    """
    if "=" not in raw or ":" not in raw:
        raise ValueError(
            f"Invalid provider spec {raw!r}. Expected label=provider:model"
        )
    label, provider_model = raw.split("=", 1)
    provider, model = provider_model.split(":", 1)
    if not label:
        raise ValueError(
            f"Invalid provider spec {raw!r}. Label must not be empty."
        )
    if not provider:
        raise ValueError(
            f"Invalid provider spec {raw!r}. Provider must not be empty."
        )
    if not model:
        raise ValueError(
            f"Invalid provider spec {raw!r}. Model must not be empty."
        )
    return ProviderSpec(label=label, provider=provider, model=model)


def add_namespace(namespace: str, content: str) -> str:
    """Prefix prompt content with a cache namespace tag.

    Args:
        namespace: Cache namespace to prepend.
        content: Prompt content to tag.

    Returns:
        Tagged prompt content.
    """
    return f"<cache_namespace>{namespace}</cache_namespace>\n{content}"


def build_messages(
    namespace: str, base_prefix: str, rolling_history: str, user_query: str
) -> list[dict[str, str]]:
    """Build messages for one probe call.

    Args:
        namespace: Cache namespace for this call.
        base_prefix: Stable system prompt content.
        rolling_history: Dynamic history block.
        user_query: User query for the call.

    Returns:
        Provider-agnostic chat messages.
    """
    return [
        {"role": "system", "content": add_namespace(namespace, base_prefix)},
        {"role": "system", "content": add_namespace(namespace, rolling_history)},
        {"role": "user", "content": user_query},
    ]


def build_scenarios(selected: set[str] | None) -> list[dict[str, Any]]:
    """Return scenario definitions, optionally filtered by name.

    Args:
        selected: Optional scenario names to include.

    Returns:
        Scenario definitions in execution order.
    """
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
                "user_query": "What exact launch date should be remembered for the user?",
            },
            "transition": {
                "base_prefix": BASE_PREFIX_VARIANT,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What exact launch date should be remembered for the user?",
            },
            "steady": {
                "base_prefix": BASE_PREFIX_VARIANT,
                "rolling_history": ROLLING_HISTORY_A,
                "user_query": "What exact launch date should be remembered for the user?",
            },
        },
    ]
    if not selected:
        return scenario_defs
    return [scenario for scenario in scenario_defs if scenario["name"] in selected]


def build_cache_namespace(
    variant: VariantSpec,
    provider: ProviderSpec,
    scenario_name: str,
    invocation_salt: str,
) -> str:
    """Build a cache namespace for one invocation.

    Args:
        variant: Worktree variant under test.
        provider: Provider/model pair being evaluated.
        scenario_name: Scenario identifier.
        invocation_salt: Per-process salt generated once at startup.

    Returns:
        Namespace stable within one run and unique across runs.
    """
    return f"{variant.label}:{provider.label}:{scenario_name}:{invocation_salt}"


def build_scenario_payload(
    variant: VariantSpec,
    provider: ProviderSpec,
    scenario: dict[str, Any],
    max_tokens: int,
    invocation_salt: str,
) -> dict[str, Any]:
    """Build the child-process payload for one scenario.

    Args:
        variant: Worktree variant under test.
        provider: Provider/model pair being evaluated.
        scenario: Scenario definition to execute.
        max_tokens: Maximum output tokens to request.
        invocation_salt: Per-process salt generated once at startup.

    Returns:
        Serialized probe payload.
    """
    namespace = build_cache_namespace(
        variant,
        provider,
        scenario["name"],
        invocation_salt,
    )
    calls: list[dict[str, Any]] = []
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
    invocation_salt: str,
) -> dict[str, Any]:
    """Run one scenario probe in a subprocess.

    Args:
        variant: Worktree variant under test.
        provider: Provider/model pair being evaluated.
        scenario: Scenario definition to execute.
        max_tokens: Maximum output tokens to request.
        invocation_salt: Per-process salt generated once at startup.

    Returns:
        Parsed JSON result from the child process.

    Raises:
        RuntimeError: If the child times out, fails, or emits invalid JSON.
    """
    payload = build_scenario_payload(
        variant,
        provider,
        scenario,
        max_tokens,
        invocation_salt,
    )
    try:
        process = subprocess.run(
            [sys.executable, "-c", CHILD_CODE],
            cwd=variant.worktree,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=PROBE_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            (
                f"{variant.label} probe timed out for {provider.label} "
                f"({scenario['name']}) after {PROBE_SUBPROCESS_TIMEOUT_SECONDS} seconds.\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            )
        ) from exc
    if process.returncode != 0:
        raise RuntimeError(
            (
                f"{variant.label} probe failed for {provider.label} "
                f"({scenario['name']}).\n"
                f"stdout:\n{process.stdout}\n"
                f"stderr:\n{process.stderr}"
            )
        )
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            (
                f"{variant.label} probe produced invalid JSON for {provider.label} "
                f"({scenario['name']}).\n"
                f"stdout:\n{process.stdout}\n"
                f"stderr:\n{process.stderr}"
            )
        ) from exc


def run_probe(
    variant: VariantSpec,
    provider: ProviderSpec,
    scenarios: list[dict[str, Any]],
    max_tokens: int,
    invocation_salt: str,
) -> dict[str, Any]:
    """Run all scenarios for one worktree/provider pair.

    Args:
        variant: Worktree variant under test.
        provider: Provider/model pair being evaluated.
        scenarios: Scenario definitions to execute.
        max_tokens: Maximum output tokens to request.
        invocation_salt: Per-process salt generated once at startup.

    Returns:
        Aggregated results for the variant/provider pair.
    """
    return {
        "scenarios": [
            run_scenario_probe(
                variant,
                provider,
                scenario,
                max_tokens,
                invocation_salt,
            )
            for scenario in scenarios
        ]
    }


def format_metric(value: Any) -> str:
    """Render a metric in stable human-readable form.

    Args:
        value: Metric value to render.

    Returns:
        String representation of the metric.
    """
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def cache_ratio_pct(call: dict[str, Any]) -> float:
    """Compute the cached-input percentage for one call.

    Args:
        call: Probe result for one model call.

    Returns:
        Percentage of input tokens reported as cache reads.
    """
    input_tokens = call["input_tokens"] or 0
    if input_tokens <= 0:
        return 0.0
    return (call["cache_read_input_tokens"] / input_tokens) * 100


def print_variant_result(variant: VariantSpec, result: dict[str, Any]) -> None:
    """Print per-call cache metrics for one variant.

    Args:
        variant: Worktree variant being reported.
        result: Probe results for the variant.
    """
    print(f"  {variant.label}")
    for scenario in result["scenarios"]:
        print(f"    [{scenario['name']}]")
        for call in scenario["calls"]:
            print(
                (
                    f"      {call['label']:<12} "
                    f"read={format_metric(call['cache_read_input_tokens']):>8} "
                    f"create={format_metric(call['cache_creation_input_tokens']):>8} "
                    f"input={format_metric(call['input_tokens']):>8} "
                    f"cached_pct={cache_ratio_pct(call):>7.1f} "
                    f"ms={format_metric(call['duration_ms']):>8}"
                )
            )


def print_delta_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    """Print candidate-versus-baseline cache and latency deltas.

    Args:
        baseline: Probe results for the baseline worktree.
        candidate: Probe results for the candidate worktree.
    """
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
                (
                    f"    {name}:{label:<11} "
                    f"read_delta={read_delta:+8.2f} "
                    f"create_delta={create_delta:+8.2f} "
                    f"cached_pct_delta={ratio_delta:+7.1f} "
                    f"latency_delta_ms={latency_delta:+8.2f}"
                )
            )


def main() -> None:
    """Run the probe and optionally write JSON output.

    Raises:
        ValueError: If one of the provider specifications is malformed.
        RuntimeError: If any scenario probe fails, times out, or emits invalid JSON.
    """
    args = parse_args()
    invocation_salt = uuid.uuid4().hex
    providers = [parse_provider_spec(raw) for raw in args.provider]
    baseline = VariantSpec("baseline", args.baseline_worktree.resolve())
    candidate = VariantSpec("candidate", args.candidate_worktree.resolve())
    scenarios = build_scenarios(set(args.scenario) if args.scenario else None)

    all_results: dict[str, Any] = {"providers": []}

    for provider in providers:
        print("=" * 100)
        print(f"Provider {provider.label}: {provider.provider}/{provider.model}")
        print("=" * 100)
        baseline_result = run_probe(
            baseline,
            provider,
            scenarios,
            args.max_tokens,
            invocation_salt,
        )
        candidate_result = run_probe(
            candidate,
            provider,
            scenarios,
            args.max_tokens,
            invocation_salt,
        )
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
