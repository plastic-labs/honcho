"""Runtime config planning and retry/fallback selection.

Owns:
- Resolution of ConfiguredModelSettings → ModelConfig.
- Per-attempt planning (AttemptPlan) including primary/fallback selection and
  reasoning-effort/thinking-budget resolution.
- Per-call effective config construction (applying caller kwarg overrides onto
  the selected ModelConfig).
- Retry attempt tracking via a ContextVar, plus the temperature-bump heuristic.
- Run/step lifecycle reporting to span-tree exporters (CapturedAgentSpan).
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from src.config import (
    ConfiguredModelSettings,
    ModelConfig,
    ModelTransport,
    resolve_model_config,
)

from .capture import (
    build_captured_span,
    dispatch_captured_span,
    has_span_tree_exporters,
)
from .registry import backend_for_provider, client_for_model_config
from .types import LLMTelemetryContext, ProviderClient, ReasoningEffortType

logger = logging.getLogger(__name__)

# ContextVar tracking the current retry attempt for provider switching.
current_attempt: ContextVar[int] = ContextVar("current_attempt", default=0)


@dataclass
class CapturedAgentSpan:
    """Reports a trace root, run, or step's start and end to span-tree exporters."""

    telemetry: LLMTelemetryContext
    kind: Literal["trace", "run", "step"]
    _ended: bool = field(default=False)

    def end(self, *, output: Any = None, is_error: bool = False) -> None:
        """Report the end (with optional output). Idempotent."""
        if self._ended:
            return
        self._ended = True
        dispatch_captured_span(
            build_captured_span(
                self.telemetry,
                kind=self.kind,
                phase="end",
                time_ns=time.time_ns(),
                output=output,
                is_error=is_error,
            )
        )


def start_captured_span(
    kind: Literal["trace", "run", "step"],
    telemetry: LLMTelemetryContext | None,
    *,
    input: Any = None,  # noqa: A002 - mirrors the observation field name
) -> CapturedAgentSpan | None:
    """Report a start to span-tree exporters. ``None`` when none are registered."""
    if telemetry is None or not telemetry.span_identity():
        return None
    if not has_span_tree_exporters():
        return None
    dispatch_captured_span(
        build_captured_span(
            telemetry, kind=kind, phase="start", time_ns=time.time_ns(), input=input
        )
    )
    return CapturedAgentSpan(telemetry=telemetry, kind=kind)


@dataclass(frozen=True)
class AttemptPlan:
    """Per-attempt plan produced by `plan_attempt`.

    Replaces the old loose tuple-of-six (`ProviderSelection`) with a single
    dataclass. Carries everything the executor / tool loop needs to make one
    backend call without re-resolving configuration mid-call.
    """

    provider: ModelTransport
    model: str
    client: ProviderClient
    thinking_budget_tokens: int | None
    reasoning_effort: ReasoningEffortType
    selected_config: ModelConfig
    attempt: int
    retry_attempts: int
    is_fallback: bool


def resolve_runtime_model_config(
    model_config: ModelConfig | ConfiguredModelSettings,
) -> ModelConfig:
    """Return a runtime ModelConfig, resolving settings-shape inputs if needed."""
    if isinstance(model_config, ModelConfig):
        return model_config
    return resolve_model_config(model_config)


def select_model_config_for_attempt(
    model_config: ModelConfig,
    *,
    attempt: int,
    retry_attempts: int,
) -> ModelConfig:
    """Pick the effective config for this attempt.

    Primary config on all attempts except the last, which swaps to the
    resolved fallback (if any).
    """
    if attempt != retry_attempts or model_config.fallback is None:
        return model_config
    fallback = fallback_model_config(model_config)
    if fallback is None:  # pragma: no cover - guarded above
        return model_config
    return fallback


def fallback_model_config(model_config: ModelConfig) -> ModelConfig | None:
    """The configured fallback as a standalone ModelConfig, or None."""
    fb = model_config.fallback
    if fb is None:
        return None
    return ModelConfig(
        model=fb.model,
        transport=fb.transport,
        fallback=None,
        api_key=fb.api_key,
        base_url=fb.base_url,
        temperature=fb.temperature,
        top_p=fb.top_p,
        top_k=fb.top_k,
        frequency_penalty=fb.frequency_penalty,
        presence_penalty=fb.presence_penalty,
        seed=fb.seed,
        thinking_effort=fb.thinking_effort,
        thinking_budget_tokens=fb.thinking_budget_tokens,
        structured_output_mode=fb.structured_output_mode,
        provider_params=fb.provider_params,
        max_output_tokens=fb.max_output_tokens,
        stop_sequences=fb.stop_sequences,
        cache_policy=fb.cache_policy,
    )


def plan_attempt(
    *,
    runtime_model_config: ModelConfig,
    attempt: int,
    retry_attempts: int,
    call_thinking_budget_tokens: int | None,
    call_reasoning_effort: ReasoningEffortType,
) -> AttemptPlan:
    """Build the AttemptPlan for `attempt`.

    Reasoning params are drawn from the caller when we're still on the
    primary config, and from the fallback config otherwise, so cross-transport
    fallbacks use provider-appropriate params.
    """
    selected = select_model_config_for_attempt(
        runtime_model_config,
        attempt=attempt,
        retry_attempts=retry_attempts,
    )
    provider = selected.transport
    client = client_for_model_config(provider, selected)

    is_primary = selected is runtime_model_config
    attempt_thinking_budget = (
        call_thinking_budget_tokens if is_primary else selected.thinking_budget_tokens
    )
    attempt_reasoning_effort: ReasoningEffortType = (
        call_reasoning_effort if is_primary else selected.thinking_effort
    )

    if attempt == retry_attempts and runtime_model_config.fallback is not None:
        logger.warning(
            f"Final retry attempt {attempt}/{retry_attempts}: switching from "
            + f"{runtime_model_config.transport}/{runtime_model_config.model} to "
            + f"backup {provider}/{selected.model}"
        )

    return AttemptPlan(
        provider=provider,
        model=selected.model,
        client=client,
        thinking_budget_tokens=attempt_thinking_budget,
        reasoning_effort=attempt_reasoning_effort,
        selected_config=selected,
        attempt=attempt,
        retry_attempts=retry_attempts,
        is_fallback=not is_primary,
    )


def plan_pinned_attempt(
    *,
    model_config: ModelConfig,
    attempt: int,
    retry_attempts: int,
    thinking_budget_tokens: int | None,
    reasoning_effort: ReasoningEffortType,
    is_fallback: bool,
) -> AttemptPlan:
    provider = model_config.transport
    return AttemptPlan(
        provider=provider,
        model=model_config.model,
        client=client_for_model_config(provider, model_config),
        thinking_budget_tokens=thinking_budget_tokens,
        reasoning_effort=reasoning_effort,
        selected_config=model_config,
        attempt=attempt,
        retry_attempts=retry_attempts,
        is_fallback=is_fallback,
    )


def effective_config_for_call(
    *,
    selected_config: ModelConfig | None,
    provider: ModelTransport,
    model: str,
    temperature: float | None,
    stop_seqs: list[str] | None,
    thinking_budget_tokens: int | None,
    reasoning_effort: ReasoningEffortType,
) -> ModelConfig:
    """Build the ModelConfig passed to the executor / request_builder.

    Per-call kwargs (temperature, stop_seqs, thinking_*) win when set; otherwise
    the selected_config's values are used. When selected_config is None
    (test-only callers passing provider+model directly) a minimal ModelConfig
    is synthesized.

    max_output_tokens is forced to None so the per-call max_tokens kwarg is
    authoritative — matching historical honcho_llm_call_inner behavior.
    """
    if selected_config is None:
        return ModelConfig(
            model=model,
            transport=provider,
            temperature=temperature,
            stop_sequences=stop_seqs,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=reasoning_effort,
        )
    updates: dict[str, Any] = {"max_output_tokens": None}
    if temperature is not None:
        updates["temperature"] = temperature
    if stop_seqs is not None:
        updates["stop_sequences"] = stop_seqs
    if thinking_budget_tokens is not None:
        updates["thinking_budget_tokens"] = thinking_budget_tokens
    if reasoning_effort is not None:
        updates["thinking_effort"] = reasoning_effort
    return selected_config.model_copy(update=updates)


def effective_temperature(temperature: float | None) -> float | None:
    """Bump temperature from 0.0 → 0.2 on retry attempts for variety."""
    if temperature == 0.0 and current_attempt.get() > 1:
        logger.debug("Bumping temperature from 0.0 to 0.2 on retry")
        return 0.2
    return temperature


def resolve_backend_for_plan(plan: AttemptPlan) -> Any:
    """Convenience helper: plan → ready-to-call ProviderBackend."""
    return backend_for_provider(plan.provider, plan.client)


__all__ = [
    "AttemptPlan",
    "CapturedAgentSpan",
    "current_attempt",
    "effective_config_for_call",
    "effective_temperature",
    "fallback_model_config",
    "plan_attempt",
    "plan_pinned_attempt",
    "resolve_backend_for_plan",
    "resolve_runtime_model_config",
    "select_model_config_for_attempt",
    "start_captured_span",
]
