"""Runtime config planning and retry/fallback selection.

Owns:
- Resolution of ConfiguredModelSettings → ModelConfig.
- Per-attempt planning (AttemptPlan) including primary/fallback selection and
  reasoning-effort/thinking-budget resolution.
- Per-call effective config construction (applying caller kwarg overrides onto
  the selected ModelConfig).
- Retry attempt tracking via a ContextVar, plus the temperature-bump heuristic.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from src.config import (
    ConfiguredModelSettings,
    ModelConfig,
    ModelTransport,
    resolve_model_config,
    settings,
)

from .registry import backend_for_provider, client_for_model_config
from .types import LLMTelemetryContext, ProviderClient, ReasoningEffortType

logger = logging.getLogger(__name__)

# ContextVar tracking the current retry attempt for provider switching.
current_attempt: ContextVar[int] = ContextVar("current_attempt", default=0)

# True while a `LangfuseAgentRun` handle is live (start → end). Set by
# `start_langfuse_agent_run`, reset by `LangfuseAgentRun.end`. Used by
# `annotate_current_langfuse_trace` to decide whether the current generation
# is the trace root (single-shot callers like the deriver — stamp trace attrs)
# or nested under an active run (multi-turn / streaming — skip trace attrs;
# the run span already carries them via `propagate_attributes`).
_in_agent_run: ContextVar[bool] = ContextVar("_in_agent_run", default=False)


def annotate_current_langfuse_trace(
    provider: ModelTransport,
    model: str,
    *,
    telemetry: LLMTelemetryContext | None = None,
) -> None:
    """Stamp provider/model + step metadata on the current Langfuse generation.

    Inside an active agent run, `propagate_attributes` already stamped
    user_id/session_id/trace_name on the run span; this call only needs to
    decorate the per-iteration generation. Outside a run (single-shot
    callers — deriver, summarizer), this generation IS the trace root, so we
    also stamp the trace attrs.

    Note: `model`/`metadata` are set on every call regardless of `inside_run`
    so multi-turn iterations no longer lose provider/model attribution.
    """
    if not settings.LANGFUSE_PUBLIC_KEY:
        return

    try:
        from langfuse import get_client, propagate_attributes

        inside_run = _in_agent_run.get()
        gen_metadata = _step_metadata(telemetry) if telemetry is not None else {}
        gen_metadata["provider"] = str(provider)
        gen_metadata["model"] = str(model)
        gen_name = (
            f"{telemetry.track_name} LLM call"
            if telemetry is not None and telemetry.track_name
            else None
        )

        if not inside_run:
            run_id = telemetry.run_id if telemetry is not None else None
            trace_name = telemetry.track_name if telemetry is not None else None
            trace_metadata: dict[str, str] = dict(gen_metadata)
            if telemetry is None:
                trace_metadata.setdefault("namespace", str(settings.NAMESPACE))
            # Empty body is intentional: propagate_attributes stamps the active
            # @observe generation (this trace root, for single-shot callers) at
            # __enter__; there are no child spans to scope here. Don't delete as
            # dead code — the enter-time side effect is the point.
            with propagate_attributes(
                user_id=str(settings.NAMESPACE),
                session_id=run_id,
                trace_name=trace_name,
                metadata=trace_metadata,
            ):
                pass

        get_client().update_current_generation(
            name=gen_name,
            model=str(model),
            metadata=gen_metadata,
        )
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to update Langfuse trace metadata: %s", exc)


def _base_metadata(telemetry: LLMTelemetryContext) -> dict[str, str]:
    """Static routing/attribution metadata (everything except ``iteration``).

    Rebuilt per run (cheap); callers that need ``iteration`` copy and add it.
    """
    metadata: dict[str, str] = {"namespace": str(settings.NAMESPACE)}
    for key, value in (
        ("workspace_name", telemetry.workspace_name),
        ("call_purpose", telemetry.call_purpose),
        ("agent_type", telemetry.agent_type),
        ("observer", telemetry.observer),
        ("observed", telemetry.observed),
        ("peer_name", telemetry.peer_name),
    ):
        if value is not None:
            metadata[key] = str(value)
    return metadata


def _step_metadata(
    telemetry: LLMTelemetryContext,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Per-step metadata: ``base`` (or freshly computed) plus ``iteration``."""
    metadata = dict(base) if base is not None else _base_metadata(telemetry)
    if telemetry.iteration is not None:
        metadata["iteration"] = str(telemetry.iteration)
    return metadata


@dataclass
class LangfuseAgentRun:
    """Imperative handle for the run-level Langfuse span.

    Owns an ``ExitStack`` that keeps ``start_as_current_observation`` and
    ``propagate_attributes`` open until ``.end()``. This lets the run span
    outlive the function that created it — streaming flows transfer the
    handle to the response wrapper, which calls ``.end(output=...)`` after
    the stream drains. While the handle is alive, the run span is the
    current OTel observation, so step spans and auto-instrumented LLM
    generations nest under it without any ContextVar choreography.

    Use ``start_langfuse_agent_run`` to construct; never instantiate directly.
    """

    span: Any  # LangfuseSpan; opaque to keep src/llm/ free of langfuse imports.
    _stack: ExitStack
    _run_token: Any
    _ended: bool = field(default=False)

    def update(self, **kwargs: Any) -> None:
        """Set input/output/metadata on the run span (best-effort, no-op if ended)."""
        if self._ended or self.span is None:
            return
        try:
            self.span.update(**kwargs)
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            logger.debug("Failed to update Langfuse run span: %s", exc)

    def end(self, *, output: Any = None) -> None:
        """Stamp final output (optional) and close the run span. Idempotent."""
        if self._ended:
            return
        self._ended = True
        try:
            if self.span is not None and output is not None:
                self.span.update(output=output)
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            logger.debug("Failed to set Langfuse run output: %s", exc)
        try:
            self._stack.close()
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            logger.debug("Failed to close Langfuse run span: %s", exc)
        try:
            _in_agent_run.reset(self._run_token)
        except (ValueError, LookupError) as exc:  # pragma: no cover
            # ContextVar.reset can raise if end() runs in a different async
            # context than start(); telemetry must not fail user code.
            logger.debug("Failed to reset _in_agent_run: %s", exc)


def start_langfuse_agent_run(
    name: str, telemetry: LLMTelemetryContext | None
) -> LangfuseAgentRun | None:
    """Open the one run-level Langfuse trace per agentic run, imperatively.

    Returns ``None`` when Langfuse is disabled or there's no ``run_id``
    (single-shot callers — those self-stamp via
    ``annotate_current_langfuse_trace``). When non-None, the caller MUST
    eventually call ``.end()`` — typically in a ``finally`` block, or by
    transferring ownership to the streaming wrapper.
    """
    if not settings.LANGFUSE_PUBLIC_KEY or telemetry is None or not telemetry.run_id:
        return None
    stack = ExitStack()
    try:
        from langfuse import get_client, propagate_attributes

        span = stack.enter_context(
            get_client().start_as_current_observation(as_type="span", name=name)
        )
        stack.enter_context(
            propagate_attributes(
                user_id=str(settings.NAMESPACE),
                session_id=telemetry.run_id,
                trace_name=name,
                metadata=_base_metadata(telemetry),
            )
        )
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to open Langfuse agent run: %s", exc)
        stack.close()
        return None

    run_token = _in_agent_run.set(True)
    return LangfuseAgentRun(span=span, _stack=stack, _run_token=run_token)


@dataclass
class LangfuseAgentStep:
    """Imperative handle for a per-iteration step span under the run root.

    Owns an ``ExitStack`` holding ``start_as_current_observation`` open until
    ``.end()``. While alive the step span is the current OTel observation,
    so the LLM generation (auto-instrumented or otherwise) nests under it.
    No trace attrs (the run root carries them); just the per-step
    ``iteration`` metadata.
    """

    span: Any
    _stack: ExitStack
    _ended: bool = field(default=False)

    def update(self, **kwargs: Any) -> None:
        """Set input/output/metadata on the step span (best-effort, no-op if ended)."""
        if self._ended or self.span is None:
            return
        try:
            self.span.update(**kwargs)
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            logger.debug("Failed to update Langfuse step span: %s", exc)

    def annotate_io(
        self,
        messages: list[dict[str, Any]],
        content: Any,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Stamp this turn's messages-in / content-or-tool-summary-out.

        On a tool-calling turn the model returns no text yet, so we summarize
        the tool calls for the step output preview; otherwise the assistant
        text is used.
        """
        if self._ended or self.span is None:
            return
        if isinstance(content, str) and content.strip():
            output: Any = content
        elif tool_calls:
            output = {"tool_calls": [tc.get("name") for tc in tool_calls]}
        else:
            output = content
        self.update(input=messages, output=output)

    def end(self, *, output: Any = None) -> None:
        """Stamp final output (optional) and close the step span. Idempotent."""
        if self._ended:
            return
        self._ended = True
        try:
            if self.span is not None and output is not None:
                self.span.update(output=output)
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            logger.debug("Failed to set Langfuse step output: %s", exc)
        try:
            self._stack.close()
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            logger.debug("Failed to close Langfuse step span: %s", exc)


def start_langfuse_agent_step(
    name: str, telemetry: LLMTelemetryContext | None
) -> LangfuseAgentStep | None:
    """Open a per-iteration step span, imperatively. Returns ``None`` when
    Langfuse is disabled or there's no ``run_id`` (no agent run to nest under).
    """
    if not settings.LANGFUSE_PUBLIC_KEY or telemetry is None or not telemetry.run_id:
        return None
    stack = ExitStack()
    try:
        from langfuse import get_client

        span = stack.enter_context(
            get_client().start_as_current_observation(
                as_type="span", name=name, metadata=_step_metadata(telemetry)
            )
        )
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to open Langfuse agent step: %s", exc)
        stack.close()
        return None
    return LangfuseAgentStep(span=span, _stack=stack)


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

    fb = model_config.fallback
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
    "LangfuseAgentRun",
    "LangfuseAgentStep",
    "annotate_current_langfuse_trace",
    "current_attempt",
    "effective_config_for_call",
    "effective_temperature",
    "plan_attempt",
    "resolve_backend_for_plan",
    "resolve_runtime_model_config",
    "select_model_config_for_attempt",
    "start_langfuse_agent_run",
    "start_langfuse_agent_step",
]
