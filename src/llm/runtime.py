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

import contextlib
import logging
from collections.abc import Generator
from contextvars import ContextVar
from dataclasses import dataclass
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

# True while inside a run-level trace (opened by `langfuse_agent_run`). Lets a
# nested generation (run owns the trace attrs → stay silent) be told apart from
# one that escaped its run root — e.g. the streamed final answer, drained after
# the run trace closed, which must self-stamp and rejoin the session by run_id.
_in_agent_run: ContextVar[bool] = ContextVar("_in_agent_run", default=False)


def annotate_current_langfuse_trace(
    provider: ModelTransport,
    model: str,
    *,
    telemetry: LLMTelemetryContext | None = None,
) -> None:
    """Annotate the current Langfuse generation (best-effort).

    Inside a run root (run_id + `_in_agent_run`), the run owns the trace attrs,
    so we only rename the generation. Otherwise this generation IS the trace
    root and stamps user_id=NAMESPACE, session_id=run_id, name, metadata —
    run_id is None for single calls (no session), or the escaped streamed-final
    generation's run_id (rejoins the run's session). session_id is the run,
    never the Honcho Session.id.
    """
    if not settings.LANGFUSE_PUBLIC_KEY:
        return

    try:
        from langfuse import get_client, propagate_attributes

        run_id = telemetry.run_id if telemetry is not None else None
        inside_run = run_id is not None and _in_agent_run.get()

        # Only a trace-root generation stamps trace attrs; nested ones inherit
        # them from the run root.
        if not inside_run:
            metadata = (
                _step_metadata(telemetry)
                if telemetry is not None
                else {"namespace": str(settings.NAMESPACE)}
            )
            metadata["provider"] = str(provider)
            metadata["model"] = str(model)
            trace_name = telemetry.track_name if telemetry is not None else None

            # Entering propagate_attributes stamps the active @observe root span;
            # the empty body is intentional (no child spans to wrap here).
            with propagate_attributes(
                user_id=str(settings.NAMESPACE),
                session_id=run_id,
                trace_name=trace_name,
                metadata=metadata,
            ):
                pass

        # Name the generation per agent (e.g. "Dialectic Agent LLM call") for
        # clean by-name cost aggregation.
        if telemetry is not None and telemetry.track_name:
            get_client().update_current_generation(
                name=f"{telemetry.track_name} LLM call"
            )
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to update Langfuse trace metadata: %s", exc)


def _step_metadata(telemetry: LLMTelemetryContext) -> dict[str, str]:
    """Routing/attribution metadata, str-coerced for propagate_attributes."""
    metadata: dict[str, str] = {"namespace": str(settings.NAMESPACE)}
    for key, value in (
        ("workspace_name", telemetry.workspace_name),
        ("call_purpose", telemetry.call_purpose),
        ("agent_type", telemetry.agent_type),
        ("observer", telemetry.observer),
        ("observed", telemetry.observed),
        ("peer_name", telemetry.peer_name),
        ("iteration", telemetry.iteration),
    ):
        if value is not None:
            metadata[key] = str(value)
    return metadata


@contextlib.contextmanager
def langfuse_agent_run(
    name: str, telemetry: LLMTelemetryContext | None
) -> Generator[Any]:
    """Open the one run-level Langfuse trace per agentic run.

    Every nested step span, generation, and tool observation inherits the trace
    attrs set once here: user_id=NAMESPACE, session_id=run_id, trace_name=name.
    session_id is run_id — globally unique, so conflict-free across tenants (the
    Honcho session *name* is not; when multi-turn lands, swap in Session.id to
    group a conversation's runs). Yields the run-root span (None when disabled /
    no run_id) for `annotate_langfuse_run_io`; no-op for single-call agents.
    """
    if not settings.LANGFUSE_PUBLIC_KEY or telemetry is None or not telemetry.run_id:
        yield None
        return

    try:
        from langfuse import get_client, propagate_attributes

        obs_cm = get_client().start_as_current_observation(as_type="span", name=name)
        attr_cm = propagate_attributes(
            user_id=str(settings.NAMESPACE),
            session_id=telemetry.run_id,
            trace_name=name,
            metadata=_step_metadata(telemetry),
        )
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to open Langfuse agent run: %s", exc)
        yield None
        return

    # Nested generations stay silent while set; reset lets a streamed final
    # answer drained afterward self-stamp as its own session-joined trace.
    token = _in_agent_run.set(True)
    try:
        with obs_cm as span, attr_cm:
            yield span
    finally:
        _in_agent_run.reset(token)


@contextlib.contextmanager
def langfuse_agent_step(
    name: str, telemetry: LLMTelemetryContext | None
) -> Generator[None]:
    """Open a per-iteration child span under the run root.

    One reasoning turn (LLM call + the tools it triggers) = one span; the inner
    generation and tool observations nest under it. No trace attrs (the run root
    owns them); just carries the step's own ``iteration`` metadata. No-op
    without ``run_id``.
    """
    if not settings.LANGFUSE_PUBLIC_KEY or telemetry is None or not telemetry.run_id:
        yield
        return

    try:
        from langfuse import get_client

        obs_cm = get_client().start_as_current_observation(
            as_type="span", name=name, metadata=_step_metadata(telemetry)
        )
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to open Langfuse agent step: %s", exc)
        yield
        return

    with obs_cm:
        yield


def annotate_langfuse_run_io(
    run_span: Any,
    *,
    input: Any = None,
    output: Any = None,
) -> None:
    """Stamp the run-root span's input/output (Langfuse derives the trace
    preview from it). Set on the captured span object, not `update_current_span`
    — at the call sites a step child span is the current observation. No-op when
    `run_span` is None; input and output can be set independently.
    """
    if run_span is None:
        return
    try:
        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        if kwargs:
            run_span.update(**kwargs)
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to set Langfuse run I/O: %s", exc)


def annotate_langfuse_step_io(
    telemetry: LLMTelemetryContext | None,
    messages: list[dict[str, Any]],
    content: Any,
    tool_calls: list[dict[str, Any]],
) -> None:
    """Set the step span's input/output (else the span shows blank in the tree).

    ``output`` is the assistant text, or a summary of the tools called on a
    tool-calling turn (which has no text yet). No-op without ``run_id``.
    """
    if not settings.LANGFUSE_PUBLIC_KEY or telemetry is None or not telemetry.run_id:
        return
    try:
        from langfuse import get_client

        if isinstance(content, str) and content.strip():
            output: Any = content
        elif tool_calls:
            output = {"tool_calls": [tc.get("name") for tc in tool_calls]}
        else:
            output = content
        get_client().update_current_span(input=messages, output=output)
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to set Langfuse step I/O: %s", exc)


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
    "annotate_current_langfuse_trace",
    "annotate_langfuse_run_io",
    "annotate_langfuse_step_io",
    "current_attempt",
    "langfuse_agent_run",
    "langfuse_agent_step",
    "effective_config_for_call",
    "effective_temperature",
    "plan_attempt",
    "resolve_backend_for_plan",
    "resolve_runtime_model_config",
    "select_model_config_for_attempt",
]
