"""Single-call executor: the inner LLM-call path without tool-loop orchestration.

`honcho_llm_call_inner` handles one backend call (complete or stream), building
the effective ModelConfig and delegating to request_builder. Result / stream
chunk types are bridged to the public Honcho* shapes here.

Used by:
- src/llm/api.py (the public entrypoint, for both tool-less and tool-enabled paths)
- src/llm/tool_loop.py (each iteration of the tool loop calls this)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Literal, TypeVar, overload

from pydantic import BaseModel

from src.config import ModelConfig, ModelTransport, settings
from src.telemetry.logging import conditional_observe

from .backend import CompletionResult as BackendCompletionResult
from .backend import StreamChunk as BackendStreamChunk
from .backend import ToolCallResult
from .capture import build_captured_call, dispatch_captured_call, has_exporters
from .registry import CLIENTS, backend_for_provider
from .request_builder import execute_completion, execute_stream
from .runtime import (
    AttemptPlan,
    annotate_current_generation_io,
    annotate_current_langfuse_trace,
    effective_config_for_call,
)
from .types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    LLMTelemetryContext,
    ProviderClient,
    ReasoningEffortType,
)

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


# ModelConfig fields that must NEVER reach a trace: secrets and nested holders
# of secrets. Everything else on the config is a safe tuning knob and is dumped
# automatically — so new knobs get traced without touching this code. Keep this
# a deny-list (small, stable) rather than an allow-list (drifts with the model).
_UNSAFE_CONFIG_FIELDS = frozenset(
    {
        "api_key",  # provider secret
        "base_url",  # may embed credentials / private host
        "fallback",  # ResolvedFallbackConfig carries its own api_key/base_url
        "provider_params",  # opaque dict; can carry auth headers/keys
    }
)


def _langfuse_model_parameters(
    *,
    max_tokens: int,
    config: ModelConfig,
    json_mode: bool,
    verbosity: str | None,
    stream: bool,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    response_model: type[BaseModel] | None,
) -> dict[str, Any]:
    """Serializable tuning knobs for the Langfuse generation.

    Surfaces everything @observe auto-capture used to show (temperature, tools,
    ...) MINUS the live client and the secret-bearing config fields. We dump the
    resolved `effective_config` and deny-list only `_UNSAFE_CONFIG_FIELDS`, so a
    new ModelConfig knob is traced automatically — no allow-list to keep in sync.
    `mode="json"` coerces enums/sub-models to JSON-safe values. See HONCHO-4HA.
    """
    params: dict[str, Any] = config.model_dump(
        exclude=set(_UNSAFE_CONFIG_FIELDS), exclude_none=True, mode="json"
    )
    # Per-call extras that live outside ModelConfig.
    params["max_tokens"] = max_tokens
    params["stream"] = stream
    params["json_mode"] = json_mode
    if verbosity is not None:
        params["verbosity"] = verbosity
    if response_model is not None:
        params["response_format"] = response_model.__name__
    if tools:
        params["tools"] = [
            t.get("name") or t.get("function", {}).get("name") or "unknown"
            for t in tools
        ]
    if tool_choice is not None:
        params["tool_choice"] = (
            tool_choice if isinstance(tool_choice, str) else str(tool_choice)
        )
    return params


def _langfuse_usage_details(response: HonchoLLMCallResponse[Any]) -> dict[str, int]:
    """Token usage duplicated onto the Langfuse generation.

    These counts are also emitted via CloudEvents (LLMCallCompletedEvent), but
    we mirror them here so Langfuse renders per-call tokens + cost natively,
    including Anthropic-style prompt-cache reads/writes. Zero-valued cache keys
    are dropped so non-cached calls stay tidy. Stream calls don't surface token
    totals at this layer, so usage is set only on the non-stream path.
    """
    usage: dict[str, int] = {
        "input": response.input_tokens,
        "output": response.output_tokens,
    }
    if response.cache_read_input_tokens:
        usage["cache_read_input_tokens"] = response.cache_read_input_tokens
    if response.cache_creation_input_tokens:
        usage["cache_creation_input_tokens"] = response.cache_creation_input_tokens
    return usage


def _outcome_from_error(
    err: BaseException | None,
) -> Literal["success", "error", "cancelled"]:
    """Map a finally-block error into the telemetry outcome literal.

    CancelledError is a normal control-flow event (client disconnect, server
    shutdown) — surface it distinctly so it doesn't pollute error-rate alerts.
    """
    if err is None:
        return "success"
    if isinstance(err, asyncio.CancelledError):
        return "cancelled"
    return "error"


def _tool_call_result_to_dict(tool_call: ToolCallResult) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": tool_call.id,
        "name": tool_call.name,
        "input": tool_call.input,
    }
    if tool_call.thought_signature is not None:
        result["thought_signature"] = tool_call.thought_signature
    return result


def _emit_llm_call_completed(
    *,
    plan: AttemptPlan | None,
    telemetry: LLMTelemetryContext | None,
    provider: ModelTransport,
    model: str,
    max_tokens: int,
    duration_ms: float,
    has_tools: bool,
    was_stream: bool,
    outcome: Literal["success", "error", "cancelled"],
    result: BackendCompletionResult | None,
    error: BaseException | None,
) -> None:
    """Build and emit an LLMCallCompletedEvent. Best-effort; swallows errors so
    telemetry failures never bleed into the LLM call path."""
    try:
        from src.telemetry.events import CallPurpose, LLMCallCompletedEvent, emit

        # call_purpose is a string slug on LLMTelemetryContext; validate against
        # the enum here (silent drop on unknown values keeps telemetry resilient).
        call_purpose: CallPurpose | None = None
        if telemetry is not None and telemetry.call_purpose:
            try:
                call_purpose = CallPurpose(telemetry.call_purpose)
            except ValueError:
                logger.debug(
                    "Unknown LLMTelemetryContext.call_purpose=%r; emitting without",
                    telemetry.call_purpose,
                )

        attempt = plan.attempt if plan is not None else 1
        retry_attempts = plan.retry_attempts if plan is not None else 1
        was_fallback = plan.is_fallback if plan is not None else False

        emit(
            LLMCallCompletedEvent(
                workspace_name=(telemetry.workspace_name if telemetry else None),
                call_purpose=call_purpose,
                parent_category=(telemetry.parent_category if telemetry else None),
                transport=provider,
                provider_label=infer_provider_label(provider, model, plan),
                model=model,
                effective_max_output_tokens=max_tokens,
                provider_input_tokens=(result.input_tokens if result else 0),
                provider_output_tokens=(result.output_tokens if result else 0),
                cache_read_tokens=(result.cache_read_input_tokens if result else 0),
                cache_creation_tokens=(
                    result.cache_creation_input_tokens if result else 0
                ),
                finish_reason=(result.finish_reason if result else None),
                outcome=outcome,
                is_final_attempt=(attempt >= retry_attempts),
                error_class=(type(error).__name__ if error else None),
                attempt=attempt,
                retry_attempts=retry_attempts,
                was_fallback=was_fallback,
                duration_ms=duration_ms,
                has_tools=has_tools,
                tool_call_count=(len(result.tool_calls) if result else 0),
                was_stream=was_stream,
                run_id=(telemetry.run_id if telemetry else None),
                iteration=(telemetry.iteration if telemetry else None),
            )
        )
    except Exception:  # pragma: no cover - telemetry must not raise
        logger.debug("Failed to emit LLMCallCompletedEvent", exc_info=True)


def infer_provider_label(
    _transport: ModelTransport, model: str, plan: AttemptPlan | None
) -> str | None:
    """Best-effort vendor inference for relay setups.

    When the model name carries a vendor prefix (OpenRouter convention:
    "anthropic/claude-..." routed through the openai transport), surface that
    as the provider label so analytics can distinguish "openai-the-vendor"
    from "openai-the-transport-pointing-at-openrouter".

    `_transport` is currently unused but kept on the signature so callers stay
    explicit about which transport produced the call — future inference rules
    (e.g. anthropic-direct vs anthropic-via-relay) may need it.
    """
    if "/" in model:
        return model.split("/", 1)[0]
    # Defensive getattr — selected_config may be a stub in tests or a config
    # without an explicit base_url. Either way the inference is best-effort.
    base_url = (
        getattr(plan.selected_config, "base_url", None) if plan is not None else None
    )
    if base_url and "openrouter" in base_url.lower():
        return "openrouter"
    return None


def _maybe_dispatch_capture(
    *,
    plan: AttemptPlan | None,
    telemetry: LLMTelemetryContext | None,
    provider: ModelTransport,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    result: BackendCompletionResult | None,
    error: BaseException | None,
) -> None:
    """Build a CapturedLLMCall and fan it out to registered exporters.

    No-op when payload capture is off
    `has_exporters()` is checked BEFORE building.
    Best-effort: never raises into the call path.
    """
    if not has_exporters():
        return
    try:
        outcome = _outcome_from_error(error)
        finish_reason = result.finish_reason if result is not None else outcome
        dispatch_captured_call(
            build_captured_call(
                telemetry=telemetry,
                transport=str(provider),
                provider_label=infer_provider_label(provider, model, plan),
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                result=result,
                attempt=plan.attempt if plan is not None else 1,
                was_fallback=plan.is_fallback if plan is not None else False,
                was_stream=False,
                finish_reason=finish_reason,
            )
        )
    except Exception:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to dispatch CapturedLLMCall", exc_info=True)


def completion_result_to_response(
    result: BackendCompletionResult,
) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content=result.content,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens or 0,
        cache_creation_input_tokens=result.cache_creation_input_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
        finish_reasons=[result.finish_reason] if result.finish_reason else [],
        tool_calls_made=[_tool_call_result_to_dict(tc) for tc in result.tool_calls],
        thinking_content=result.thinking_content,
        thinking_blocks=result.thinking_blocks,
        reasoning_details=result.reasoning_details,
    )


def stream_chunk_to_response_chunk(
    chunk: BackendStreamChunk,
) -> HonchoLLMCallStreamChunk:
    return HonchoLLMCallStreamChunk(
        content=chunk.content,
        is_done=chunk.is_done,
        finish_reasons=[chunk.finish_reason] if chunk.finish_reason else [],
        output_tokens=chunk.output_tokens,
    )


@overload
async def honcho_llm_call_inner(
    provider: ModelTransport,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[M],
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    stream: Literal[False] = False,
    client_override: ProviderClient | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    selected_config: ModelConfig | None = None,
    plan: AttemptPlan | None = None,
    telemetry: LLMTelemetryContext | None = None,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call_inner(
    provider: ModelTransport,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    stream: Literal[False] = False,
    client_override: ProviderClient | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    selected_config: ModelConfig | None = None,
    plan: AttemptPlan | None = None,
    telemetry: LLMTelemetryContext | None = None,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call_inner(
    provider: ModelTransport,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    stream: Literal[True] = ...,
    client_override: ProviderClient | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    selected_config: ModelConfig | None = None,
    plan: AttemptPlan | None = None,
    telemetry: LLMTelemetryContext | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]: ...


@conditional_observe(
    name="LLM Call",
    as_type="generation",
    # Disable @observe auto-capture: it would serialize `client_override` (a
    # live AsyncOpenAI/genai client) and `selected_config` (carries api_key)
    # into the span. Auto-capture deep-copies the client into a half-built
    # object whose teardown raises `_state`/`_http_options` AttributeErrors
    # (HONCHO-4HA) and leaks the key. We set curated input/output explicitly
    # below via `annotate_current_generation_io`, preserving full fidelity.
    capture_input=False,
    capture_output=False,
)
async def honcho_llm_call_inner(
    provider: ModelTransport,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    stream: bool = False,
    client_override: ProviderClient | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    selected_config: ModelConfig | None = None,
    plan: AttemptPlan | None = None,
    telemetry: LLMTelemetryContext | None = None,
) -> HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]:
    """One backend call. No retry, no fallback, no tool loop.

    This is the Langfuse trace boundary (``@conditional_observe``): every
    provider call is its own trace. Multi-turn agents thread a shared
    ``run_id`` through ``telemetry`` so their per-iteration traces roll up into
    one Langfuse session (see ``annotate_current_langfuse_trace``).

    The outer src/llm/api.py `honcho_llm_call` handles retry + fallback +
    tool orchestration on top of this.

    Emits one LLMCallCompletedEvent per call. On the stream path, setup
    runs inside the awaited coroutine (so it sits inside any outer retry
    wrapper) and emits its own event on failure; the wrapping generator
    emits a second event from its finally block after drain completes or
    raises. `was_stream` is True for streamed calls. Token counts are
    zero on the stream path because provider token totals aren't surfaced
    post-stream at this layer; aggregate envelopes (DialecticCompletedEvent
    etc.) carry the accurate totals.
    """
    client = client_override or CLIENTS.get(provider)
    if client is None:
        raise ValueError(f"Missing client for {provider}")

    # Stamp this trace (user_id/session_id/metadata) now that the @observe
    # span is open and the resolved provider/model are known. Set early so the
    # annotation lands even on the stream path, where the span closes once the
    # generator is returned (before chunks drain).
    annotate_current_langfuse_trace(provider, model, telemetry=telemetry)

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    backend = backend_for_provider(provider, client)

    effective_config = effective_config_for_call(
        selected_config=selected_config,
        provider=provider,
        model=model,
        temperature=temperature,
        stop_seqs=stop_seqs,
        thinking_budget_tokens=thinking_budget_tokens,
        reasoning_effort=reasoning_effort,
    )

    # Explicit generation input + tuning knobs (replaces @observe auto-capture,
    # which would serialize the live client / api key). Set before the stream
    # branch so it lands on the generation span for both paths. Guard on inline
    # mode (matching annotate_current_generation_io's own gate) so we don't
    # build the (model_dump-backed) payload when the helper would no-op — in
    # exporter mode there's no active generation span to stamp.
    if settings.langfuse_inline_enabled:
        annotate_current_generation_io(
            input=messages,
            model_parameters=_langfuse_model_parameters(
                max_tokens=max_tokens,
                config=effective_config,
                json_mode=json_mode,
                verbosity=verbosity,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
                response_model=response_model,
            ),
        )
    # json_mode + verbosity are per-call transport toggles, not ModelConfig
    # knobs — they pass through extra_params. execute_completion merges
    # build_config_extra_params(effective_config) on top for top_p/seed/etc.
    call_extras: dict[str, Any] = {"json_mode": json_mode, "verbosity": verbosity}

    if stream:
        # Stream path: setup must run inside the awaited coroutine so it
        # sits inside the outer retry wrapper (tool_loop.stream_final_response
        # wraps `await honcho_llm_call_inner(stream=True)` with tenacity).
        # If we deferred `execute_stream` into the generator body, a transient
        # setup failure (rate-limit, auth, network) would surface at first
        # iteration — outside retry — and crash the request.
        #
        # Drain failures stay unretried by design (chunks may have already
        # been sent to the client) and report via the wrapper's finally.
        # Token counts are 0 on this path; aggregate envelopes carry totals.
        stream_start = time.perf_counter()
        try:
            stream_iter = await execute_stream(
                backend,
                effective_config,
                messages=messages,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_model,
                cache_policy=effective_config.cache_policy,
                extra_params=call_extras,
            )
        except BaseException as exc:
            _emit_llm_call_completed(
                plan=plan,
                telemetry=telemetry,
                provider=provider,
                model=model,
                max_tokens=max_tokens,
                duration_ms=(time.perf_counter() - stream_start) * 1000,
                has_tools=bool(tools),
                was_stream=True,
                outcome=_outcome_from_error(exc),
                result=None,
                error=exc,
            )
            raise

        async def _wrap_stream() -> AsyncIterator[HonchoLLMCallStreamChunk]:
            stream_error: BaseException | None = None
            try:
                async for chunk in stream_iter:
                    yield stream_chunk_to_response_chunk(chunk)
            except BaseException as exc:
                stream_error = exc
                raise
            finally:
                _emit_llm_call_completed(
                    plan=plan,
                    telemetry=telemetry,
                    provider=provider,
                    model=model,
                    max_tokens=max_tokens,
                    duration_ms=(time.perf_counter() - stream_start) * 1000,
                    has_tools=bool(tools),
                    was_stream=True,
                    outcome=_outcome_from_error(stream_error),
                    result=None,
                    error=stream_error,
                )

        return _wrap_stream()

    start = time.perf_counter()
    backend_result: BackendCompletionResult | None = None
    error: BaseException | None = None
    try:
        backend_result = await execute_completion(
            backend,
            effective_config,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_model,
            cache_policy=effective_config.cache_policy,
            extra_params=call_extras,
        )
        response = completion_result_to_response(backend_result)
        # Explicit generation output + token usage (replaces @observe
        # auto-capture). The stream path closes this span before drain, so its
        # output is stamped on the run-level span instead
        if settings.langfuse_inline_enabled:
            annotate_current_generation_io(
                output=response,
                usage_details=_langfuse_usage_details(response),
            )
        return response
    except BaseException as exc:
        error = exc
        raise
    finally:
        _emit_llm_call_completed(
            plan=plan,
            telemetry=telemetry,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            duration_ms=(time.perf_counter() - start) * 1000,
            has_tools=bool(tools),
            was_stream=False,
            outcome=_outcome_from_error(error),
            result=backend_result,
            error=error,
        )
        _maybe_dispatch_capture(
            plan=plan,
            telemetry=telemetry,
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            result=backend_result,
            error=error,
        )


__all__ = [
    "completion_result_to_response",
    "honcho_llm_call_inner",
    "stream_chunk_to_response_chunk",
]
