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

from src.config import ModelConfig, ModelTransport

from .backend import CompletionResult as BackendCompletionResult
from .backend import StreamChunk as BackendStreamChunk
from .backend import ToolCallResult
from .registry import CLIENTS, backend_for_provider
from .request_builder import execute_completion, execute_stream
from .runtime import AttemptPlan, effective_config_for_call
from .types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    LLMTelemetryContext,
    ProviderClient,
    ReasoningEffortType,
)

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


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
    result = {
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
                provider_label=_infer_provider_label(provider, model, plan),
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


def _infer_provider_label(
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


def completion_result_to_response(
    result: BackendCompletionResult,
) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content=result.content,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
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
    # json_mode + verbosity are per-call transport toggles, not ModelConfig
    # knobs — they pass through extra_params. execute_completion merges
    # build_config_extra_params(effective_config) on top for top_p/seed/etc.
    call_extras: dict[str, Any] = {"json_mode": json_mode, "verbosity": verbosity}

    # Propagate Langfuse session_id for trace grouping via LiteLLM proxy.
    # When set, backends include it in the request metadata so LiteLLM's
    # Langfuse callback groups all calls from the same agent operation.
    if (
        telemetry
        and telemetry.langfuse_session_id
        and isinstance(telemetry.langfuse_session_id, str)
    ):
        call_extras["langfuse_session_id"] = telemetry.langfuse_session_id

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
        return completion_result_to_response(backend_result)
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


__all__ = [
    "completion_result_to_response",
    "honcho_llm_call_inner",
    "stream_chunk_to_response_chunk",
]
