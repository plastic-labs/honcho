"""Public LLM entrypoint: `honcho_llm_call`.

Orchestrates:
- Runtime config resolution from ConfiguredModelSettings → ModelConfig.
- Per-attempt planning (primary vs fallback selection).
- Retry with exponential backoff via tenacity.
- Tool-loop delegation when tools are supplied.
- Single-call delegation to the executor otherwise.
- Reasoning-trace telemetry emission.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, TypeVar, cast, overload

from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import ConfiguredModelSettings, ModelConfig
from src.exceptions import ValidationException
from src.telemetry.logging import conditional_observe
from src.telemetry.reasoning_traces import log_reasoning_trace

from .executor import honcho_llm_call_inner
from .runtime import (
    AttemptPlan,
    current_attempt,
    effective_temperature,
    plan_attempt,
    resolve_runtime_model_config,
)
from .tool_loop import execute_tool_loop
from .types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    IterationCallback,
    ReasoningEffortType,
    StreamingResponseWithMetadata,
)

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


@overload
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[M],
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    stream_final_only: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
    trace_name: str | None = None,
    iteration_callback: IterationCallback | None = None,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    stream_final_only: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
    trace_name: str | None = None,
    iteration_callback: IterationCallback | None = None,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[True] = ...,
    stream_final_only: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
    trace_name: str | None = None,
    iteration_callback: IterationCallback | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk] | StreamingResponseWithMetadata: ...


@conditional_observe(name="LLM Call")
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: bool = False,
    stream_final_only: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
    trace_name: str | None = None,
    iteration_callback: IterationCallback | None = None,
) -> (
    HonchoLLMCallResponse[Any]
    | AsyncIterator[HonchoLLMCallStreamChunk]
    | StreamingResponseWithMetadata
):
    """Make an LLM call with retry, optional backup failover, and optional tool loop.

    Backup provider/model (if configured on the primary ModelConfig's
    `fallback`) is used on the final retry attempt, which is 3 by default.

    Raises:
        ValidationException: If streaming and tool calling are combined
                             without `stream_final_only=True`.
    """
    runtime_model_config = resolve_runtime_model_config(model_config)

    # Caller kwargs left at None are resolved downstream by
    # effective_config_for_call against whichever ModelConfig wins the
    # attempt (primary or fallback). Defaulting here from
    # runtime_model_config would clobber a fallback config's own
    # temperature/thinking params on the final retry, so we deliberately
    # keep the locals as the caller supplied them.

    if stream and tools and not stream_final_only:
        raise ValidationException(
            "Streaming is not supported with tool calling. "
            + "Set stream=False when using tools, or use stream_final_only=True "
            + "to stream only the final response after tool calls."
        )

    # tenacity uses 1-indexed attempts.
    current_attempt.set(1)

    def _get_attempt_plan() -> AttemptPlan:
        return plan_attempt(
            runtime_model_config=runtime_model_config,
            attempt=current_attempt.get(),
            retry_attempts=retry_attempts,
            call_thinking_budget_tokens=thinking_budget_tokens,
            call_reasoning_effort=reasoning_effort,
        )

    async def _call_with_provider_selection() -> (
        HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]
    ):
        """Select provider/model based on current attempt, then call once.

        This closure is what tenacity wraps, so selection re-runs per attempt
        (and the fallback kicks in on the final attempt automatically).
        """
        plan = _get_attempt_plan()

        if stream:
            return await honcho_llm_call_inner(
                plan.provider,
                plan.model,
                prompt,
                max_tokens,
                response_model,
                json_mode,
                effective_temperature(temperature),
                stop_seqs,
                plan.reasoning_effort,
                verbosity,
                plan.thinking_budget_tokens,
                stream=True,
                client_override=plan.client,
                tools=tools,
                tool_choice=tool_choice,
                selected_config=plan.selected_config,
            )
        return await honcho_llm_call_inner(
            plan.provider,
            plan.model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            effective_temperature(temperature),
            stop_seqs,
            plan.reasoning_effort,
            verbosity,
            plan.thinking_budget_tokens,
            stream=False,
            client_override=plan.client,
            tools=tools,
            tool_choice=tool_choice,
            selected_config=plan.selected_config,
        )

    decorated = _call_with_provider_selection

    if track_name:
        decorated = ai_track(track_name)(decorated)

    def before_retry_callback(retry_state: Any) -> None:
        """Update attempt counter before each retry + log transient failures.

        tenacity's before_sleep fires AFTER an attempt fails, BEFORE sleeping,
        so we increment to the next attempt number here.
        """
        next_attempt = retry_state.attempt_number + 1
        current_attempt.set(next_attempt)
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc:
            logger.warning(
                f"Error on attempt {retry_state.attempt_number}/{retry_attempts} with "
                + f"{runtime_model_config.transport}/{runtime_model_config.model}: {exc}"
            )
            logger.info(f"Will retry with attempt {next_attempt}/{retry_attempts}")

    if enable_retry:
        decorated = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
        )(decorated)

    def _trace_thinking_budget() -> int | None:
        # Trace log should reflect what got applied, so fall back to the
        # runtime config's value when the caller left the kwarg unset.
        return (
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else runtime_model_config.thinking_budget_tokens
        )

    def _trace_reasoning_effort() -> ReasoningEffortType:
        if reasoning_effort is not None:
            return reasoning_effort
        config_effort = runtime_model_config.thinking_effort
        return cast(ReasoningEffortType, config_effort) if config_effort else None

    def _trace_stop_seqs() -> list[str] | None:
        return (
            stop_seqs if stop_seqs is not None else runtime_model_config.stop_sequences
        )

    # Tool-less path: call once and return.
    if not tools or not tool_executor:
        result: (
            HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]
        ) = await decorated()
        if trace_name and isinstance(result, HonchoLLMCallResponse):
            log_reasoning_trace(
                task_type=trace_name,
                model_config=runtime_model_config,
                prompt=prompt,
                response=result,
                max_tokens=max_tokens,
                thinking_budget_tokens=_trace_thinking_budget(),
                reasoning_effort=_trace_reasoning_effort(),
                json_mode=json_mode,
                stop_seqs=_trace_stop_seqs(),
                messages=messages,
            )
        return result

    # execute_tool_loop raises ValidationException on out-of-range
    # max_tool_iterations; fail-fast is cheaper than silent clamping here.
    result = await execute_tool_loop(
        prompt=prompt,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        tool_executor=tool_executor,
        max_tool_iterations=max_tool_iterations,
        response_model=response_model,
        json_mode=json_mode,
        temperature=temperature,
        stop_seqs=stop_seqs,
        verbosity=verbosity,
        enable_retry=enable_retry,
        retry_attempts=retry_attempts,
        max_input_tokens=max_input_tokens,
        get_attempt_plan=_get_attempt_plan,
        before_retry_callback=before_retry_callback,
        stream_final=stream_final_only,
        iteration_callback=iteration_callback,
    )
    if trace_name and isinstance(result, HonchoLLMCallResponse):
        log_reasoning_trace(
            task_type=trace_name,
            model_config=runtime_model_config,
            prompt=prompt,
            response=result,
            max_tokens=max_tokens,
            thinking_budget_tokens=_trace_thinking_budget(),
            reasoning_effort=_trace_reasoning_effort(),
            json_mode=json_mode,
            stop_seqs=_trace_stop_seqs(),
            messages=messages,
        )
    return result


__all__ = ["honcho_llm_call"]
