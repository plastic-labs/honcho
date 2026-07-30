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
from src.telemetry.reasoning_traces import log_reasoning_trace

from .executor import honcho_llm_call_inner
from .runtime import (
    AttemptPlan,
    current_attempt,
    effective_temperature,
    plan_attempt,
    resolve_runtime_model_config,
    start_langfuse_agent_run,
)
from .tool_loop import execute_tool_loop
from .types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    IterationCallback,
    LLMTelemetryContext,
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
    telemetry: LLMTelemetryContext | None = None,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
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
    telemetry: LLMTelemetryContext | None = None,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
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
    telemetry: LLMTelemetryContext | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk] | StreamingResponseWithMetadata: ...


async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
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
    telemetry: LLMTelemetryContext | None = None,
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
        plan = plan_attempt(
            runtime_model_config=runtime_model_config,
            attempt=current_attempt.get(),
            retry_attempts=retry_attempts,
            call_thinking_budget_tokens=thinking_budget_tokens,
            call_reasoning_effort=reasoning_effort,
        )
        return plan

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
                plan=plan,
                telemetry=telemetry,
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
            plan=plan,
            telemetry=telemetry,
        )

    decorated = _call_with_provider_selection

    sentry_track_name = telemetry.track_name if telemetry is not None else None
    if sentry_track_name:
        decorated = ai_track(sentry_track_name)(decorated)

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
        # enforce `max_input_tokens` for tool-less calls too. Before
        # this change, only `execute_tool_loop` consumed the kwarg — the
        # deriver passed it but it was silently dropped, so the cap-hit
        # signal it needed for RepresentationCompletedEvent could not be
        # measured. Now we run the same message-list truncation helper
        # and surface a `hit_input_token_cap` boolean on the response.
        #
        # The signal is purely token-based ("did the input exceed cap?")
        # rather than message-count-based — the helper deliberately keeps
        # the last conversation unit even when it's oversized (see
        # truncate_messages_to_fit), so a single-message over-cap input
        # (the deriver's prompt-only case) would otherwise silently fly
        # through with hit=False. Token-based comparison catches it.
        toolless_hit_input_token_cap = False
        toolless_messages = messages
        if max_input_tokens is not None:
            from .conversation import count_message_tokens, truncate_messages_to_fit

            base_messages = messages or [{"role": "user", "content": prompt}]
            toolless_hit_input_token_cap = (
                count_message_tokens(base_messages) > max_input_tokens
            )
            toolless_messages = truncate_messages_to_fit(
                base_messages, max_input_tokens
            )

        # Re-bind the closure to use the truncated message list.
        if toolless_messages is not None:
            captured_messages = toolless_messages

            async def _toolless_call() -> (
                HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]
            ):
                plan = _get_attempt_plan()
                # Branch on stream so each call site lands on the right
                # `Literal[True]/False` overload — basedpyright won't infer
                # which overload a runtime `bool` matches.
                if stream:
                    return await honcho_llm_call_inner(
                        plan.provider,
                        plan.model,
                        prompt,
                        max_tokens,
                        response_model=response_model,
                        json_mode=json_mode,
                        temperature=effective_temperature(temperature),
                        stop_seqs=stop_seqs,
                        reasoning_effort=plan.reasoning_effort,
                        verbosity=verbosity,
                        thinking_budget_tokens=plan.thinking_budget_tokens,
                        stream=True,
                        client_override=plan.client,
                        tools=tools,
                        tool_choice=tool_choice,
                        selected_config=plan.selected_config,
                        plan=plan,
                        telemetry=telemetry,
                        messages=captured_messages,
                    )
                return await honcho_llm_call_inner(
                    plan.provider,
                    plan.model,
                    prompt,
                    max_tokens,
                    response_model=response_model,
                    json_mode=json_mode,
                    temperature=effective_temperature(temperature),
                    stop_seqs=stop_seqs,
                    reasoning_effort=plan.reasoning_effort,
                    verbosity=verbosity,
                    thinking_budget_tokens=plan.thinking_budget_tokens,
                    stream=False,
                    client_override=plan.client,
                    tools=tools,
                    tool_choice=tool_choice,
                    selected_config=plan.selected_config,
                    plan=plan,
                    telemetry=telemetry,
                    messages=captured_messages,
                )

            wrapped = _toolless_call
            if sentry_track_name:
                wrapped = ai_track(sentry_track_name)(wrapped)
            if enable_retry:
                wrapped = retry(
                    stop=stop_after_attempt(retry_attempts),
                    wait=wait_exponential(multiplier=1, min=4, max=10),
                    before_sleep=before_retry_callback,
                )(wrapped)
            result: (
                HonchoLLMCallResponse[Any]
                | AsyncIterator[HonchoLLMCallStreamChunk]
                | StreamingResponseWithMetadata
            ) = await wrapped()
        else:
            result = await decorated()

        if toolless_hit_input_token_cap and isinstance(result, HonchoLLMCallResponse):
            result.hit_input_token_cap = True

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

    # One run-level Langfuse trace wraps the whole run; step/LLM/tool spans
    # nest under it (the run handle keeps `start_as_current_observation` open
    # via ExitStack, so the run span stays current OTel-wise even though we
    # never use a `with` block here). The handle is passed into
    # `execute_tool_loop` so streaming results own it from construction and
    # close the span after drain — that's how the streamed text shows up as
    # the trace's output instead of blank. Non-streaming results: we end in
    # the `finally`.
    run_label = (telemetry.track_name if telemetry else None) or "Agent"
    run_handle = start_langfuse_agent_run(run_label, telemetry)
    if run_handle is not None:
        # Mirror execute_tool_loop's prompt-only handling: when messages is
        # omitted it seeds the conversation with a single user message built
        # from prompt. Record that same effective input so the run span isn't
        # blank for prompt-only calls.
        run_handle.update(
            input=messages if messages else [{"role": "user", "content": prompt}]
        )
    try:
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
            telemetry=telemetry,
            langfuse_run_handle=run_handle,
        )
    except BaseException:
        if run_handle is not None:
            run_handle.end()
        raise
    # Streaming wrapper owns the handle and closes it after drain;
    # non-streaming paths (always a HonchoLLMCallResponse here) close it now
    # with the final content as output.
    if run_handle is not None and isinstance(result, HonchoLLMCallResponse):
        run_handle.end(output=result.content)
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
