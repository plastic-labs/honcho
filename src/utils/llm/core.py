"""
Provider-agnostic LLM call orchestration.

This module provides the primary entry points (`honcho_llm_call`, `honcho_llm_call_inner`)
and encapsulates retry/failover behavior. Provider-specific request/response logic
is delegated to adapters.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Any, Literal, TypeVar, overload

from anthropic import AsyncAnthropic
from google import genai
from groq import AsyncGroq
from openai import AsyncOpenAI
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import LLMComponentSettings
from src.telemetry.logging import conditional_observe
from src.telemetry.reasoning_traces import log_reasoning_trace
from src.utils.llm.adapters import get_adapter
from src.utils.llm.models import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    IterationCallback,
    StreamingResponseWithMetadata,
)
from src.utils.llm.registry import CLIENTS
from src.utils.llm.tool_loop import (
    MAX_TOOL_ITERATIONS,
    MIN_TOOL_ITERATIONS,
    execute_tool_loop,
)
from src.utils.types import SupportedProviders

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

ReasoningEffortType = Literal["low", "medium", "high", "minimal"] | None
VerbosityType = Literal["low", "medium", "high"] | None

_current_attempt: ContextVar[int] = ContextVar("current_attempt", default=0)


def _get_effective_temperature(temperature: float | None) -> float | None:
    """Adjust temperature on retries - bump 0.0 to 0.2 to get different results."""
    if temperature == 0.0 and _current_attempt.get() > 1:
        logger.debug("Bumping temperature from 0.0 to 0.2 on retry")
        return 0.2
    return temperature


async def _stream_final_response(
    *,
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    conversation_messages: list[dict[str, Any]],
    response_model: type[BaseModel] | None,
    json_mode: bool,
    temperature: float | None,
    stop_seqs: list[str] | None,
    reasoning_effort: ReasoningEffortType,
    verbosity: VerbosityType,
    thinking_budget_tokens: int | None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    """
    Stream the final response after tool execution is complete.

    This performs a streaming call without tools using the accumulated conversation
    messages (including tool call results).
    """
    provider = llm_settings.PROVIDER
    model = llm_settings.MODEL

    client = CLIENTS.get(provider)
    if not client:
        raise ValueError(f"Missing client for {provider}")

    stream_response = await honcho_llm_call_inner(
        provider,
        model,
        prompt,
        max_tokens,
        response_model,
        json_mode,
        _get_effective_temperature(temperature),
        stop_seqs,
        reasoning_effort,
        verbosity,
        thinking_budget_tokens,
        True,
        None,
        None,
        conversation_messages,
    )

    async for chunk in stream_response:
        yield chunk


async def handle_streaming_response(
    client: AsyncAnthropic | AsyncOpenAI | genai.Client | AsyncGroq,
    params: dict[str, Any],
    json_mode: bool,
    thinking_budget_tokens: int | None,
    response_model: type[BaseModel] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    """
    Compatibility wrapper for streaming responses.

    The legacy implementation selected behavior based on the concrete SDK client
    type. This wrapper retains the same signature for tests and internal callers.
    """
    if isinstance(client, AsyncAnthropic):
        provider: SupportedProviders = "anthropic"
    elif isinstance(client, genai.Client):
        provider = "google"
    elif isinstance(client, AsyncGroq):
        provider = "groq"
    else:
        provider = "openai"

    adapter = get_adapter(provider)
    stream = adapter.stream(
        client=client,
        provider=provider,
        model=params["model"],
        prompt=params["messages"][0]["content"] if params.get("messages") else "",
        max_tokens=params["max_tokens"],
        messages=params["messages"],
        response_model=response_model,
        json_mode=json_mode,
        temperature=params.get("temperature"),
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    async for chunk in stream:
        yield chunk


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[M],
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    stream: Literal[False] = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    stream: Literal[False] = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    stream: Literal[True] = ...,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]: ...


async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
    thinking_budget_tokens: int | None = None,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]:
    """
    Perform a single provider call (streaming or non-streaming).

    Callers are responsible for converting tools to the provider-specific format.
    """
    client = CLIENTS[provider]

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    adapter = get_adapter(provider)

    if stream:
        return adapter.stream(
            client=client,
            provider=provider,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            messages=messages,
            response_model=response_model,
            json_mode=json_mode,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            thinking_budget_tokens=thinking_budget_tokens,
        )

    return await adapter.call(
        client=client,
        provider=provider,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        messages=messages,
        response_model=response_model,
        json_mode=json_mode,
        temperature=temperature,
        stop_seqs=stop_seqs,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        thinking_budget_tokens=thinking_budget_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )


@overload
async def honcho_llm_call(
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    *,
    response_model: type[M],
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
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
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
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
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
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
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
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
    """
    Make an LLM call with automatic backup provider failover.

    Backup provider/model is used on the final retry attempt (3 by default).
    """
    if stream and tools and not stream_final_only:
        raise ValueError(
            "Streaming is not supported with tool calling. Set stream=False when using tools, "
            + "or use stream_final_only=True to stream only the final response after tool calls."
        )

    _current_attempt.set(1)

    def _get_provider_and_model() -> (
        tuple[SupportedProviders, str, int | None, ReasoningEffortType, VerbosityType]
    ):
        """Get the provider and model to use based on current attempt."""
        attempt = _current_attempt.get()

        provider: SupportedProviders
        model: str
        thinking_budget: int | None
        gpt5_reasoning_effort: ReasoningEffortType
        gpt5_verbosity: VerbosityType

        if (
            attempt == retry_attempts
            and llm_settings.BACKUP_PROVIDER is not None
            and llm_settings.BACKUP_MODEL is not None
            and llm_settings.BACKUP_PROVIDER in CLIENTS
        ):
            provider = llm_settings.BACKUP_PROVIDER
            model = llm_settings.BACKUP_MODEL
            thinking_budget = thinking_budget_tokens
            gpt5_reasoning_effort = reasoning_effort
            gpt5_verbosity = verbosity

            if provider != "anthropic" and thinking_budget:
                logger.warning(
                    "thinking_budget_tokens not supported by %s, ignoring", provider
                )
                thinking_budget = None

            if "gpt-5" not in model and (gpt5_reasoning_effort or gpt5_verbosity):
                logger.warning(
                    "reasoning_effort/verbosity only supported by GPT-5 models, ignoring"
                )
                gpt5_reasoning_effort = None
                gpt5_verbosity = None

            logger.warning(
                "Final retry attempt %s/%s: switching from %s/%s to backup %s/%s",
                attempt,
                retry_attempts,
                llm_settings.PROVIDER,
                llm_settings.MODEL,
                provider,
                model,
            )
        else:
            provider = llm_settings.PROVIDER
            model = llm_settings.MODEL
            thinking_budget = thinking_budget_tokens
            gpt5_reasoning_effort = reasoning_effort
            gpt5_verbosity = verbosity

        return provider, model, thinking_budget, gpt5_reasoning_effort, gpt5_verbosity

    async def _call_with_provider_selection() -> (
        HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]
    ):
        """Select provider/model per attempt and call once."""
        provider, model, thinking_budget, gpt5_reasoning_effort, gpt5_verbosity = (
            _get_provider_and_model()
        )

        client = CLIENTS.get(provider)
        if not client:
            raise ValueError(f"Missing client for {provider}")

        converted_tools = None
        if tools:
            adapter = get_adapter(provider)
            converted_tools = adapter.convert_tools(tools)

        if stream:
            return await honcho_llm_call_inner(
                provider,
                model,
                prompt,
                max_tokens,
                response_model,
                json_mode,
                _get_effective_temperature(temperature),
                stop_seqs,
                gpt5_reasoning_effort,
                gpt5_verbosity,
                thinking_budget,
                True,
                converted_tools,
                tool_choice,
                messages,
            )

        return await honcho_llm_call_inner(
            provider,
            model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            _get_effective_temperature(temperature),
            stop_seqs,
            gpt5_reasoning_effort,
            gpt5_verbosity,
            thinking_budget,
            False,
            converted_tools,
            tool_choice,
            messages,
        )

    decorated = _call_with_provider_selection

    if track_name:
        decorated = ai_track(track_name)(decorated)

    def before_retry_callback(retry_state: Any) -> None:
        """Update attempt counter before each retry."""
        next_attempt = retry_state.attempt_number + 1
        _current_attempt.set(next_attempt)
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc:
            logger.warning(
                "Error on attempt %s/%s with %s/%s: %s",
                retry_state.attempt_number,
                retry_attempts,
                llm_settings.PROVIDER,
                llm_settings.MODEL,
                exc,
            )
            logger.info("Will retry with attempt %s/%s", next_attempt, retry_attempts)

    if enable_retry:
        decorated = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
        )(decorated)

    if not tools or not tool_executor:
        result: (
            HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]
        ) = await decorated()
        if trace_name and isinstance(result, HonchoLLMCallResponse):
            log_reasoning_trace(
                task_type=trace_name,
                llm_settings=llm_settings,
                prompt=prompt,
                response=result,
                max_tokens=max_tokens,
                thinking_budget_tokens=thinking_budget_tokens,
                reasoning_effort=reasoning_effort,
                json_mode=json_mode,
                stop_seqs=stop_seqs,
                messages=messages,
            )
        return result

    clamped_iterations = max(
        MIN_TOOL_ITERATIONS, min(max_tool_iterations, MAX_TOOL_ITERATIONS)
    )
    if clamped_iterations != max_tool_iterations:
        logger.warning(
            "max_tool_iterations %s clamped to %s (valid range: %s-%s)",
            max_tool_iterations,
            clamped_iterations,
            MIN_TOOL_ITERATIONS,
            MAX_TOOL_ITERATIONS,
        )

    def _set_attempt(attempt: int) -> None:
        """Set the current retry attempt counter."""
        _current_attempt.set(attempt)

    result = await execute_tool_loop(
        llm_settings=llm_settings,
        prompt=prompt,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        tool_executor=tool_executor,
        max_tool_iterations=clamped_iterations,
        response_model=response_model,
        json_mode=json_mode,
        temperature=temperature,
        stop_seqs=stop_seqs,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_retry=enable_retry,
        retry_attempts=retry_attempts,
        max_input_tokens=max_input_tokens,
        get_provider_and_model=_get_provider_and_model,
        before_retry_callback=before_retry_callback,
        get_effective_temperature=_get_effective_temperature,
        set_attempt=_set_attempt,
        call_inner=honcho_llm_call_inner,
        stream_final_response=_stream_final_response,
        stream_final=stream_final_only,
        iteration_callback=iteration_callback,
    )

    if trace_name and isinstance(result, HonchoLLMCallResponse):
        log_reasoning_trace(
            task_type=trace_name,
            llm_settings=llm_settings,
            prompt=prompt,
            response=result,
            max_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
            reasoning_effort=reasoning_effort,
            json_mode=json_mode,
            stop_seqs=stop_seqs,
            messages=messages,
        )
    return result
