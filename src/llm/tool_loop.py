"""Agentic/tool orchestration — the multi-iteration tool execution loop.

`execute_tool_loop` owns:
- initial tool-enabled call
- tool execution
- conversation augmentation with assistant messages + tool results
- max-iteration handling and synthesis call
- stream-final-only mode
- empty-response retry (one retry nudge when the model returns empty content)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import ModelTransport
from src.exceptions import ValidationException
from src.utils.types import set_current_iteration

from .executor import honcho_llm_call_inner
from .registry import history_adapter_for_provider
from .runtime import (
    AttemptPlan,
    current_attempt,
    effective_temperature,
)
from .types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    IterationCallback,
    IterationData,
    StreamingResponseWithMetadata,
    VerbosityType,
)

logger = logging.getLogger(__name__)

# Bounds for max_tool_iterations to prevent runaway loops.
MIN_TOOL_ITERATIONS = 1
MAX_TOOL_ITERATIONS = 100


def format_assistant_tool_message(
    provider: ModelTransport,
    content: Any,
    tool_calls: list[dict[str, Any]],
    thinking_blocks: list[dict[str, Any]] | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Format an assistant message with tool calls in provider-native shape."""
    from .backend import CompletionResult as BackendCompletionResult
    from .backend import ToolCallResult

    adapter = history_adapter_for_provider(provider)
    result = BackendCompletionResult(
        content=content,
        tool_calls=[
            ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                input=tool_call["input"],
                thought_signature=tool_call.get("thought_signature"),
            )
            for tool_call in tool_calls
        ],
        thinking_blocks=thinking_blocks or [],
        reasoning_details=reasoning_details or [],
    )
    return adapter.format_assistant_tool_message(result)


def append_tool_results(
    provider: ModelTransport,
    tool_results: list[dict[str, Any]],
    conversation_messages: list[dict[str, Any]],
) -> None:
    """Append tool results to `conversation_messages` in provider-native shape."""
    adapter = history_adapter_for_provider(provider)
    conversation_messages.extend(adapter.format_tool_results(tool_results))


async def stream_final_response(
    *,
    winning_plan: AttemptPlan,
    prompt: str,
    max_tokens: int,
    conversation_messages: list[dict[str, Any]],
    response_model: type[BaseModel] | None,
    json_mode: bool,
    temperature: float | None,
    stop_seqs: list[str] | None,
    verbosity: VerbosityType,
    enable_retry: bool,
    retry_attempts: int,
    before_retry_callback: Callable[[Any], None],
) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    """Stream the final response after tool execution is complete.

    Uses the AttemptPlan captured at the moment streaming began (typically
    the plan whose inner LLM call just succeeded) and pins it across any
    retries of the stream setup. Re-running provider selection here would
    bleed the outer current_attempt ContextVar into streaming retries,
    potentially rolling the selection back to primary after the tool loop
    had already settled on fallback. Tenacity retries re-issue the same
    streaming call against the same pinned model for transient errors.
    """

    async def _setup_stream() -> AsyncIterator[HonchoLLMCallStreamChunk]:
        return await honcho_llm_call_inner(
            winning_plan.provider,
            winning_plan.model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            effective_temperature(temperature),
            stop_seqs,
            winning_plan.reasoning_effort,
            verbosity,
            winning_plan.thinking_budget_tokens,
            stream=True,
            client_override=winning_plan.client,
            tools=None,
            tool_choice=None,
            messages=conversation_messages,
            selected_config=winning_plan.selected_config,
        )

    if enable_retry:
        wrapped = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
        )(_setup_stream)
        stream = await wrapped()
    else:
        stream = await _setup_stream()

    async for chunk in stream:
        yield chunk


async def execute_tool_loop(
    *,
    prompt: str,
    max_tokens: int,
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None,
    tool_executor: Callable[[str, dict[str, Any]], Any],
    max_tool_iterations: int,
    response_model: type[BaseModel] | None,
    json_mode: bool,
    temperature: float | None,
    stop_seqs: list[str] | None,
    verbosity: VerbosityType,
    enable_retry: bool,
    retry_attempts: int,
    max_input_tokens: int | None,
    get_attempt_plan: Callable[[], AttemptPlan],
    before_retry_callback: Callable[[Any], None],
    stream_final: bool = False,
    iteration_callback: IterationCallback | None = None,
) -> HonchoLLMCallResponse[Any] | StreamingResponseWithMetadata:
    """Run the iterative tool calling loop for agentic LLM interactions.

    Loop per iteration:
      1. Make an LLM call with tools available
      2. Execute any tool calls the LLM requests
      3. Append tool results to the conversation
      4. Repeat until the LLM stops calling tools or max iterations reached

    Returns:
        Final HonchoLLMCallResponse with accumulated token counts and tool call
        history, or a StreamingResponseWithMetadata if stream_final=True.
    """
    from .conversation import truncate_messages_to_fit

    if not MIN_TOOL_ITERATIONS <= max_tool_iterations <= MAX_TOOL_ITERATIONS:
        raise ValidationException(
            "max_tool_iterations must be in "
            + f"[{MIN_TOOL_ITERATIONS}, {MAX_TOOL_ITERATIONS}]; "
            + f"got {max_tool_iterations}"
        )

    conversation_messages: list[dict[str, Any]] = (
        messages.copy() if messages else [{"role": "user", "content": prompt}]
    )

    iteration = 0
    all_tool_calls: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    empty_response_retries = 0
    # Track effective tool_choice — switches from "required"/"any" to "auto" after iter 1.
    effective_tool_choice = tool_choice

    while iteration < max_tool_iterations:
        # Reset attempt counter so each iteration starts with the primary provider.
        current_attempt.set(1)
        logger.debug(f"Tool execution iteration {iteration + 1}/{max_tool_iterations}")

        if max_input_tokens is not None:
            conversation_messages = truncate_messages_to_fit(
                conversation_messages, max_input_tokens
            )

        async def _call_with_messages(
            effective_tool_choice: str | dict[str, Any] | None = effective_tool_choice,
            conversation_messages: list[dict[str, Any]] = conversation_messages,
        ) -> HonchoLLMCallResponse[Any]:
            plan = get_attempt_plan()
            return await honcho_llm_call_inner(
                plan.provider,
                plan.model,
                prompt,  # ignored when messages is passed
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
                tool_choice=effective_tool_choice,
                messages=conversation_messages,
                selected_config=plan.selected_config,
            )

        if enable_retry:
            call_func = retry(
                stop=stop_after_attempt(retry_attempts),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                before_sleep=before_retry_callback,
            )(_call_with_messages)
        else:
            call_func = _call_with_messages

        response = await call_func()

        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens
        total_cache_creation_tokens += response.cache_creation_input_tokens
        total_cache_read_tokens += response.cache_read_input_tokens

        if not response.tool_calls_made:
            logger.debug("No tool calls in response, finishing")

            if (
                isinstance(response.content, str)
                and not response.content.strip()
                and empty_response_retries < 1
                and iteration < max_tool_iterations - 1
            ):
                empty_response_retries += 1
                conversation_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response was empty. Provide a concise answer "
                            "to the original query using the available context."
                        ),
                    }
                )
                iteration += 1
                continue

            if stream_final:
                # Snapshot the plan that just succeeded — streaming retries
                # pin to this exact client/model so we don't bounce back to
                # primary after the tool loop settled on fallback.
                winning_plan = get_attempt_plan()
                stream = stream_final_response(
                    winning_plan=winning_plan,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    conversation_messages=conversation_messages,
                    response_model=response_model,
                    json_mode=json_mode,
                    temperature=temperature,
                    stop_seqs=stop_seqs,
                    verbosity=verbosity,
                    enable_retry=enable_retry,
                    retry_attempts=retry_attempts,
                    before_retry_callback=before_retry_callback,
                )
                return StreamingResponseWithMetadata(
                    stream=stream,
                    tool_calls_made=all_tool_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_creation_input_tokens=total_cache_creation_tokens,
                    cache_read_input_tokens=total_cache_read_tokens,
                    thinking_content=response.thinking_content,
                    iterations=iteration + 1,
                )

            response.tool_calls_made = all_tool_calls
            response.input_tokens = total_input_tokens
            response.output_tokens = total_output_tokens
            response.cache_creation_input_tokens = total_cache_creation_tokens
            response.cache_read_input_tokens = total_cache_read_tokens
            response.iterations = iteration + 1
            return response

        current_provider = get_attempt_plan().provider

        assistant_message = format_assistant_tool_message(
            current_provider,
            response.content,
            response.tool_calls_made,
            response.thinking_blocks,
            response.reasoning_details,
        )
        conversation_messages.append(assistant_message)

        # Telemetry context — 1-indexed iteration.
        set_current_iteration(iteration + 1)

        tool_results: list[dict[str, Any]] = []
        for tool_call in response.tool_calls_made:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            tool_id = tool_call.get("id", "")

            logger.debug(f"Executing tool: {tool_name}")

            try:
                tool_result = await tool_executor(tool_name, tool_input)
                tool_results.append(
                    {
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "result": tool_result,
                    }
                )
                all_tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_result": tool_result,
                    }
                )
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                tool_results.append(
                    {
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "result": f"Error: {str(e)}",
                        "is_error": True,
                    }
                )

        append_tool_results(current_provider, tool_results, conversation_messages)

        if iteration_callback is not None:
            try:
                iteration_data = IterationData(
                    iteration=iteration + 1,
                    tool_calls=[tc["name"] for tc in response.tool_calls_made],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cache_read_tokens=response.cache_read_input_tokens or 0,
                    cache_creation_tokens=response.cache_creation_input_tokens or 0,
                )
                iteration_callback(iteration_data)
            except Exception:
                logger.warning("iteration_callback failed", exc_info=True)

        # After first iteration, switch "required"/"any" → "auto" so the model can stop.
        if iteration == 0 and effective_tool_choice in ("required", "any"):
            effective_tool_choice = "auto"
            logger.debug(
                "Switched tool_choice from 'required'/'any' to 'auto' after first iteration"
            )

        iteration += 1

    logger.warning(
        f"Tool execution loop reached max iterations ({max_tool_iterations})"
    )

    synthesis_prompt = (
        "You have reached the maximum number of tool calls. "
        "Based on all the information you have gathered, provide your final response now. "
        "Do not attempt to call any more tools."
    )
    conversation_messages.append({"role": "user", "content": synthesis_prompt})

    # Truncate again — the per-iteration truncate ran before the last tool
    # call, so appending synthesis_prompt could nudge us back over the cap.
    if max_input_tokens is not None:
        conversation_messages = truncate_messages_to_fit(
            conversation_messages, max_input_tokens
        )

    if stream_final:
        # Snapshot the plan the loop settled on — streaming retries pin to
        # this exact client/model rather than re-running provider selection.
        winning_plan = get_attempt_plan()
        stream = stream_final_response(
            winning_plan=winning_plan,
            prompt=prompt,
            max_tokens=max_tokens,
            conversation_messages=conversation_messages,
            response_model=response_model,
            json_mode=json_mode,
            temperature=temperature,
            stop_seqs=stop_seqs,
            verbosity=verbosity,
            enable_retry=enable_retry,
            retry_attempts=retry_attempts,
            before_retry_callback=before_retry_callback,
        )
        return StreamingResponseWithMetadata(
            stream=stream,
            tool_calls_made=all_tool_calls,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cache_creation_input_tokens=total_cache_creation_tokens,
            cache_read_input_tokens=total_cache_read_tokens,
            thinking_content=None,
            iterations=iteration + 1,
        )

    current_attempt.set(1)

    async def _final_call() -> HonchoLLMCallResponse[Any]:
        plan = get_attempt_plan()
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
            tools=None,
            tool_choice=None,
            messages=conversation_messages,
            selected_config=plan.selected_config,
        )

    if enable_retry:
        final_call_func = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
        )(_final_call)
    else:
        final_call_func = _final_call

    final_response = await final_call_func()
    final_response.tool_calls_made = all_tool_calls
    final_response.iterations = iteration + 1
    final_response.input_tokens = total_input_tokens + final_response.input_tokens
    final_response.output_tokens = total_output_tokens + final_response.output_tokens
    final_response.cache_creation_input_tokens = (
        total_cache_creation_tokens + final_response.cache_creation_input_tokens
    )
    final_response.cache_read_input_tokens = (
        total_cache_read_tokens + final_response.cache_read_input_tokens
    )
    return final_response


__all__ = [
    "MAX_TOOL_ITERATIONS",
    "MIN_TOOL_ITERATIONS",
    "append_tool_results",
    "execute_tool_loop",
    "format_assistant_tool_message",
    "stream_final_response",
]
