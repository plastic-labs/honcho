"""
Tool execution loop orchestration for agentic LLM interactions.

This module is intentionally provider-agnostic. Provider-specific formatting of
tool-call messages and tool-result messages is delegated to adapters.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import LLMComponentSettings
from src.utils.llm.adapters import get_adapter
from src.utils.llm.history import truncate_messages_to_fit
from src.utils.llm.models import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    IterationCallback,
    IterationData,
    StreamingResponseWithMetadata,
)
from src.utils.types import SupportedProviders, set_current_iteration

logger = logging.getLogger(__name__)

# Bounds for max_tool_iterations to prevent runaway loops
MIN_TOOL_ITERATIONS = 1
MAX_TOOL_ITERATIONS = 100


async def execute_tool_loop(
    *,
    llm_settings: LLMComponentSettings,
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
    reasoning_effort: str | None,
    verbosity: str | None,
    thinking_budget_tokens: int | None,
    enable_retry: bool,
    retry_attempts: int,
    max_input_tokens: int | None,
    get_provider_and_model: Callable[
        [],
        tuple[
            SupportedProviders,
            str,
            int | None,
            str | None,
            str | None,
        ],
    ],
    before_retry_callback: Callable[[Any], None],
    get_effective_temperature: Callable[[float | None], float | None],
    set_attempt: Callable[[int], None],
    call_inner: Callable[..., Awaitable[HonchoLLMCallResponse[Any]]],
    stream_final_response: Callable[..., AsyncIterator[HonchoLLMCallStreamChunk]],
    stream_final: bool = False,
    iteration_callback: IterationCallback | None = None,
) -> HonchoLLMCallResponse[Any] | StreamingResponseWithMetadata:
    """
    Execute the tool calling loop for agentic LLM interactions.

    The loop repeatedly calls the LLM with tools available, executes any requested
    tools, appends results back into the conversation, and continues until the
    model stops calling tools or max iterations is reached.
    """
    conversation_messages: list[dict[str, Any]] = (
        messages.copy() if messages else [{"role": "user", "content": prompt}]
    )

    iteration = 0
    all_tool_calls: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    effective_tool_choice = tool_choice

    while iteration < max_tool_iterations:
        if max_input_tokens is not None:
            conversation_messages = truncate_messages_to_fit(
                conversation_messages, max_input_tokens
            )

        async def _call_with_messages(
            effective_tool_choice: str | dict[str, Any] | None = effective_tool_choice,
            conversation_messages: list[dict[str, Any]] = conversation_messages,
        ) -> HonchoLLMCallResponse[Any]:
            (
                provider,
                model,
                thinking_budget,
                provider_reasoning_effort,
                provider_verbosity,
            ) = get_provider_and_model()
            adapter = get_adapter(provider)
            converted_tools = adapter.convert_tools(tools) if tools else None

            return await call_inner(
                provider,
                model,
                prompt,
                max_tokens,
                response_model,
                json_mode,
                get_effective_temperature(temperature),
                stop_seqs,
                provider_reasoning_effort,
                provider_verbosity,
                thinking_budget,
                False,
                converted_tools,
                effective_tool_choice,
                conversation_messages,
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
            if stream_final:
                stream = stream_final_response(
                    llm_settings=llm_settings,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    conversation_messages=conversation_messages,
                    response_model=response_model,
                    json_mode=json_mode,
                    temperature=temperature,
                    stop_seqs=stop_seqs,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    thinking_budget_tokens=thinking_budget_tokens,
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
                    messages=conversation_messages,
                )

            response.tool_calls_made = all_tool_calls
            response.input_tokens = total_input_tokens
            response.output_tokens = total_output_tokens
            response.cache_creation_input_tokens = total_cache_creation_tokens
            response.cache_read_input_tokens = total_cache_read_tokens
            response.iterations = iteration + 1
            response.messages = conversation_messages
            return response

        current_provider, _, _, _, _ = get_provider_and_model()
        adapter = get_adapter(current_provider)

        assistant_message = adapter.format_assistant_tool_message(
            content=response.content,
            tool_calls=response.tool_calls_made,
            thinking_blocks=response.thinking_blocks,
            reasoning_details=response.reasoning_details,
        )
        conversation_messages.append(assistant_message)

        set_current_iteration(iteration + 1)

        tool_results: list[dict[str, Any]] = []
        for tool_call in response.tool_calls_made:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            tool_id = tool_call.get("id", "")

            try:
                tool_result = await tool_executor(tool_name, tool_input)
                tool_results.append(
                    {"tool_id": tool_id, "tool_name": tool_name, "result": tool_result}
                )
                all_tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_result": tool_result,
                    }
                )
            except Exception as e:
                logger.error("Tool execution failed for %s: %s", tool_name, e)
                tool_results.append(
                    {
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "result": f"Error: {str(e)}",
                        "is_error": True,
                    }
                )

        adapter.append_tool_results(
            tool_results=tool_results,
            conversation_messages=conversation_messages,
        )

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

        if iteration == 0 and effective_tool_choice in ("required", "any"):
            effective_tool_choice = "auto"

        iteration += 1

    synthesis_prompt = (
        "You have reached the maximum number of tool calls. "
        "Based on all the information you have gathered, provide your final response now. "
        "Do not attempt to call any more tools."
    )
    conversation_messages.append({"role": "user", "content": synthesis_prompt})

    if stream_final:
        stream = stream_final_response(
            llm_settings=llm_settings,
            prompt=prompt,
            max_tokens=max_tokens,
            conversation_messages=conversation_messages,
            response_model=response_model,
            json_mode=json_mode,
            temperature=temperature,
            stop_seqs=stop_seqs,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            thinking_budget_tokens=thinking_budget_tokens,
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
            messages=conversation_messages,
        )

    set_attempt(1)

    async def _final_call() -> HonchoLLMCallResponse[Any]:
        return await call_inner(
            llm_settings.PROVIDER,
            llm_settings.MODEL,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            get_effective_temperature(temperature),
            stop_seqs,
            reasoning_effort,
            verbosity,
            thinking_budget_tokens,
            False,
            None,
            None,
            conversation_messages,
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
    final_response.messages = conversation_messages
    return final_response
