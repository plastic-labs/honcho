"""Single-call executor: the inner LLM-call path without tool-loop orchestration.

`honcho_llm_call_inner` handles one backend call (complete or stream), building
the effective ModelConfig and delegating to request_builder. Result / stream
chunk types are bridged to the public Honcho* shapes here.

Used by:
- src/llm/api.py (the public entrypoint, for both tool-less and tool-enabled paths)
- src/llm/tool_loop.py (each iteration of the tool loop calls this)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, TypeVar, overload

from pydantic import BaseModel

from src.config import ModelConfig, ModelTransport

from .backend import CompletionResult as BackendCompletionResult
from .backend import StreamChunk as BackendStreamChunk
from .backend import ToolCallResult
from .registry import CLIENTS, backend_for_provider
from .request_builder import execute_completion, execute_stream
from .runtime import effective_config_for_call
from .types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    ProviderClient,
    ReasoningEffortType,
)

M = TypeVar("M", bound=BaseModel)


def _tool_call_result_to_dict(tool_call: ToolCallResult) -> dict[str, Any]:
    result = {
        "id": tool_call.id,
        "name": tool_call.name,
        "input": tool_call.input,
    }
    if tool_call.thought_signature is not None:
        result["thought_signature"] = tool_call.thought_signature
    return result


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
) -> HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]:
    """One backend call. No retry, no fallback, no tool loop.

    The outer src/llm/api.py `honcho_llm_call` handles retry + fallback +
    tool orchestration on top of this.
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

    if stream:

        async def _stream() -> AsyncIterator[HonchoLLMCallStreamChunk]:
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
            async for chunk in stream_iter:
                yield stream_chunk_to_response_chunk(chunk)

        return _stream()

    result = await execute_completion(
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
    return completion_result_to_response(result)


__all__ = [
    "completion_result_to_response",
    "honcho_llm_call_inner",
    "stream_chunk_to_response_chunk",
]
