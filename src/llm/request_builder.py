"""Low-level request assembly: flatten a ModelConfig into backend calls.

Does NOT own: retry, fallback, tool loop, provider selection. Those live in
src/llm/api.py, src/llm/tool_loop.py, src/llm/runtime.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from src.config import ModelConfig, PromptCachePolicy

from .backend import CompletionResult, ProviderBackend, StreamChunk


def build_config_extra_params(config: ModelConfig) -> dict[str, Any]:
    """Flatten ModelConfig's optional knobs and provider_params into extra_params.

    Backends read per-call tuning parameters (top_p, top_k, frequency_penalty,
    presence_penalty, seed) and the free-form provider_params passthrough out
    of ``extra_params``. Single source of truth for that translation.
    """
    extra_params: dict[str, Any] = {}
    if config.top_p is not None:
        extra_params["top_p"] = config.top_p
    if config.top_k is not None:
        extra_params["top_k"] = config.top_k
    if config.frequency_penalty is not None:
        extra_params["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        extra_params["presence_penalty"] = config.presence_penalty
    if config.seed is not None:
        extra_params["seed"] = config.seed

    if config.provider_params:
        extra_params.update(config.provider_params)

    return extra_params


async def execute_completion(
    backend: ProviderBackend,
    config: ModelConfig,
    *,
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    stop: list[str] | None = None,
    cache_policy: PromptCachePolicy | None = None,
    extra_params: dict[str, Any] | None = None,
) -> CompletionResult:
    # Preserve 0 as an explicit "disable thinking" value (used by Gemini);
    # only convert to None when the field is truly unset.
    effective_max_tokens = config.max_output_tokens or max_tokens

    merged_extra_params = {
        **build_config_extra_params(config),
        **(extra_params or {}),
    }
    if cache_policy is not None:
        merged_extra_params["cache_policy"] = cache_policy

    return await backend.complete(
        model=config.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=config.temperature,
        stop=stop if stop is not None else config.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        thinking_budget_tokens=config.thinking_budget_tokens,
        thinking_effort=config.thinking_effort,
        max_output_tokens=effective_max_tokens,
        extra_params=merged_extra_params,
    )


async def execute_stream(
    backend: ProviderBackend,
    config: ModelConfig,
    *,
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    stop: list[str] | None = None,
    cache_policy: PromptCachePolicy | None = None,
    extra_params: dict[str, Any] | None = None,
) -> AsyncIterator[StreamChunk]:
    effective_max_tokens = config.max_output_tokens or max_tokens

    merged_extra_params = {
        **build_config_extra_params(config),
        **(extra_params or {}),
    }
    if cache_policy is not None:
        merged_extra_params["cache_policy"] = cache_policy

    return backend.stream(
        model=config.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=config.temperature,
        stop=stop if stop is not None else config.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        thinking_budget_tokens=config.thinking_budget_tokens,
        thinking_effort=config.thinking_effort,
        max_output_tokens=effective_max_tokens,
        extra_params=merged_extra_params,
    )
