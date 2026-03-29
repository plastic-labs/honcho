from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from src.config import ModelConfig

from .backend import CompletionResult, ProviderBackend, StreamChunk
from .caching import PromptCachePolicy
from .capabilities import ModelCapabilities, get_model_capabilities
from .credentials import resolve_credentials


def _adjust_max_tokens_for_explicit_budget(
    capabilities: ModelCapabilities,
    max_tokens: int,
    thinking_budget_tokens: int | None,
) -> int:
    if (
        capabilities.shared_reasoning_budget
        and thinking_budget_tokens is not None
        and thinking_budget_tokens > 0
    ):
        return max_tokens + thinking_budget_tokens
    return max_tokens


def _build_config_extra_params(config: ModelConfig) -> dict[str, Any]:
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
    capabilities = get_model_capabilities(config)
    credentials = resolve_credentials(config)
    thinking_budget_tokens = (
        config.thinking_budget_tokens
        if config.thinking_budget_tokens is not None
        and config.thinking_budget_tokens > 0
        else None
    )
    requested_output_tokens = config.max_output_tokens or max_tokens
    effective_max_tokens = _adjust_max_tokens_for_explicit_budget(
        capabilities,
        requested_output_tokens,
        thinking_budget_tokens,
    )

    merged_extra_params = {
        **_build_config_extra_params(config),
        **(extra_params or {}),
    }
    if cache_policy is not None:
        merged_extra_params["cache_policy"] = cache_policy

    return await backend.complete(
        model=config.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=config.temperature,
        stop=stop or config.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        thinking_budget_tokens=thinking_budget_tokens,
        thinking_effort=config.thinking_effort,
        max_output_tokens=requested_output_tokens,
        api_key=credentials.get("api_key"),
        api_base=credentials.get("api_base"),
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
    capabilities = get_model_capabilities(config)
    credentials = resolve_credentials(config)
    thinking_budget_tokens = (
        config.thinking_budget_tokens
        if config.thinking_budget_tokens is not None
        and config.thinking_budget_tokens > 0
        else None
    )
    requested_output_tokens = config.max_output_tokens or max_tokens
    effective_max_tokens = _adjust_max_tokens_for_explicit_budget(
        capabilities,
        requested_output_tokens,
        thinking_budget_tokens,
    )

    merged_extra_params = {
        **_build_config_extra_params(config),
        **(extra_params or {}),
    }
    if cache_policy is not None:
        merged_extra_params["cache_policy"] = cache_policy

    return backend.stream(
        model=config.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=config.temperature,
        stop=stop or config.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        thinking_budget_tokens=thinking_budget_tokens,
        thinking_effort=config.thinking_effort,
        max_output_tokens=requested_output_tokens,
        api_key=credentials.get("api_key"),
        api_base=credentials.get("api_base"),
        extra_params=merged_extra_params,
    )
