"""Low-level request assembly: flatten a ModelConfig into backend calls.

Does NOT own: retry, fallback, tool loop, provider selection. Those live in
src/llm/api.py, src/llm/tool_loop.py, src/llm/runtime.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from pydantic import BaseModel

from src.config import ModelConfig, PromptCachePolicy
from src.exceptions import ValidationException

from .backend import CompletionResult, ProviderBackend, StreamChunk

# Operator escape-hatch keys recognized inside ModelConfig.provider_params.
PASSTHROUGH_KEYS = ("extra_body", "extra_headers", "extra_query")


def coerce_passthrough_mapping(key: str, value: Any) -> dict[str, Any]:
    """Validate an operator-supplied provider_params passthrough is a mapping.

    ``provider_params`` is typed ``dict[str, Any]`` with no nested schema, so an
    operator can supply a non-mapping (e.g. a list or string) for one of the
    passthrough keys. Catch that here with a clear error instead of letting a
    later ``dict.update()`` raise an opaque ``TypeError`` deep in the transport.

    Args:
        key: The passthrough key name, used only for the error message.
        value: The operator-supplied value to validate.

    Returns:
        The value, narrowed to ``dict[str, Any]``.

    Raises:
        ValidationException: If ``value`` is not a mapping.
    """
    if not isinstance(value, dict):
        raise ValidationException(
            f"provider_params.{key} must be a mapping, got {type(value).__name__}"
        )
    return cast(dict[str, Any], value)


def apply_sdk_passthroughs(
    params: dict[str, Any], extra_params: dict[str, Any]
) -> None:
    """Forward operator provider_params passthroughs onto an SDK call dict.

    OpenAI and Anthropic both accept ``extra_body`` / ``extra_headers`` /
    ``extra_query`` as identically-named SDK kwargs, so they share this merge.
    Operator values shallow-merge onto ``params`` in place, winning over any
    value Honcho already set under the same top-level key (e.g. an auto-injected
    ``extra_body.reasoning``). Gemini handles passthroughs separately because the
    google-genai SDK does not expose these as kwargs.

    Args:
        params: The SDK call kwargs being assembled; mutated in place.
        extra_params: Flattened per-call params (see build_config_extra_params).

    Raises:
        ValidationException: If a passthrough value is not a mapping.
    """
    for passthrough_key in PASSTHROUGH_KEYS:
        operator_value = extra_params.get(passthrough_key)
        if not operator_value:
            continue
        existing = params.setdefault(passthrough_key, {})
        existing.update(coerce_passthrough_mapping(passthrough_key, operator_value))


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
