"""
Shared helpers for OpenAI-compatible providers.

OpenAI, OpenRouter, and vLLM all use the OpenAI-compatible chat API shape but have
provider-specific quirks. This module holds shared parsing and streaming helpers
to keep per-provider adapters small and explicit.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from src.utils.llm.models import HonchoLLMCallStreamChunk

logger = logging.getLogger(__name__)


def extract_openai_tool_calls(message: Any) -> list[dict[str, Any]]:
    """
    Extract tool calls from an OpenAI-compatible message object.

    OpenAI-compatible SDKs may return different tool call shapes. This helper
    treats tool call entries as dynamic objects and extracts:
    - id
    - function name
    - function arguments (JSON string)
    """
    tool_calls_obj = getattr(message, "tool_calls", None)
    if not tool_calls_obj:
        return []

    tool_calls_any = cast(list[Any], tool_calls_obj)
    parsed: list[dict[str, Any]] = []

    for tc in tool_calls_any:
        if isinstance(tc, dict):
            tc_dict: dict[str, Any] = cast(dict[str, Any], tc)
            tc_id = cast(str, tc_dict.get("id", ""))
            func = tc_dict.get("function")
            if isinstance(func, dict):
                func_dict: dict[str, Any] = cast(dict[str, Any], func)
                name = cast(str, func_dict.get("name", ""))
                args_str = func_dict.get("arguments")
            else:
                name = cast(str, tc_dict.get("name", ""))
                args_str = tc_dict.get("arguments")
        else:
            tc_id = cast(str, getattr(tc, "id", ""))
            func_obj = getattr(tc, "function", None)
            if func_obj is not None:
                name = cast(str, getattr(func_obj, "name", ""))
                args_str = getattr(func_obj, "arguments", None)
            else:
                name = cast(str, getattr(tc, "name", ""))
                args_str = getattr(tc, "arguments", None)

        args: dict[str, Any] = {}
        if isinstance(args_str, str) and args_str:
            try:
                parsed_args_any = json.loads(args_str)
                if isinstance(parsed_args_any, dict):
                    args = cast(dict[str, Any], parsed_args_any)
            except Exception:
                args = {}

        parsed.append({"id": tc_id, "name": name, "input": args})

    return parsed


def extract_openai_reasoning_content(response: Any) -> str | None:
    """
    Extract reasoning/thinking content from an OpenAI ChatCompletion response.

    GPT-5 and o1 models may include `reasoning_details` in the response message.
    Some OpenAI-compatible proxies may include `reasoning_content`.
    """
    try:
        message = response.choices[0].message
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            reasoning_parts: list[Any] = []
            for detail in message.reasoning_details:
                if hasattr(detail, "content") and detail.content:
                    reasoning_parts.append(detail.content)
                elif isinstance(detail, dict):
                    detail_dict: dict[str, Any] = cast(dict[str, Any], detail)
                    content = detail_dict.get("content")
                    if isinstance(content, str) and content:
                        reasoning_parts.append(content)
            if reasoning_parts:
                return "\n".join(reasoning_parts)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return message.reasoning_content
    except (AttributeError, IndexError, TypeError):
        pass
    return None


def extract_openai_reasoning_details(response: Any) -> list[dict[str, Any]]:
    """
    Extract `reasoning_details` array from an OpenAI-compatible ChatCompletion response.

    OpenRouter may return reasoning blocks in `reasoning_details` that must be preserved
    and passed back in subsequent requests for Gemini models with tool use.
    """
    try:
        message = response.choices[0].message
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            return [
                detail.model_dump() if hasattr(detail, "model_dump") else dict(detail)
                for detail in message.reasoning_details
            ]
    except (AttributeError, IndexError, TypeError):
        pass
    return []


def extract_openai_cache_tokens(usage: Any) -> tuple[int, int]:
    """
    Extract cache token counts from OpenAI-style usage objects.

    Returns:
        Tuple of (cache_creation_tokens, cache_read_tokens).
    """
    if not usage:
        return 0, 0

    cache_read = 0

    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        details = usage.prompt_tokens_details
        if hasattr(details, "cached_tokens") and details.cached_tokens:
            cache_read = details.cached_tokens

    if cache_read == 0:
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
            cache_read = usage.cache_read_input_tokens
        elif hasattr(usage, "cached_tokens") and usage.cached_tokens:
            cache_read = usage.cached_tokens

    cache_creation = 0
    if (
        hasattr(usage, "cache_creation_input_tokens")
        and usage.cache_creation_input_tokens
    ):
        cache_creation = usage.cache_creation_input_tokens

    return cache_creation, cache_read


async def stream_openai_compatible(
    *,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel] | None,
    json_mode: bool,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None,
    verbosity: Literal["low", "medium", "high"] | None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    """
    Stream an OpenAI-compatible response and normalize chunks.

    The OpenAI python client returns an async iterator of ChatCompletionChunk items.
    With `include_usage`, a final chunk can include usage details in `chunk.usage`.
    """
    openai_params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if "gpt-5" in model:
        openai_params["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            openai_params["reasoning_effort"] = reasoning_effort
        if verbosity:
            openai_params["verbosity"] = verbosity
    else:
        openai_params["max_tokens"] = max_tokens

    if response_model:
        openai_params["response_format"] = response_model
    elif json_mode:
        openai_params["response_format"] = {"type": "json_object"}

    openai_stream = cast(
        AsyncIterator[ChatCompletionChunk],
        await client.chat.completions.create(**openai_params),
    )
    finish_reason: str | None = None
    usage_chunk_received = False

    async for chunk in openai_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield HonchoLLMCallStreamChunk(content=chunk.choices[0].delta.content)
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if hasattr(chunk, "usage") and chunk.usage:
            yield HonchoLLMCallStreamChunk(
                content="",
                is_done=True,
                finish_reasons=[finish_reason] if finish_reason else [],
                output_tokens=chunk.usage.completion_tokens,
            )
            usage_chunk_received = True

    if not usage_chunk_received and finish_reason:
        logger.warning(
            "OpenAI-compatible stream ended without usage chunk (interrupted)"
        )
        yield HonchoLLMCallStreamChunk(
            content="",
            is_done=True,
            finish_reasons=[finish_reason],
            output_tokens=None,
        )
