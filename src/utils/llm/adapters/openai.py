"""
OpenAI provider adapter.

This adapter implements the OpenAI-native provider (`openai`). OpenAI-compatible
providers with special quirks (OpenRouter, vLLM) have their own adapters.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from src.utils.llm.adapters.base import (
    ProviderAdapter,
    openai_tool_calls_from_tool_calls,
)
from src.utils.llm.adapters.openai_common import (
    extract_openai_cache_tokens,
    extract_openai_reasoning_content,
    extract_openai_reasoning_details,
    extract_openai_tool_calls,
    stream_openai_compatible,
)
from src.utils.llm.models import HonchoLLMCallResponse, HonchoLLMCallStreamChunk
from src.utils.types import SupportedProviders

logger = logging.getLogger(__name__)


class OpenAIAdapter(ProviderAdapter):
    """ProviderAdapter implementation for OpenAI's native API."""

    provider: SupportedProviders = "openai"

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-style tool schemas to OpenAI tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def format_assistant_tool_message(
        self,
        *,
        content: Any,
        tool_calls: list[dict[str, Any]],
        thinking_blocks: list[dict[str, Any]] | None = None,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Format an assistant message with tool calls for OpenAI-compatible APIs."""
        _ = thinking_blocks
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content if isinstance(content, str) else None,
            "tool_calls": openai_tool_calls_from_tool_calls(tool_calls),
        }
        if reasoning_details:
            msg["reasoning_details"] = reasoning_details
        return msg

    def append_tool_results(
        self,
        *,
        tool_results: list[dict[str, Any]],
        conversation_messages: list[dict[str, Any]],
    ) -> None:
        """Append tool results in OpenAI tool-message format."""
        for tr in tool_results:
            conversation_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tr["tool_id"],
                    "content": str(tr["result"]),
                }
            )

    async def call(
        self,
        *,
        client: Any,
        provider: SupportedProviders,
        model: str,
        prompt: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None,
        json_mode: bool,
        temperature: float | None,
        stop_seqs: list[str] | None,
        reasoning_effort: str | None,
        verbosity: str | None,
        thinking_budget_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> HonchoLLMCallResponse[Any]:
        """Perform a non-streaming OpenAI call and normalize the response."""
        _ = (prompt, thinking_budget_tokens)
        openai_params: dict[str, Any] = {"model": model, "messages": messages}

        if temperature is not None and "gpt-5" not in model:
            openai_params["temperature"] = temperature

        if "gpt-5" in model:
            openai_params["max_completion_tokens"] = max_tokens
            if reasoning_effort:
                openai_params["reasoning_effort"] = reasoning_effort
            if verbosity:
                openai_params["verbosity"] = verbosity
        else:
            openai_params["max_tokens"] = max_tokens

        if tools and not response_model:
            openai_params["tools"] = tools
            if tool_choice:
                openai_params["tool_choice"] = tool_choice

        if json_mode:
            openai_params["response_format"] = {"type": "json_object"}

        openai_client = cast(AsyncOpenAI, client)

        if response_model:
            openai_params["response_format"] = response_model
            response = cast(
                ChatCompletion,
                await openai_client.chat.completions.parse(**openai_params),
            )
            message_any = cast(Any, response.choices[0].message)
            parsed_content = getattr(message_any, "parsed", None)
            if parsed_content is None:
                raise ValueError("No parsed content in structured response")

            usage = response.usage
            finish_reason = response.choices[0].finish_reason

            if not isinstance(parsed_content, response_model):
                raise ValueError(
                    f"Parsed content does not match the response model: {parsed_content} != {response_model}"
                )

            parsed_tool_calls = extract_openai_tool_calls(response.choices[0].message)

            cache_creation, cache_read = extract_openai_cache_tokens(usage)
            return HonchoLLMCallResponse(
                content=parsed_content,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                finish_reasons=[finish_reason] if finish_reason else [],
                tool_calls_made=parsed_tool_calls,
                thinking_content=extract_openai_reasoning_content(response),
                reasoning_details=extract_openai_reasoning_details(response),
            )

        response = cast(
            ChatCompletion, await openai_client.chat.completions.create(**openai_params)
        )

        usage = response.usage
        finish_reason = response.choices[0].finish_reason

        tool_calls_list = extract_openai_tool_calls(response.choices[0].message)

        cache_creation, cache_read = extract_openai_cache_tokens(usage)
        return HonchoLLMCallResponse(
            content=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            finish_reasons=[finish_reason] if finish_reason else [],
            tool_calls_made=tool_calls_list,
            thinking_content=extract_openai_reasoning_content(response),
            reasoning_details=extract_openai_reasoning_details(response),
        )

    def stream(
        self,
        *,
        client: Any,
        provider: SupportedProviders,
        model: str,
        prompt: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None,
        json_mode: bool,
        temperature: float | None,
        reasoning_effort: str | None,
        verbosity: str | None,
        thinking_budget_tokens: int | None,
    ) -> AsyncIterator[HonchoLLMCallStreamChunk]:
        """Stream an OpenAI response and normalize chunks."""
        _ = (provider, prompt, temperature, thinking_budget_tokens)

        openai_client = cast(AsyncOpenAI, client)
        return stream_openai_compatible(
            client=openai_client,
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            response_model=response_model,
            json_mode=json_mode,
            reasoning_effort=cast(Any, reasoning_effort),
            verbosity=cast(Any, verbosity),
        )
