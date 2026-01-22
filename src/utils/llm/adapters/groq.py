"""
Groq provider adapter.

Groq's API is OpenAI-chat compatible for basic completions and streaming. Honcho
currently uses it without tool calling.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from groq import AsyncGroq
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, ValidationError

from src.utils.llm.adapters.base import (
    ProviderAdapter,
    openai_tool_calls_from_tool_calls,
)
from src.utils.llm.adapters.openai_common import extract_openai_cache_tokens
from src.utils.llm.models import HonchoLLMCallResponse, HonchoLLMCallStreamChunk
from src.utils.types import SupportedProviders

logger = logging.getLogger(__name__)


class GroqAdapter(ProviderAdapter):
    """ProviderAdapter implementation for the Groq SDK client."""

    provider: SupportedProviders = "groq"

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return tools unchanged; tool calling is not implemented for Groq."""
        logger.warning(
            "Tool calling not implemented for provider groq, returning tools as-is"
        )
        return tools

    def format_assistant_tool_message(
        self,
        *,
        content: Any,
        tool_calls: list[dict[str, Any]],
        thinking_blocks: list[dict[str, Any]] | None = None,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Format an assistant message with tool calls in OpenAI format."""
        _ = (thinking_blocks, reasoning_details)
        return {
            "role": "assistant",
            "content": content if isinstance(content, str) else None,
            "tool_calls": openai_tool_calls_from_tool_calls(tool_calls),
        }

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
        """Perform a non-streaming Groq call and normalize the response."""
        _ = (
            provider,
            prompt,
            stop_seqs,
            reasoning_effort,
            verbosity,
            thinking_budget_tokens,
            tools,
            tool_choice,
        )

        groq_client = cast(AsyncGroq, client)

        groq_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if temperature is not None:
            groq_params["temperature"] = temperature

        if response_model:
            groq_params["response_format"] = response_model
        elif json_mode:
            groq_params["response_format"] = {"type": "json_object"}

        response = cast(
            ChatCompletion, await groq_client.chat.completions.create(**groq_params)
        )
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")

        usage = response.usage
        finish_reason = response.choices[0].finish_reason

        cache_creation, cache_read = extract_openai_cache_tokens(usage)
        if response_model:
            try:
                json_content = json.loads(response.choices[0].message.content)
                parsed_content = response_model.model_validate(json_content)
                return HonchoLLMCallResponse(
                    content=parsed_content,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    cache_creation_input_tokens=cache_creation,
                    cache_read_input_tokens=cache_read,
                    finish_reasons=[finish_reason] if finish_reason else [],
                    tool_calls_made=[],
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                raise ValueError(
                    f"Failed to parse Groq response as {response_model}: {e}. Raw content: {response.choices[0].message.content}"
                ) from e

        return HonchoLLMCallResponse(
            content=response.choices[0].message.content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            finish_reasons=[finish_reason] if finish_reason else [],
            tool_calls_made=[],
        )

    async def stream(
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
        """Stream a Groq response and normalize chunks."""
        _ = (provider, prompt, reasoning_effort, verbosity, thinking_budget_tokens)

        groq_client = cast(AsyncGroq, client)

        groq_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": True,
        }

        if response_model:
            groq_params["response_format"] = response_model
        elif json_mode:
            groq_params["response_format"] = {"type": "json_object"}

        groq_stream = cast(
            AsyncIterator[ChatCompletionChunk],
            await groq_client.chat.completions.create(**groq_params),
        )
        async for chunk in groq_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield HonchoLLMCallStreamChunk(content=chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                yield HonchoLLMCallStreamChunk(
                    content="",
                    is_done=True,
                    finish_reasons=[chunk.choices[0].finish_reason],
                )
