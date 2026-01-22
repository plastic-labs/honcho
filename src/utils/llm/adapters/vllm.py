"""
vLLM provider adapter.

Honcho uses vLLM via an OpenAI-compatible API surface. However, structured output
support is currently implemented only for `PromptRepresentation` using JSON Schema,
and includes schema-aware repair to prevent downstream validation failures.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from src.utils.json_parser import validate_and_repair_json
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
from src.utils.representation import PromptRepresentation
from src.utils.types import SupportedProviders

logger = logging.getLogger(__name__)


class VLLMAdapter(ProviderAdapter):
    """ProviderAdapter implementation for vLLM via OpenAI-compatible SDK client."""

    provider: SupportedProviders = "vllm"

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
        """Format an assistant message with tool calls for vLLM."""
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
        """Perform a non-streaming vLLM call and normalize the response."""
        _ = (
            provider,
            prompt,
            json_mode,
            reasoning_effort,
            verbosity,
            thinking_budget_tokens,
        )

        openai_client = cast(AsyncOpenAI, client)

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

        if response_model:
            if response_model is not PromptRepresentation:
                raise NotImplementedError(
                    "vLLM structured output currently supports only PromptRepresentation"
                )
            openai_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            }
            if stop_seqs:
                openai_params["stop"] = stop_seqs

            vllm_response: ChatCompletion = cast(
                ChatCompletion,
                await openai_client.chat.completions.create(**openai_params),
            )

            usage = vllm_response.usage
            finish_reason = vllm_response.choices[0].finish_reason

            try:
                test_rep = ""
                if vllm_response.choices[0].message.content is not None:
                    test_rep = vllm_response.choices[0].message.content

                final = validate_and_repair_json(test_rep)
                repaired_data = json.loads(final)

                if "deductive" in repaired_data and isinstance(
                    repaired_data["deductive"], list
                ):
                    for i, item in enumerate(repaired_data["deductive"]):
                        if isinstance(item, dict):
                            if "conclusion" not in item and "premises" in item:
                                logger.warning(
                                    "Deductive observation %s missing conclusion, adding placeholder",
                                    i,
                                )
                                if item["premises"]:
                                    item["conclusion"] = (
                                        f"[Incomplete reasoning from premises: {item['premises'][0][:100]}...]"
                                    )
                                else:
                                    item["conclusion"] = (
                                        "[Incomplete reasoning - conclusion missing]"
                                    )
                            if "premises" not in item:
                                item["premises"] = []

                final = json.dumps(repaired_data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                final = ""
                logger.warning("Could not perform schema-aware repair: %s", e)

            try:
                response_obj = PromptRepresentation.model_validate_json(final)
            except ValidationError as e:
                logger.error("Validation error after repair: %s", e)
                logger.debug("Problematic JSON: %s", final)
                logger.warning(
                    "Using fallback empty Representation due to validation error"
                )
                response_obj = PromptRepresentation(explicit=[])

            cache_creation, cache_read = extract_openai_cache_tokens(usage)
            return HonchoLLMCallResponse(
                content=response_obj,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                finish_reasons=[finish_reason] if finish_reason else [],
                tool_calls_made=[],
                thinking_content=extract_openai_reasoning_content(vllm_response),
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
        """Stream a vLLM response and normalize chunks."""
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
