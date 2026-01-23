"""
Anthropic provider adapter.

This adapter encapsulates all Anthropic-specific behaviors:
- system messages passed as a top-level `system` parameter with cache_control
- extended thinking blocks (including signatures) and tool-use blocks
- JSON mode / response_model via prompt instructions and a prefixed `{`
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from anthropic.types.message import Message as AnthropicMessage
from anthropic.types.usage import Usage
from pydantic import BaseModel, ValidationError

from src.utils.llm.adapters.base import ProviderAdapter
from src.utils.llm.models import HonchoLLMCallResponse, HonchoLLMCallStreamChunk
from src.utils.types import SupportedProviders

logger = logging.getLogger(__name__)


class AnthropicAdapter(ProviderAdapter):
    """ProviderAdapter implementation for Anthropic SDK clients."""

    provider: SupportedProviders = "anthropic"

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return tools unchanged (Anthropic-native schema)."""
        return tools

    def format_assistant_tool_message(
        self,
        *,
        content: Any,
        tool_calls: list[dict[str, Any]],
        thinking_blocks: list[dict[str, Any]] | None = None,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Format an assistant tool-use message for Anthropic."""
        _ = reasoning_details
        content_blocks: list[dict[str, Any]] = []

        if thinking_blocks:
            content_blocks.extend(thinking_blocks)

        if isinstance(content, str) and content:
            content_blocks.append({"type": "text", "text": content})

        for tool_call in tool_calls:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": tool_call["input"],
                }
            )

        return {"role": "assistant", "content": content_blocks}

    def append_tool_results(
        self,
        *,
        tool_results: list[dict[str, Any]],
        conversation_messages: list[dict[str, Any]],
    ) -> None:
        """Append tool results as Anthropic tool_result blocks."""
        result_blocks: list[dict[str, Any]] = []
        for tr in tool_results:
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tr["tool_id"],
                    "content": str(tr["result"]),
                    "is_error": tr.get("is_error", False),
                }
            )

        conversation_messages.append({"role": "user", "content": result_blocks})

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
        """Perform a non-streaming Anthropic call and normalize the response."""
        _ = (provider, prompt, stop_seqs, reasoning_effort, verbosity)

        system_messages: list[str] = []
        non_system_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(cast(str, msg["content"]))
            else:
                non_system_messages.append(msg)

        anthropic_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": non_system_messages,
        }

        if temperature is not None:
            anthropic_params["temperature"] = temperature

        if system_messages:
            anthropic_params["system"] = [
                {
                    "type": "text",
                    "text": "\n\n".join(system_messages),
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if tools:
            anthropic_params["tools"] = tools
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        anthropic_params["tool_choice"] = {"type": "auto"}
                    elif tool_choice in ("any", "required"):
                        anthropic_params["tool_choice"] = {"type": "any"}
                    elif tool_choice == "none":
                        pass
                    else:
                        anthropic_params["tool_choice"] = {
                            "type": "tool",
                            "name": tool_choice,
                        }
                else:
                    anthropic_params["tool_choice"] = tool_choice

        if response_model or json_mode:
            if response_model:
                schema_json = json.dumps(response_model.model_json_schema(), indent=2)
                anthropic_params["messages"][-1]["content"] += (
                    f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"
                )
            anthropic_params["messages"].append({"role": "assistant", "content": "{"})

        if thinking_budget_tokens:
            anthropic_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }

        anthropic_client = cast(AsyncAnthropic, client)
        anthropic_response: AnthropicMessage = cast(
            AnthropicMessage, await anthropic_client.messages.create(**anthropic_params)
        )

        text_blocks: list[str] = []
        thinking_text_blocks: list[str] = []
        thinking_full_blocks: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for block in anthropic_response.content:
            if isinstance(block, TextBlock):
                text_blocks.append(block.text)
            elif isinstance(block, ThinkingBlock):
                thinking_text_blocks.append(block.thinking)
                thinking_full_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    }
                )
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        usage: Any | Usage = anthropic_response.usage
        stop_reason = anthropic_response.stop_reason

        text_content = "\n".join(text_blocks)
        thinking_content = (
            "\n".join(thinking_text_blocks) if thinking_text_blocks else None
        )

        cache_creation_tokens = (
            getattr(usage, "cache_creation_input_tokens", 0) or 0 if usage else 0
        )
        cache_read_tokens = (
            getattr(usage, "cache_read_input_tokens", 0) or 0 if usage else 0
        )
        uncached_tokens = usage.input_tokens if usage else 0
        total_input_tokens = uncached_tokens + cache_read_tokens + cache_creation_tokens

        if response_model:
            try:
                json_content = "{" + text_content
                parsed_json = json.loads(json_content)
                parsed_content = response_model.model_validate(parsed_json)
                return HonchoLLMCallResponse(
                    content=parsed_content,
                    input_tokens=total_input_tokens,
                    output_tokens=usage.output_tokens if usage else 0,
                    cache_creation_input_tokens=cache_creation_tokens,
                    cache_read_input_tokens=cache_read_tokens,
                    finish_reasons=[stop_reason] if stop_reason else [],
                    tool_calls_made=tool_calls,
                    thinking_content=thinking_content,
                    thinking_blocks=thinking_full_blocks,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                raise ValueError(
                    f"Failed to parse Anthropic response as {response_model}: {e}. Raw content: {text_content}"
                ) from e

        return HonchoLLMCallResponse(
            content=text_content,
            input_tokens=total_input_tokens,
            output_tokens=usage.output_tokens if usage else 0,
            cache_creation_input_tokens=cache_creation_tokens,
            cache_read_input_tokens=cache_read_tokens,
            finish_reasons=[stop_reason] if stop_reason else [],
            tool_calls_made=tool_calls,
            thinking_content=thinking_content,
            thinking_blocks=thinking_full_blocks,
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
        """Stream an Anthropic response and normalize chunks."""
        _ = (provider, prompt, temperature, reasoning_effort, verbosity)

        system_content = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        anthropic_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [m for m in messages if m.get("role") != "system"],
        }
        if system_content:
            anthropic_params["system"] = [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if response_model or json_mode:
            if response_model:
                schema_json = json.dumps(response_model.model_json_schema(), indent=2)
                anthropic_params["messages"][-1]["content"] += (
                    f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"
                )
            anthropic_params["messages"].append({"role": "assistant", "content": "{"})

        if thinking_budget_tokens:
            anthropic_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }

        anthropic_client = cast(AsyncAnthropic, client)
        async with anthropic_client.messages.stream(
            **anthropic_params
        ) as anthropic_stream:
            async for chunk in anthropic_stream:
                if (
                    chunk.type == "content_block_delta"
                    and hasattr(chunk, "delta")
                    and hasattr(chunk.delta, "text")
                ):
                    text_content = getattr(chunk.delta, "text", "")
                    yield HonchoLLMCallStreamChunk(content=text_content)
            final_message = await anthropic_stream.get_final_message()
            usage = final_message.usage
            output_tokens = usage.output_tokens if usage else None
            yield HonchoLLMCallStreamChunk(
                content="",
                is_done=True,
                finish_reasons=[final_message.stop_reason]
                if final_message.stop_reason
                else [],
                output_tokens=output_tokens,
            )
