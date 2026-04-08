from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from pydantic import BaseModel, ValidationError

from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.structured_output import repair_response_model_json


class AnthropicBackend:
    """Provider backend wrapping the native Anthropic SDK."""

    def __init__(self, client: Any) -> None:
        self._client: Any = client

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> CompletionResult:
        del max_output_tokens, api_key, api_base
        if thinking_effort is not None:
            raise ValueError(
                "Anthropic backend does not support thinking_effort; use thinking_budget_tokens instead"
            )

        request_messages, system_messages = self._extract_system(messages)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": request_messages,
        }

        if temperature is not None:
            params["temperature"] = temperature
        if stop:
            params["stop_sequences"] = stop
        if system_messages:
            params["system"] = [
                {
                    "type": "text",
                    "text": "\n\n".join(system_messages),
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            params["tools"] = tools
            converted_tool_choice = self._convert_tool_choice(tool_choice)
            if converted_tool_choice is not None:
                params["tool_choice"] = converted_tool_choice
        if thinking_budget_tokens:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }
        if extra_params:
            for key in ("top_p", "top_k"):
                if key in extra_params:
                    params[key] = extra_params[key]

        use_json_prefill = (
            bool(response_format or self._json_mode(extra_params))
            and not thinking_budget_tokens
            and self._supports_assistant_prefill(model)
        )
        if use_json_prefill and params["messages"]:
            if response_format and isinstance(response_format, type):
                schema_json = json.dumps(response_format.model_json_schema(), indent=2)
                params["messages"][-1]["content"] += (
                    f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"
                )
            params["messages"].append({"role": "assistant", "content": "{"})
        elif (
            response_format and isinstance(response_format, type) and params["messages"]
        ):
            schema_json = json.dumps(response_format.model_json_schema(), indent=2)
            params["messages"][-1]["content"] += (
                f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"
            )

        response = await self._client.messages.create(**params)
        return self._normalize_response(
            response=response,
            response_format=response_format
            if isinstance(response_format, type)
            else None,
            prefilled_json=use_json_prefill,
            model_name=model,
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        is_json_mode = self._json_mode(extra_params)
        del max_output_tokens, api_key, api_base
        if thinking_effort is not None:
            raise ValueError(
                "Anthropic backend does not support thinking_effort; use thinking_budget_tokens instead"
            )

        request_messages, system_messages = self._extract_system(messages)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": request_messages,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if stop:
            params["stop_sequences"] = stop
        if tools:
            params["tools"] = tools
            converted_tool_choice = self._convert_tool_choice(tool_choice)
            if converted_tool_choice is not None:
                params["tool_choice"] = converted_tool_choice
        if system_messages:
            params["system"] = [
                {
                    "type": "text",
                    "text": "\n\n".join(system_messages),
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if extra_params:
            for key in ("top_p", "top_k"):
                if key in extra_params:
                    params[key] = extra_params[key]
        use_json_prefill = (
            bool(response_format or is_json_mode)
            and not thinking_budget_tokens
            and self._supports_assistant_prefill(model)
        )
        if use_json_prefill and params["messages"]:
            if response_format and isinstance(response_format, type):
                schema_json = json.dumps(response_format.model_json_schema(), indent=2)
                params["messages"][-1]["content"] += (
                    f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"
                )
            params["messages"].append({"role": "assistant", "content": "{"})
        elif (
            response_format and isinstance(response_format, type) and params["messages"]
        ):
            schema_json = json.dumps(response_format.model_json_schema(), indent=2)
            params["messages"][-1]["content"] += (
                f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"
            )
        if thinking_budget_tokens:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }

        async with self._client.messages.stream(**params) as stream:
            async for chunk in stream:
                if (
                    chunk.type == "content_block_delta"
                    and hasattr(chunk, "delta")
                    and hasattr(chunk.delta, "text")
                ):
                    yield StreamChunk(content=getattr(chunk.delta, "text", ""))

            final_message = await stream.get_final_message()
            output_tokens = (
                final_message.usage.output_tokens if final_message.usage else None
            )
            yield StreamChunk(
                is_done=True,
                finish_reason=final_message.stop_reason,
                output_tokens=output_tokens,
            )

    def _normalize_response(
        self,
        *,
        response: Any,
        response_format: type[BaseModel] | None,
        prefilled_json: bool,
        model_name: str,
    ) -> CompletionResult:
        text_blocks: list[str] = []
        thinking_text_blocks: list[str] = []
        thinking_full_blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCallResult] = []

        for block in response.content:
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
                    ToolCallResult(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input),
                    )
                )

        usage = response.usage
        cache_creation_tokens = (
            getattr(usage, "cache_creation_input_tokens", 0) or 0 if usage else 0
        )
        cache_read_tokens = (
            getattr(usage, "cache_read_input_tokens", 0) or 0 if usage else 0
        )
        uncached_tokens = usage.input_tokens if usage else 0
        total_input_tokens = uncached_tokens + cache_creation_tokens + cache_read_tokens

        text_content = "\n".join(text_blocks)
        thinking_content = (
            "\n".join(thinking_text_blocks) if thinking_text_blocks else None
        )

        content: Any = text_content
        if response_format is not None:
            raw_content = f"{{{text_content}" if prefilled_json else text_content
            try:
                if prefilled_json:
                    parsed_json = json.loads(raw_content)
                    content = response_format.model_validate(parsed_json)
                else:
                    content = response_format.model_validate_json(raw_content)
            except (json.JSONDecodeError, ValidationError, ValueError):
                content = repair_response_model_json(
                    raw_content,
                    response_format,
                    model_name,
                )

        return CompletionResult(
            content=content,
            input_tokens=total_input_tokens,
            output_tokens=usage.output_tokens if usage else 0,
            cache_creation_input_tokens=cache_creation_tokens,
            cache_read_input_tokens=cache_read_tokens,
            finish_reason=response.stop_reason or "stop",
            tool_calls=tool_calls,
            thinking_content=thinking_content,
            thinking_blocks=thinking_full_blocks,
            raw_response=response,
        )

    @staticmethod
    def _supports_assistant_prefill(model: str) -> bool:
        # Claude 4-class models reject assistant-prefill and require the
        # conversation to end with a user message.
        return not model.startswith(
            (
                "claude-opus-4",
                "claude-sonnet-4",
                "claude-haiku-4",
            )
        )

    @staticmethod
    def _extract_system(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        system_messages: list[str] = []
        non_system_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "system" and isinstance(
                message.get("content"),
                str,
            ):
                system_messages.append(message["content"])
            else:
                non_system_messages.append(copy.deepcopy(message))
        return non_system_messages, system_messages

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, dict):
            return tool_choice
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice in {"any", "required"}:
            return {"type": "any"}
        if tool_choice == "none":
            return None
        return {"type": "tool", "name": tool_choice}

    @staticmethod
    def _json_mode(extra_params: dict[str, Any] | None) -> bool:
        return bool(extra_params and extra_params.get("json_mode"))
