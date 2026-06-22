from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
from openai import OpenAIError
from pydantic import BaseModel

from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.structured_output import (
    repair_response_model_json,
    validate_structured_output,
)

logger = logging.getLogger(__name__)

DEFAULT_CODEX_INSTRUCTIONS = "Follow the user's instructions."


class CodexResponsesBackend:
    """Provider backend for ChatGPT Codex subscription OAuth.

    The Codex subscription endpoint is OpenAI Responses-shaped, not
    Chat Completions-shaped. This adapter keeps Honcho's internal
    chat-style history and tool loop intact while issuing Responses calls.
    """

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
        extra_params: dict[str, Any] | None = None,
    ) -> CompletionResult:
        params = self._build_params(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
        )
        async with self._client.responses.stream(**params) as stream:
            async for _event in stream:
                pass
            response = await stream.get_final_response()
        return self._normalize_response(
            response,
            response_format=response_format,
            model=model,
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
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        params = self._build_params(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_budget_tokens=thinking_budget_tokens,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
        )

        finish_reason: str | None = None
        output_tokens: int | None = None
        async with self._client.responses.stream(**params) as stream:
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type in {
                    "response.output_text.delta",
                    "response.refusal.delta",
                }:
                    delta = getattr(event, "delta", "")
                    if isinstance(delta, str) and delta:
                        yield StreamChunk(content=delta)
                elif event_type in {"response.completed", "response.incomplete"}:
                    response = getattr(event, "response", None)
                    finish_reason = self._finish_reason(response)
                    output_tokens = self._usage_output_tokens(getattr(response, "usage", None))

            if output_tokens is None:
                try:
                    final_response = await stream.get_final_response()
                    output_tokens = self._usage_output_tokens(
                        getattr(final_response, "usage", None)
                    )
                    finish_reason = finish_reason or self._finish_reason(final_response)
                except OpenAIError as exc:
                    logger.debug(
                        "Codex Responses stream final usage failed with OpenAI error: %s",
                        exc,
                        exc_info=True,
                    )
                except httpx.RequestError as exc:
                    logger.debug(
                        "Codex Responses stream final usage failed with request error: %s",
                        exc,
                        exc_info=True,
                    )
                except httpx.HTTPError as exc:
                    logger.debug(
                        "Codex Responses stream final usage failed with HTTP error: %s",
                        exc,
                        exc_info=True,
                    )
                except RuntimeError as exc:
                    logger.debug(
                        "Codex Responses stream final usage failed during stream finalization: %s",
                        exc,
                        exc_info=True,
                    )

        yield StreamChunk(
            is_done=True,
            finish_reason=finish_reason or "stop",
            output_tokens=output_tokens,
        )

    def _build_params(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: type[BaseModel] | dict[str, Any] | None,
        thinking_budget_tokens: int | None,
        thinking_effort: str | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        _ = thinking_budget_tokens
        instructions, input_items = self._messages_to_responses(messages)
        params: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if temperature is not None:
            params["temperature"] = temperature
        params["instructions"] = instructions or DEFAULT_CODEX_INSTRUCTIONS
        if stop:
            params["stop"] = stop

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools
            converted_tool_choice = self._convert_tool_choice(tool_choice)
            if converted_tool_choice is not None:
                params["tool_choice"] = converted_tool_choice

        reasoning: dict[str, Any] = {}
        if thinking_effort and thinking_effort != "none":
            reasoning["effort"] = self._codex_reasoning_effort(thinking_effort)
            reasoning["summary"] = "auto"
        if reasoning:
            params["reasoning"] = reasoning

        text_config = self._response_text_config(response_format, extra_params)
        if text_config:
            params["text"] = text_config

        if extra_params:
            for key in ("top_p", "metadata", "parallel_tool_calls", "service_tier"):
                if key in extra_params:
                    params[key] = extra_params[key]
            if "truncation" in extra_params:
                params["truncation"] = extra_params["truncation"]
            if "store" in extra_params:
                params["store"] = bool(extra_params["store"])
        return params

    @staticmethod
    def _codex_reasoning_effort(effort: str) -> str:
        if effort == "minimal":
            return "low"
        if effort in {"xhigh", "max"}:
            return "high"
        return effort

    def _normalize_response(
        self,
        response: Any,
        *,
        response_format: type[BaseModel] | dict[str, Any] | None,
        model: str,
    ) -> CompletionResult:
        content = self._response_text(response)
        if isinstance(response_format, type):
            parsed = repair_response_model_json(content, response_format, model)
            content = validate_structured_output(parsed, response_format)

        usage = getattr(response, "usage", None)
        return CompletionResult(
            content=content,
            input_tokens=self._usage_input_tokens(usage),
            output_tokens=self._usage_output_tokens(usage) or 0,
            cache_read_input_tokens=self._usage_cache_read_tokens(usage),
            finish_reason=self._finish_reason(response),
            tool_calls=self._response_tool_calls(response),
            reasoning_details=self._response_reasoning_details(response),
            raw_response=response,
        )

    @staticmethod
    def _messages_to_responses(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                if content:
                    instructions.append(str(content))
                continue
            if role in {"user", "assistant"}:
                if content is not None and content != "":
                    items.append({"role": role, "content": content})
                items.extend(CodexResponsesBackend._message_reasoning_items(message))
                raw_tool_calls = message.get("tool_calls")
                tool_calls: list[Any] = (
                    cast(list[Any], raw_tool_calls)
                    if isinstance(raw_tool_calls, list)
                    else []
                )
                for tool_call_value in tool_calls:
                    if not isinstance(tool_call_value, dict):
                        continue
                    tool_call = cast(dict[str, Any], tool_call_value)
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        continue
                    function = cast(dict[str, Any], function)
                    name = function.get("name")
                    if not isinstance(name, str) or not name.strip():
                        continue
                    arguments = function.get("arguments", "{}")
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    elif not isinstance(arguments, str):
                        arguments = str(arguments)
                    call_id = tool_call.get("id")
                    if not isinstance(call_id, str) or not call_id.strip():
                        call_id = f"call_{len(items)}"
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments or "{}",
                        }
                    )
                continue
            if role == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": str(content or ""),
                    }
                )
        return "\n\n".join(instructions) if instructions else None, items

    @staticmethod
    def _message_reasoning_items(message: dict[str, Any]) -> list[dict[str, Any]]:
        raw_reasoning = message.get("reasoning_details")
        if raw_reasoning is None:
            raw_reasoning = message.get("reasoning")
        if not isinstance(raw_reasoning, list):
            return []

        items: list[dict[str, Any]] = []
        for value in cast(list[Any], raw_reasoning):
            if isinstance(value, dict):
                items.append(cast(dict[str, Any], value))
        return items

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for item in tools:
            if item.get("type") == "function" and isinstance(item.get("function"), dict):
                function = item["function"]
                converted.append(
                    {
                        "type": "function",
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": function.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        ),
                        "strict": False,
                    }
                )
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append(
                {
                    "type": "function",
                    "name": name,
                    "description": item.get("description", ""),
                    "parameters": item.get(
                        "input_schema",
                        {"type": "object", "properties": {}},
                    ),
                    "strict": False,
                }
            )
        return converted or None

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if isinstance(function, dict):
                function = cast(dict[str, Any], function)
                name = function.get("name")
                if isinstance(name, str):
                    return {"type": "function", "name": name}
            return tool_choice
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {"type": "function", "name": tool_choice}

    @staticmethod
    def _response_text_config(
        response_format: type[BaseModel] | dict[str, Any] | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        text_config: dict[str, Any] = {}
        verbosity = (extra_params or {}).get("verbosity")
        if verbosity:
            text_config["verbosity"] = verbosity
        if isinstance(response_format, type):
            text_config["format"] = {
                "type": "json_schema",
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
                "strict": False,
            }
        elif isinstance(response_format, dict):
            text_config["format"] = response_format
        elif extra_params and extra_params.get("json_mode"):
            text_config["format"] = {"type": "json_object"}
        return text_config or None

    @staticmethod
    def _response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content_part in getattr(item, "content", []) or []:
                text = getattr(content_part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _response_tool_calls(response: Any) -> list[ToolCallResult]:
        tool_calls: list[ToolCallResult] = []
        for item in CodexResponsesBackend._response_output_items(response):
            if getattr(item, "type", None) != "function_call":
                continue
            call_id = getattr(item, "call_id", None) or getattr(item, "id", "")
            name = getattr(item, "name", "")
            raw_arguments = getattr(item, "arguments", "{}") or "{}"
            try:
                decoded_arguments: Any = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError):
                decoded_arguments = {}
            arguments = (
                cast(dict[str, Any], decoded_arguments)
                if isinstance(decoded_arguments, dict)
                else {}
            )
            tool_calls.append(
                ToolCallResult(
                    id=str(call_id),
                    name=str(name),
                    input=arguments,
                )
            )
        return tool_calls

    @staticmethod
    def _response_output_items(response: Any) -> list[Any]:
        raw_items: Any = getattr(response, "output", None)
        return cast(list[Any], raw_items) if isinstance(raw_items, list) else []

    @staticmethod
    def _response_reasoning_details(response: Any) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for item in CodexResponsesBackend._response_output_items(response):
            if isinstance(item, dict):
                item_dict = cast(dict[str, Any], item)
                item_type = item_dict.get("type")
                if isinstance(item_type, str) and item_type.startswith("reasoning"):
                    details.append(item_dict)
                continue
            if getattr(item, "type", None) != "reasoning":
                continue
            model_dump = getattr(item, "model_dump", None)
            if callable(model_dump):
                dumped: Any = model_dump()
                if isinstance(dumped, dict):
                    details.append(cast(dict[str, Any], dumped))
                continue
            detail: dict[str, Any] = {"type": "reasoning"}
            for key in ("id", "summary", "content", "encrypted_content"):
                value = getattr(item, key, None)
                if value is not None:
                    detail[key] = value
            details.append(detail)
        return details

    @staticmethod
    def _usage_input_tokens(usage: Any) -> int:
        return int(
            getattr(usage, "input_tokens", 0)
            or getattr(usage, "prompt_tokens", 0)
            or 0
        )

    @staticmethod
    def _usage_output_tokens(usage: Any) -> int | None:
        value = getattr(usage, "output_tokens", None)
        if value is None:
            value = getattr(usage, "completion_tokens", None)
        return int(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _usage_cache_read_tokens(usage: Any) -> int:
        details = getattr(usage, "input_tokens_details", None)
        if details is None:
            details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
        return int(cached or 0)

    @staticmethod
    def _finish_reason(response: Any) -> str:
        status = getattr(response, "status", None)
        if status == "completed":
            return "stop"
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None)
            return str(reason or "length")
        return str(status or "stop")
