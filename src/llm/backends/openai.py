from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import BadRequestError, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from src.exceptions import ValidationException
from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.structured_output import (
    repair_response_model_json,
    validate_structured_output,
)

logger = logging.getLogger(__name__)


def _uses_max_completion_tokens(model: str) -> bool:
    """OpenAI reasoning models (gpt-5 family + o-series) require
    ``max_completion_tokens`` instead of the classic ``max_tokens`` parameter.

    Matches: gpt-5, gpt-5-anything, gpt-5.anything (incl. gpt-5.4, gpt-5.4-mini),
    o1*, o3*, o4*. Anything else (gpt-4.x, gpt-4o, chat models on proxies)
    stays on ``max_tokens``.
    """
    m = model.lower()
    if m == "gpt-5" or m.startswith("gpt-5-") or m.startswith("gpt-5."):
        return True
    for prefix in ("o1", "o3", "o4"):
        if m == prefix or m.startswith(prefix + "-"):
            return True
    return False


def extract_openai_reasoning_content(response: Any) -> str | None:
    try:
        message = response.choices[0].message
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            reasoning_parts: list[str] = []
            for detail in message.reasoning_details:
                detail_content = getattr(detail, "content", None)
                if isinstance(detail_content, str) and detail_content:
                    reasoning_parts.append(detail_content)
                elif isinstance(detail, dict):
                    detail_dict = cast(dict[str, Any], detail)
                    dict_content = detail_dict.get("content")
                    if isinstance(dict_content, str) and dict_content:
                        reasoning_parts.append(dict_content)
            if reasoning_parts:
                return "\n".join(reasoning_parts)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return message.reasoning_content
    except (AttributeError, IndexError, TypeError):
        return None
    return None


def extract_openai_reasoning_details(response: Any) -> list[dict[str, Any]]:
    try:
        message = response.choices[0].message
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            details: list[dict[str, Any]] = []
            for detail in message.reasoning_details:
                if hasattr(detail, "model_dump"):
                    dumped = detail.model_dump()
                    if isinstance(dumped, dict):
                        details.append(cast(dict[str, Any], dumped))
                elif isinstance(detail, dict):
                    details.append(cast(dict[str, Any], detail))
                else:
                    detail_content = getattr(detail, "content", None)
                    if isinstance(detail_content, str) and detail_content:
                        details.append({"content": detail_content})
            return details
    except (AttributeError, IndexError, TypeError):
        return []
    return []


def extract_openai_cache_tokens(usage: Any) -> tuple[int, int]:
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


class OpenAIBackend:
    """Provider backend wrapping AsyncOpenAI."""

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
        if thinking_budget_tokens is not None:
            raise ValidationException(
                "OpenAI backend does not support thinking_budget_tokens; use thinking_effort instead"
            )

        params = self._build_params(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
        )

        if isinstance(response_format, type):
            params["response_format"] = response_format
            try:
                response = await self._client.chat.completions.parse(**params)
            except LengthFinishReasonError as exc:
                truncated = exc.completion
                raw_content = truncated.choices[0].message.content or ""
                content = repair_response_model_json(
                    raw_content,
                    response_format,
                    model,
                )
                return self._normalize_response(
                    truncated,
                    content_override=content,
                )
            except (BadRequestError, json.JSONDecodeError, ValidationError):
                fallback_response = await self._create_structured_response(
                    params=params,
                    response_format=response_format,
                )
                content = self._parse_or_repair_structured_content(
                    fallback_response,
                    response_format,
                    model,
                )
                return self._normalize_response(
                    fallback_response,
                    content_override=content,
                )
            parsed = response.choices[0].message.parsed
            raw_content = response.choices[0].message.content or ""
            if parsed is None and raw_content:
                content = repair_response_model_json(
                    raw_content,
                    response_format,
                    model,
                )
                return self._normalize_response(response, content_override=content)
            if parsed is None:
                refusal = getattr(response.choices[0].message, "refusal", None)
                if refusal:
                    return self._normalize_response(
                        response,
                        content_override=refusal,
                    )
                raise ValueError("No parsed content in structured response")
            return self._normalize_response(
                response,
                content_override=validate_structured_output(parsed, response_format),
            )
        if response_format is not None:
            params["response_format"] = response_format

        if extra_params and extra_params.get("json_mode"):
            params["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**params)
        return self._normalize_response(response)

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
        if thinking_budget_tokens is not None:
            raise ValidationException(
                "OpenAI backend does not support thinking_budget_tokens; use thinking_effort instead"
            )

        params = self._build_params(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
        )
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        if isinstance(response_format, type):
            # parse() supports BaseModel types but streaming create() does not —
            # convert to a json_schema dict so the streaming path works.
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                },
            }
        elif response_format is not None:
            params["response_format"] = response_format
        elif extra_params and extra_params.get("json_mode"):
            params["response_format"] = {"type": "json_object"}

        response_stream = await self._client.chat.completions.create(**params)
        finish_reason: str | None = None
        usage_chunk_received = False
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(content=chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if hasattr(chunk, "usage") and chunk.usage:
                yield StreamChunk(
                    is_done=True,
                    finish_reason=finish_reason,
                    output_tokens=chunk.usage.completion_tokens,
                )
                usage_chunk_received = True

        if not usage_chunk_received and finish_reason:
            yield StreamChunk(is_done=True, finish_reason=finish_reason)

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
        thinking_effort: str | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if _uses_max_completion_tokens(model):
            params["max_completion_tokens"] = max_tokens
            if extra_params and extra_params.get("verbosity"):
                params["verbosity"] = extra_params["verbosity"]
        else:
            params["max_tokens"] = max_tokens

        if temperature is not None:
            params["temperature"] = temperature

        if thinking_effort:
            params["reasoning_effort"] = thinking_effort

        if stop:
            params["stop"] = stop
        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
        if extra_params:
            for key in (
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "seed",
            ):
                if key in extra_params:
                    params[key] = extra_params[key]
        return params

    def _normalize_response(
        self,
        response: Any,
        *,
        content_override: Any | None = None,
    ) -> CompletionResult:
        usage = response.usage
        finish_reason = response.choices[0].finish_reason
        tool_calls: list[ToolCallResult] = []
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                tool_input: dict[str, Any] = {}
                if tool_call.function.arguments:
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "Malformed tool arguments for %s (id=%s): %s",
                            tool_call.function.name,
                            tool_call.id,
                            tool_call.function.arguments,
                        )
                tool_calls.append(
                    ToolCallResult(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=tool_input,
                    )
                )

        cache_creation, cache_read = extract_openai_cache_tokens(usage)
        return CompletionResult(
            content=content_override
            if content_override is not None
            else (message.content or ""),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            finish_reason=finish_reason or "stop",
            tool_calls=tool_calls,
            thinking_content=extract_openai_reasoning_content(response),
            reasoning_details=extract_openai_reasoning_details(response),
            raw_response=response,
        )

    async def _create_structured_response(
        self,
        *,
        params: dict[str, Any],
        response_format: type[BaseModel],
    ) -> Any:
        structured_params = dict(params)
        structured_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
            },
        }
        return await self._client.chat.completions.create(**structured_params)

    @staticmethod
    def _parse_or_repair_structured_content(
        response: Any,
        response_format: type[BaseModel],
        model: str,
    ) -> BaseModel | str:
        raw_content = response.choices[0].message.content or ""
        if raw_content:
            return repair_response_model_json(raw_content, response_format, model)
        refusal = getattr(response.choices[0].message, "refusal", None)
        if refusal:
            return refusal
        raise ValidationException(
            "No raw content available for structured output repair"
        )

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tools or tools[0].get("type") == "function":
            return tools
        # Tool schemas in src/utils/agent_tools.py use optional fields with
        # defaults and don't declare additionalProperties: false. OpenAI's
        # strict function-calling mode forbids both, so we intentionally
        # don't set strict: True. Standard function calling on GPT-4.x /
        # GPT-5 remains reliable, and this stays compatible with
        # OpenAI-compatible proxies (OpenRouter, Together, vLLM, Ollama)
        # whose strict-mode support is inconsistent.
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
