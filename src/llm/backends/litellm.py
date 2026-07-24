"""LiteLLM provider backend.

Routes to 100+ LLM providers via a unified interface using provider-prefixed
model names (e.g. ``anthropic/claude-sonnet-4-6``, ``gemini/gemini-2.5-flash``).

Install: ``pip install litellm``
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from src.exceptions import ValidationException
from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult

logger = logging.getLogger(__name__)


class LiteLLMBackend:
    """Provider backend wrapping litellm.acompletion."""

    def __init__(self, api_key: str | None = None, api_base: str | None = None) -> None:
        self._api_key = api_key
        self._api_base = api_base

    def _base_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"drop_params": True}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        return kwargs

    @staticmethod
    def _import_litellm() -> Any:
        try:
            import litellm
        except ModuleNotFoundError as exc:
            raise ValidationException(
                "LiteLLM transport requires optional dependency 'litellm'. "
                "Install with: pip install honcho[litellm]"
            ) from exc
        return litellm

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
        litellm = self._import_litellm()

        params = self._build_params(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
        )

        response = await litellm.acompletion(**params)
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
        litellm = self._import_litellm()

        params = self._build_params(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
        )
        params["stream"] = True

        response_stream = await litellm.acompletion(**params)
        finish_reason: str | None = None
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(content=chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            usage = getattr(chunk, "usage", None)
            if usage:
                yield StreamChunk(
                    is_done=True,
                    finish_reason=finish_reason,
                    output_tokens=getattr(usage, "completion_tokens", None),
                )
                return

        if finish_reason:
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
        response_format: type[BaseModel] | dict[str, Any] | None,
        thinking_effort: str | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **self._base_kwargs(),
        }
        if temperature is not None:
            params["temperature"] = temperature
        if stop:
            params["stop"] = stop
        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                params["response_format"] = response_format
            else:
                params["response_format"] = response_format
        if thinking_effort:
            params["reasoning_effort"] = thinking_effort
        if extra_params:
            for key in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
                if key in extra_params:
                    params[key] = extra_params[key]
        return params

    @staticmethod
    def _normalize_response(response: Any) -> CompletionResult:
        usage = getattr(response, "usage", None)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        tool_calls: list[ToolCallResult] = []
        for tc in getattr(message, "tool_calls", None) or []:
            tool_input: dict[str, Any] = {}
            if tc.function.arguments:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Malformed tool arguments for %s (id=%s)",
                        tc.function.name,
                        tc.id,
                    )
            tool_calls.append(
                ToolCallResult(id=tc.id, name=tc.function.name, input=tool_input)
            )

        return CompletionResult(
            content=getattr(message, "content", "") or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            finish_reason=finish_reason or "stop",
            tool_calls=tool_calls,
            raw_response=response,
        )

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tools or tools[0].get("type") == "function":
            return tools
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
