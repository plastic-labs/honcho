from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ValidationError

from src.llm.backend import CompletionResult, StreamChunk
from src.llm.backends.openai import extract_openai_cache_tokens
from src.llm.structured_output import repair_response_model_json


class GroqBackend:
    """Thin wrapper around Groq's OpenAI-compatible chat completions."""

    def __init__(self, client: Any) -> None:
        self._client = client

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
        del tools, tool_choice, api_key, api_base
        if thinking_budget_tokens is not None or thinking_effort is not None:
            raise ValueError(
                "Groq backend does not support thinking_budget_tokens or thinking_effort"
            )

        params: dict[str, Any] = {
            "model": self._strip_prefix(model),
            "messages": messages,
            "max_tokens": max_output_tokens or max_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if stop:
            params["stop"] = stop
        if response_format is not None:
            params["response_format"] = response_format
        elif extra_params and extra_params.get("json_mode"):
            params["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**params)
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")
        usage = response.usage
        finish_reason = response.choices[0].finish_reason
        cache_creation, cache_read = extract_openai_cache_tokens(usage)

        content: Any = response.choices[0].message.content
        if response_format is not None:
            try:
                content = response_format.model_validate(
                    json.loads(response.choices[0].message.content)
                )
            except (json.JSONDecodeError, ValidationError, ValueError):
                content = repair_response_model_json(
                    response.choices[0].message.content or "",
                    response_format,
                    self._strip_prefix(model),
                )

        return CompletionResult(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            finish_reason=finish_reason or "stop",
            raw_response=response,
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
        del tools, tool_choice, api_key, api_base
        if thinking_budget_tokens is not None or thinking_effort is not None:
            raise ValueError(
                "Groq backend does not support thinking_budget_tokens or thinking_effort"
            )

        params: dict[str, Any] = {
            "model": self._strip_prefix(model),
            "messages": messages,
            "max_tokens": max_output_tokens or max_tokens,
            "stream": True,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if stop:
            params["stop"] = stop
        if response_format is not None:
            params["response_format"] = response_format
        elif extra_params and extra_params.get("json_mode"):
            params["response_format"] = {"type": "json_object"}

        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(content=chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                yield StreamChunk(
                    is_done=True,
                    finish_reason=chunk.choices[0].finish_reason,
                )

    @staticmethod
    def _strip_prefix(model: str) -> str:
        return model.split("/", 1)[1] if "/" in model else model
