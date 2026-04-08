from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from pydantic import BaseModel

from src.exceptions import LLMError
from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.caching import (
    GeminiCacheHandle,
    PromptCachePolicy,
    build_cache_key,
    gemini_cache_store,
)
from src.llm.structured_output import repair_response_model_json

GEMINI_BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
}


class GeminiBackend:
    """Provider backend wrapping the Google GenAI SDK."""

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
        del api_key, api_base

        contents, system_instruction = self._convert_messages(messages)
        config = self._build_config(
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
        if system_instruction:
            config["system_instruction"] = system_instruction

        cache_policy = (
            extra_params.get("cache_policy")
            if extra_params and "cache_policy" in extra_params
            else None
        )
        if isinstance(cache_policy, PromptCachePolicy):
            await self._attach_cached_content(
                model=model,
                config=config,
                cache_policy=cache_policy,
                contents=contents if isinstance(contents, list) else [],
                tools=tools,
            )
            # When cached_content is attached, the cached material is served
            # via the handle — only send the final user message as new input.
            if "cached_content" in config and isinstance(contents, list) and contents:
                contents = contents[-1:]

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config or None,
        )
        return self._normalize_response(
            response=response,
            response_format=response_format
            if isinstance(response_format, type)
            else None,
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
        del api_key, api_base

        contents, system_instruction = self._convert_messages(messages)
        config = self._build_config(
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
        if system_instruction:
            config["system_instruction"] = system_instruction

        stream = await self._client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config or None,
        )

        final_chunk = None
        async for chunk in stream:
            if chunk.text:
                yield StreamChunk(content=chunk.text)
            final_chunk = chunk

        finish_reason = "stop"
        output_tokens: int | None = None
        if (
            final_chunk
            and getattr(final_chunk, "candidates", None)
            and final_chunk.candidates[0].finish_reason
        ):
            finish_reason = final_chunk.candidates[0].finish_reason.name
        if (
            final_chunk
            and getattr(final_chunk, "usage_metadata", None)
            and getattr(final_chunk.usage_metadata, "candidates_token_count", None)
        ):
            output_tokens = final_chunk.usage_metadata.candidates_token_count or None

        yield StreamChunk(
            is_done=True,
            finish_reason=finish_reason,
            output_tokens=output_tokens,
        )

    def _build_config(
        self,
        *,
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
        config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
        }
        if temperature is not None:
            config["temperature"] = temperature
        if stop:
            config["stop_sequences"] = stop
        if tools:
            config["tools"] = self._convert_tools(tools)
        if tool_choice:
            config["tool_config"] = self._convert_tool_choice(tool_choice)
        if response_format is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_format
        elif extra_params and extra_params.get("json_mode") and not tools:
            config["response_mime_type"] = "application/json"
        thinking_config: dict[str, Any] = {}
        if thinking_budget_tokens is not None:
            thinking_config["thinking_budget"] = thinking_budget_tokens
        if thinking_effort is not None:
            thinking_config["thinking_level"] = thinking_effort
        if len(thinking_config) > 1:
            raise ValueError(
                "Gemini backend does not support sending both thinking_budget_tokens and thinking_effort in the same request"
            )
        if thinking_config:
            config["thinking_config"] = thinking_config
        for key in ("top_p", "top_k", "frequency_penalty", "presence_penalty", "seed"):
            if extra_params and key in extra_params:
                config[key] = extra_params[key]
        return config

    def _normalize_response(
        self,
        *,
        response: Any,
        response_format: type[BaseModel] | None,
        model_name: str,
    ) -> CompletionResult:
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = (
            candidate.finish_reason.name
            if candidate is not None and candidate.finish_reason
            else "stop"
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCallResult] = []
        candidate_parts = (
            cast(list[Any] | None, getattr(candidate.content, "parts", None))
            if candidate is not None and getattr(candidate, "content", None)
            else None
        )
        if isinstance(candidate_parts, list):
            for part in candidate_parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    text_parts.append(part_text)
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    function_name = getattr(function_call, "name", None)
                    function_args = getattr(function_call, "args", None)
                    if not isinstance(function_name, str):
                        continue
                    tool_calls.append(
                        ToolCallResult(
                            id=f"call_{function_name}_{len(tool_calls)}",
                            name=function_name,
                            input=dict(cast(dict[str, Any], function_args))
                            if function_args
                            else {},
                            thought_signature=getattr(part, "thought_signature", None),
                        )
                    )
        response_text = getattr(response, "text", None)
        if not text_parts and isinstance(response_text, str) and response_text:
            text_parts.append(response_text)
        response_function_calls = cast(
            list[Any] | None,
            getattr(response, "function_calls", None),
        )
        if not tool_calls and isinstance(response_function_calls, list):
            for function_call in response_function_calls:
                function_name = getattr(function_call, "name", None)
                function_args = getattr(function_call, "args", None)
                if not isinstance(function_name, str):
                    continue
                tool_calls.append(
                    ToolCallResult(
                        id=f"call_{function_name}_{len(tool_calls)}",
                        name=function_name,
                        input=dict(cast(dict[str, Any], function_args))
                        if function_args
                        else {},
                    )
                )

        content: Any = "\n".join(text_parts) if text_parts else ""
        if response_format is not None:
            parsed_response = getattr(response, "parsed", None)
            if isinstance(parsed_response, response_format):
                content = parsed_response
            elif isinstance(parsed_response, dict):
                content = response_format.model_validate(parsed_response)
            elif isinstance(parsed_response, str):
                content = response_format.model_validate_json(parsed_response)
            else:
                if finish_reason in GEMINI_BLOCKED_FINISH_REASONS:
                    raise LLMError(
                        f"Gemini response blocked (finish_reason={finish_reason})",
                        provider="google",
                        model=model_name,
                        finish_reason=finish_reason,
                    )
                raw_text = "".join(text_parts)
                content = repair_response_model_json(
                    raw_text,
                    response_format,
                    model_name,
                )
        elif (
            not content
            and not tool_calls
            and finish_reason in GEMINI_BLOCKED_FINISH_REASONS
        ):
            raise LLMError(
                f"Gemini response blocked (finish_reason={finish_reason})",
                provider="google",
                model=model_name,
                finish_reason=finish_reason,
            )

        usage = response.usage_metadata
        return CompletionResult(
            content=content,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw_response=response,
        )

    async def _attach_cached_content(
        self,
        *,
        model: str,
        config: dict[str, Any],
        cache_policy: PromptCachePolicy,
        contents: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        if cache_policy.mode != "gemini_cached_content" or not contents:
            return

        cache_key = build_cache_key(
            config=self._cache_model_config(model),
            cache_policy=cache_policy,
            cacheable_messages=contents,
            tools=tools,
        )
        cached_handle = gemini_cache_store.get(cache_key)
        if cached_handle is None:
            ttl_seconds = cache_policy.ttl_seconds or 300
            cached_content = await self._client.aio.caches.create(
                model=model,
                config={
                    "contents": contents,
                    "system_instruction": config.get("system_instruction"),
                    "tools": config.get("tools"),
                    "tool_config": config.get("tool_config"),
                    "ttl": f"{ttl_seconds}s",
                },
            )
            expires_at = getattr(cached_content, "expire_time", None)
            if expires_at is None:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            cached_handle = gemini_cache_store.set(
                GeminiCacheHandle(
                    key=cache_key,
                    cached_content_name=cached_content.name,
                    expires_at=expires_at,
                )
            )
        # Once a cached-content handle is attached, Gemini rejects repeating
        # system/tool configuration on the generate call.
        config.pop("system_instruction", None)
        config.pop("tools", None)
        config.pop("tool_config", None)
        config["cached_content"] = cached_handle.cached_content_name

    @staticmethod
    def _cache_model_config(model: str):
        from src.config import ModelConfig

        return ModelConfig(transport="gemini", model=model)

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]] | str, str | None]:
        system_messages: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                if isinstance(message.get("content"), str):
                    system_messages.append(message["content"])
                continue

            if role == "assistant":
                role = "model"

            if isinstance(message.get("parts"), list):
                message_copy = message.copy()
                message_copy["role"] = role
                contents.append(message_copy)
                continue

            if isinstance(message.get("content"), str):
                contents.append({"role": role, "parts": [{"text": message["content"]}]})
                continue

            if isinstance(message.get("content"), list):
                parts: list[dict[str, Any]] = []
                for block in message["content"]:
                    if block.get("type") == "text":
                        parts.append({"text": block["text"]})
                if parts:
                    contents.append({"role": role, "parts": parts})

        system_instruction = "\n\n".join(system_messages) if system_messages else None
        return contents, system_instruction

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if tools and "function_declarations" in tools[0]:
            return tools
        return [
            {
                "function_declarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    }
                    for tool in tools
                ]
            }
        ]

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(tool_choice, dict) and "name" in tool_choice:
            return {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [tool_choice["name"]],
                }
            }
        if tool_choice == "auto":
            return {"function_calling_config": {"mode": "AUTO"}}
        if tool_choice in {"any", "required"}:
            return {"function_calling_config": {"mode": "ANY"}}
        if tool_choice == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        return {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [tool_choice],
            }
        }
