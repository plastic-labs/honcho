"""
Google Gemini provider adapter.

This adapter encapsulates Gemini-specific behaviors:
- `system_instruction` is used instead of system-role messages in `contents`
- tool calling uses `function_call`/`function_response` parts
- `thought_signature` must be preserved and replayed for multi-turn tool use
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from google import genai
from google.genai.types import (
    ContentListUnionDict,
    GenerateContentConfigDict,
    GenerateContentResponse,
)
from pydantic import BaseModel

from src.utils.llm.adapters.base import ProviderAdapter
from src.utils.llm.models import HonchoLLMCallResponse, HonchoLLMCallStreamChunk
from src.utils.types import SupportedProviders

logger = logging.getLogger(__name__)


class GoogleAdapter(ProviderAdapter):
    """ProviderAdapter implementation for Google Gemini SDK clients."""

    provider: SupportedProviders = "google"

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-style tool schemas to Gemini function_declarations."""
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

    def format_assistant_tool_message(
        self,
        *,
        content: Any,
        tool_calls: list[dict[str, Any]],
        thinking_blocks: list[dict[str, Any]] | None = None,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Format a Gemini model message containing function calls."""
        _ = (thinking_blocks, reasoning_details)

        parts: list[dict[str, Any]] = []
        if isinstance(content, str) and content:
            parts.append({"text": content})

        for tool_call in tool_calls:
            part_data: dict[str, Any] = {
                "function_call": {"name": tool_call["name"], "args": tool_call["input"]}
            }
            if "thought_signature" in tool_call:
                part_data["thought_signature"] = tool_call["thought_signature"]
            parts.append(part_data)

        return {"role": "model", "parts": parts}

    def append_tool_results(
        self,
        *,
        tool_results: list[dict[str, Any]],
        conversation_messages: list[dict[str, Any]],
    ) -> None:
        """Append tool results in Gemini `function_response` part format."""
        response_parts: list[dict[str, Any]] = []
        for tr in tool_results:
            response_parts.append(
                {
                    "function_response": {
                        "name": tr["tool_name"],
                        "response": {"result": str(tr["result"])},
                    }
                }
            )
        conversation_messages.append({"role": "user", "parts": response_parts})

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
        """Perform a non-streaming Gemini call and normalize the response."""
        _ = (
            provider,
            max_tokens,
            stop_seqs,
            reasoning_effort,
            verbosity,
            thinking_budget_tokens,
        )

        gemini_client = cast(genai.Client, client)

        system_messages: list[str] = []
        non_system_messages: list[dict[str, Any]] = []

        gemini_config: dict[str, Any] = {}

        if temperature is not None:
            gemini_config["temperature"] = temperature

        if tools:
            gemini_config["tools"] = tools
            if tool_choice:
                if tool_choice == "auto":
                    gemini_config["tool_config"] = {
                        "function_calling_config": {"mode": "AUTO"}
                    }
                elif tool_choice == "any" or tool_choice == "required":
                    gemini_config["tool_config"] = {
                        "function_calling_config": {"mode": "ANY"}
                    }
                elif tool_choice == "none":
                    gemini_config["tool_config"] = {
                        "function_calling_config": {"mode": "NONE"}
                    }
                elif isinstance(tool_choice, dict) and "name" in tool_choice:
                    gemini_config["tool_config"] = {
                        "function_calling_config": {
                            "mode": "ANY",
                            "allowed_function_names": [tool_choice["name"]],
                        }
                    }

        if response_model is None:
            if json_mode and not tools:
                gemini_config["response_mime_type"] = "application/json"

            if messages:
                for msg in messages:
                    if msg.get("role") == "system":
                        if isinstance(msg.get("content"), str):
                            system_messages.append(msg["content"])
                    else:
                        non_system_messages.append(msg)

                if system_messages:
                    gemini_config["system_instruction"] = "\n\n".join(system_messages)

                gemini_contents: list[dict[str, Any]] = []
                for msg in non_system_messages:
                    role = msg.get("role", "user")
                    if role == "assistant":
                        role = "model"

                    if isinstance(msg.get("content"), str):
                        gemini_contents.append(
                            {"role": role, "parts": [{"text": msg["content"]}]}
                        )
                    elif isinstance(msg.get("parts"), list):
                        msg_copy = msg.copy()
                        msg_copy["role"] = role
                        gemini_contents.append(msg_copy)
                    elif isinstance(msg.get("content"), list):
                        continue
                    else:
                        continue

                contents: ContentListUnionDict = cast(
                    ContentListUnionDict, gemini_contents
                )
            else:
                contents = prompt

            gemini_response: GenerateContentResponse = (
                await gemini_client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=cast(GenerateContentConfigDict, cast(object, gemini_config))
                    if gemini_config
                    else None,
                )
            )

            text_parts: list[str] = []
            gemini_tool_calls: list[dict[str, Any]] = []

            if gemini_response.candidates and gemini_response.candidates[0].content:
                for part in gemini_response.candidates[0].content.parts or []:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_call_data: dict[str, Any] = {
                            "id": f"call_{fc.name}_{len(gemini_tool_calls)}",
                            "name": fc.name,
                            "input": dict(fc.args) if fc.args else {},
                        }
                        if (
                            hasattr(part, "thought_signature")
                            and part.thought_signature
                        ):
                            tool_call_data["thought_signature"] = part.thought_signature
                        gemini_tool_calls.append(tool_call_data)

            text_content = "\n".join(text_parts) if text_parts else ""
            input_token_count = (
                gemini_response.usage_metadata.prompt_token_count or 0
                if gemini_response.usage_metadata
                else 0
            )
            output_token_count = (
                gemini_response.usage_metadata.candidates_token_count or 0
                if gemini_response.usage_metadata
                else 0
            )
            finish_reason = (
                gemini_response.candidates[0].finish_reason.name
                if gemini_response.candidates
                and gemini_response.candidates[0].finish_reason
                else "stop"
            )

            return HonchoLLMCallResponse(
                content=text_content,
                input_tokens=input_token_count,
                output_tokens=output_token_count,
                finish_reasons=[finish_reason],
                tool_calls_made=gemini_tool_calls,
            )

        gemini_config["response_mime_type"] = "application/json"
        gemini_config["response_schema"] = response_model

        gemini_response = await gemini_client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=cast(GenerateContentConfigDict, cast(object, gemini_config)),
        )

        input_token_count = (
            gemini_response.usage_metadata.prompt_token_count or 0
            if gemini_response.usage_metadata
            else 0
        )
        output_token_count = (
            gemini_response.usage_metadata.candidates_token_count or 0
            if gemini_response.usage_metadata
            else 0
        )
        finish_reason = (
            gemini_response.candidates[0].finish_reason.name
            if gemini_response.candidates
            and gemini_response.candidates[0].finish_reason
            else "stop"
        )

        if not isinstance(gemini_response.parsed, response_model):
            raise ValueError(
                f"Parsed content does not match the response model: {gemini_response.parsed} != {response_model}"
            )

        return HonchoLLMCallResponse(
            content=gemini_response.parsed,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            finish_reasons=[finish_reason],
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
        """Stream a Gemini response and normalize chunks."""
        _ = (
            provider,
            prompt,
            max_tokens,
            temperature,
            reasoning_effort,
            verbosity,
            thinking_budget_tokens,
        )

        gemini_client = cast(genai.Client, client)

        prompt_text = messages[0]["content"] if messages else ""

        if response_model is not None:
            response_stream = await gemini_client.aio.models.generate_content_stream(
                model=model,
                contents=prompt_text,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_model,
                },
            )
        else:
            response_stream = await gemini_client.aio.models.generate_content_stream(
                model=model,
                contents=prompt_text,
                config={
                    "response_mime_type": "application/json" if json_mode else None
                },
            )

        final_chunk = None
        async for chunk in response_stream:
            if chunk.text:
                yield HonchoLLMCallStreamChunk(content=chunk.text)
            final_chunk = chunk

        finish_reason = "stop"
        gemini_output_tokens: int | None = None

        if (
            final_chunk
            and hasattr(final_chunk, "candidates")
            and final_chunk.candidates
            and hasattr(final_chunk.candidates[0], "finish_reason")
            and final_chunk.candidates[0].finish_reason
        ):
            finish_reason = final_chunk.candidates[0].finish_reason.name

        if (
            final_chunk
            and hasattr(final_chunk, "usage_metadata")
            and final_chunk.usage_metadata
            and hasattr(final_chunk.usage_metadata, "candidates_token_count")
        ):
            gemini_output_tokens = (
                final_chunk.usage_metadata.candidates_token_count or None
            )

        yield HonchoLLMCallStreamChunk(
            content="",
            is_done=True,
            finish_reasons=[finish_reason],
            output_tokens=gemini_output_tokens,
        )
