from __future__ import annotations

import json
from typing import Any, Protocol

from .backend import CompletionResult


class HistoryAdapter(Protocol):
    def format_assistant_tool_message(
        self,
        result: CompletionResult,
    ) -> dict[str, Any]: ...

    def format_tool_results(
        self,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


class AnthropicHistoryAdapter:
    def format_assistant_tool_message(
        self,
        result: CompletionResult,
    ) -> dict[str, Any]:
        content_blocks: list[dict[str, Any]] = []
        if result.thinking_blocks:
            content_blocks.extend(result.thinking_blocks)
        if isinstance(result.content, str) and result.content:
            content_blocks.append({"type": "text", "text": result.content})
        for tool_call in result.tool_calls:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.input,
                }
            )
        return {"role": "assistant", "content": content_blocks}

    def format_tool_results(
        self,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr["tool_id"],
                        "content": str(tr["result"]),
                        "is_error": tr.get("is_error", False),
                    }
                    for tr in tool_results
                ],
            }
        ]


class GeminiHistoryAdapter:
    def format_assistant_tool_message(
        self,
        result: CompletionResult,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        if isinstance(result.content, str) and result.content:
            parts.append({"text": result.content})
        for tool_call in result.tool_calls:
            part: dict[str, Any] = {
                "function_call": {
                    "name": tool_call.name,
                    "args": tool_call.input,
                }
            }
            if tool_call.thought_signature is not None:
                part["thought_signature"] = tool_call.thought_signature
            parts.append(part)
        return {"role": "model", "parts": parts}

    def format_tool_results(
        self,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": tr["tool_name"],
                            "response": {"result": str(tr["result"])},
                        }
                    }
                    for tr in tool_results
                ],
            }
        ]


class OpenAIHistoryAdapter:
    def format_assistant_tool_message(
        self,
        result: CompletionResult,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": result.content if isinstance(result.content, str) else None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.input),
                    },
                }
                for tool_call in result.tool_calls
            ],
        }
        if result.reasoning_details:
            message["reasoning_details"] = result.reasoning_details
        return message

    def format_tool_results(
        self,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": tr["tool_id"],
                "content": str(tr["result"]),
            }
            for tr in tool_results
        ]
