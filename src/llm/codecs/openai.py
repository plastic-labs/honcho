"""OpenAI Chat Completions codec, also used for OpenAI-compatible endpoints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from src.llm.backend import CompletionResult
from src.llm.messages import (
    AIMessage,
    ChatMessage,
    ModelOrigin,
    ReasoningPart,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from . import RenderTarget, replayable_reasoning


class OpenAICodec:
    """`tool_calls` on the assistant turn with JSON-string arguments; one
    `role: "tool"` message per result. Reasoning goes back as
    `reasoning_details` (OpenRouter) or else `reasoning_content`."""

    def render(
        self, messages: Sequence[ChatMessage], target: RenderTarget
    ) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, SystemMessage | UserMessage):
                rendered.append({"role": message.role, "content": message.content})
            elif isinstance(message, ToolMessage):
                rendered.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
            else:
                rendered.append(self._render_ai(message, target))
        return rendered

    @staticmethod
    def _render_ai(message: AIMessage, target: RenderTarget) -> dict[str, Any]:
        if not message.tool_calls:
            return {"role": "assistant", "content": message.text or ""}
        rendered: dict[str, Any] = {
            "role": "assistant",
            "content": message.text,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.input),
                    },
                }
                for call in message.tool_calls
            ],
        }
        reasoning = replayable_reasoning(message, target)
        details = [
            part.payload
            for part in reasoning
            if part.kind == "openai_reasoning_details"
        ]
        if details:
            rendered["reasoning_details"] = details
        else:
            content = next(
                (
                    part.payload["text"]
                    for part in reasoning
                    if part.kind == "openai_reasoning_content"
                ),
                None,
            )
            if content:
                rendered["reasoning_content"] = content
        return rendered

    def parse(self, result: CompletionResult, origin: ModelOrigin) -> AIMessage:
        reasoning = [
            ReasoningPart(
                kind="openai_reasoning_details", payload=detail, origin=origin
            )
            for detail in result.reasoning_details
        ]
        if not reasoning and result.thinking_content:
            reasoning.append(
                ReasoningPart(
                    kind="openai_reasoning_content",
                    payload={"text": result.thinking_content},
                    origin=origin,
                )
            )
        return AIMessage(
            text=result.content if isinstance(result.content, str) else None,
            tool_calls=[
                ToolCall(id=call.id, name=call.name, input=call.input)
                for call in result.tool_calls
            ],
            reasoning=reasoning,
            origin=origin,
        )


__all__ = ["OpenAICodec"]
