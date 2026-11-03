"""OpenAI Chat Completions codec, also used for OpenAI-compatible endpoints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from src.llm.backend import CompletionResult
from src.llm.capabilities import capabilities_for
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

from . import RenderedHistory, RenderTarget, replayable_reasoning


class OpenAICodec:
    """`tool_calls` on the assistant turn with JSON-string arguments; one
    `role: "tool"` message per result. Reasoning goes back as
    `reasoning_details` (OpenRouter) or else `reasoning_content`. For a model
    that takes no user turn after a tool result, a user message there is
    appended to that result instead."""

    def render(
        self, messages: Sequence[ChatMessage], target: RenderTarget
    ) -> RenderedHistory:
        rendered: list[dict[str, Any]] = []
        user_after_tool = capabilities_for(
            target.transport, target.model
        ).user_after_tool
        for message in messages:
            previous = rendered[-1] if rendered else None
            if (
                isinstance(message, UserMessage)
                and not user_after_tool
                and previous is not None
                and previous["role"] == "tool"
            ):
                # The loop's notes after a round (a budget line, the synthesis
                # prompt, the empty-answer nudge) ride on the last result.
                previous["content"] = f"{previous['content']}\n\n{message.content}"
            elif isinstance(message, SystemMessage | UserMessage):
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
        return RenderedHistory(rendered)

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
