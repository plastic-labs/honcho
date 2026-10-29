"""Gemini codec."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.llm.backend import CompletionResult
from src.llm.messages import (
    AIMessage,
    ChatMessage,
    ModelOrigin,
    SystemMessage,
    ToolCall,
    UserMessage,
)

from . import RenderTarget, can_replay, group_tool_results


class GeminiCodec:
    """`function_call` parts on a `model` turn, carrying Gemini's thought
    signatures; each round's results as `function_response` parts in one user
    turn. Plain turns stay `{"role", "content"}` dicts, which the backend
    converts to parts."""

    def render(
        self, messages: Sequence[ChatMessage], target: RenderTarget
    ) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for item in group_tool_results(messages):
            if isinstance(item, list):
                rendered.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": result.name,
                                    "response": {"result": result.content},
                                }
                            }
                            for result in item
                        ],
                    }
                )
            elif isinstance(item, SystemMessage | UserMessage):
                rendered.append({"role": item.role, "content": item.content})
            elif isinstance(item, AIMessage):
                rendered.append(self._render_ai(item, target))
        return rendered

    @staticmethod
    def _render_ai(message: AIMessage, target: RenderTarget) -> dict[str, Any]:
        if not message.tool_calls:
            return {"role": "assistant", "content": message.text or ""}
        signed = can_replay(message.origin, target)
        parts: list[dict[str, Any]] = []
        if message.text:
            parts.append({"text": message.text})
        for call in message.tool_calls:
            part: dict[str, Any] = {
                "function_call": {"name": call.name, "args": call.input}
            }
            if signed and call.thought_signature is not None:
                part["thought_signature"] = call.thought_signature
            parts.append(part)
        return {"role": "model", "parts": parts}

    def parse(self, result: CompletionResult, origin: ModelOrigin) -> AIMessage:
        return AIMessage(
            text=result.content if isinstance(result.content, str) else None,
            tool_calls=[
                ToolCall(
                    id=call.id,
                    name=call.name,
                    input=call.input,
                    thought_signature=call.thought_signature,
                )
                for call in result.tool_calls
            ],
            origin=origin,
        )


__all__ = ["GeminiCodec"]
