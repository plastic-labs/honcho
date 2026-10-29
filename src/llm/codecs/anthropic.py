"""Anthropic Messages API codec."""

from __future__ import annotations

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
    UserMessage,
)

from . import RenderTarget, group_tool_results, replayable_reasoning


class AnthropicCodec:
    """Thinking blocks, text and `tool_use` blocks in one assistant turn; each
    round's results as `tool_result` blocks in one user turn. System messages
    stay `{"role": "system"}` dicts, which the backend lifts into `system`."""

    def render(
        self, messages: Sequence[ChatMessage], target: RenderTarget
    ) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for item in group_tool_results(messages):
            if isinstance(item, list):
                rendered.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result.tool_call_id,
                                "content": result.content,
                                "is_error": result.is_error,
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
        reasoning = replayable_reasoning(message, target)
        if not message.tool_calls and not reasoning:
            return {"role": "assistant", "content": message.text or ""}
        blocks: list[dict[str, Any]] = [part.payload for part in reasoning]
        if message.text:
            blocks.append({"type": "text", "text": message.text})
        blocks.extend(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            for call in message.tool_calls
        )
        return {"role": "assistant", "content": blocks}

    def parse(self, result: CompletionResult, origin: ModelOrigin) -> AIMessage:
        return AIMessage(
            text=result.content if isinstance(result.content, str) else None,
            tool_calls=[
                ToolCall(id=call.id, name=call.name, input=call.input)
                for call in result.tool_calls
            ],
            reasoning=[
                ReasoningPart(
                    kind="anthropic_thinking"
                    if block.get("type") == "thinking"
                    else "anthropic_redacted_thinking",
                    payload=block,
                    origin=origin,
                )
                for block in result.thinking_blocks
            ],
            origin=origin,
        )


__all__ = ["AnthropicCodec"]
