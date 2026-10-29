"""Provider codecs: typed history to each transport's message shape, and back.

A codec renders `ChatMessage` history into the message dicts its backend
accepts, and parses a backend's `CompletionResult` into an `AIMessage`. Codecs
are pure: the same history and target always render the same dicts, so a
history that moves between transports (a fallback mid tool loop) is rendered
afresh for each attempt instead of carrying one provider's format into another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import ModelTransport
from src.llm.backend import CompletionResult
from src.llm.messages import (
    AIMessage,
    ChatMessage,
    ModelOrigin,
    ReasoningPart,
    ToolMessage,
)


@dataclass(frozen=True, slots=True)
class RenderTarget:
    """The model a history is being rendered for."""

    transport: ModelTransport
    model: str
    base_url: str | None = None


class ProviderCodec(Protocol):
    def render(
        self, messages: Sequence[ChatMessage], target: RenderTarget
    ) -> list[dict[str, Any]]:
        """Render `messages` as the message dicts the target's backend accepts."""
        ...

    def parse(self, result: CompletionResult, origin: ModelOrigin) -> AIMessage:
        """The assistant turn `result` carries, produced by `origin`."""
        ...


def can_replay(origin: ModelOrigin | None, target: RenderTarget) -> bool:
    """Whether reasoning produced by `origin` may be sent to `target`.

    Reasoning only means something to the provider that produced it. Claude
    models accept each other's thinking blocks (the API drops what a model
    can't use), so any Claude model on the Anthropic transport qualifies. An
    OpenAI-compatible endpoint's reasoning is tied to that endpoint and model,
    and a Gemini thought signature to Gemini.
    """
    if origin is None or origin.transport != target.transport:
        return False
    if target.transport == "openai":
        return origin.model == target.model and origin.base_url == target.base_url
    return True


def replayable_reasoning(
    message: AIMessage, target: RenderTarget
) -> list[ReasoningPart]:
    return [part for part in message.reasoning if can_replay(part.origin, target)]


def group_tool_results(
    messages: Sequence[ChatMessage],
) -> list[ChatMessage | list[ToolMessage]]:
    """`messages` with each run of consecutive `ToolMessage`s gathered into a list.

    Anthropic and Gemini send all the results of one round in a single user
    turn.
    """
    grouped: list[ChatMessage | list[ToolMessage]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            last = grouped[-1] if grouped else None
            if isinstance(last, list):
                last.append(message)
            else:
                grouped.append([message])
        else:
            grouped.append(message)
    return grouped


__all__ = [
    "ProviderCodec",
    "RenderTarget",
    "can_replay",
    "group_tool_results",
    "replayable_reasoning",
]
