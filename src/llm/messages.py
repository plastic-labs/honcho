"""Typed conversation history for LLM calls.

Every call path inside `src/llm` carries history as a list of `ChatMessage`.
Provider wire formats exist only at the backend boundary, where a codec renders
the history for the transport an attempt targets (see `src/llm/codecs`). A
history can therefore move between providers mid-loop, e.g. when a fallback on
another transport answers one round of a tool loop.

Messages are immutable. Provider-specific reasoning (Anthropic thinking blocks,
OpenAI-compatible reasoning details, Gemini thought signatures) is kept opaque,
tagged with the model that produced it, so a codec can decide whether a target
model can be sent it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import ModelTransport
from src.exceptions import ValidationException


class _Frozen(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class ModelOrigin(_Frozen):
    """The model that produced an assistant turn or a piece of reasoning."""

    transport: ModelTransport
    model: str
    base_url: str | None = None


ReasoningKind = Literal[
    "anthropic_thinking",
    "anthropic_redacted_thinking",
    "openai_reasoning_details",
    "openai_reasoning_content",
]


class ReasoningPart(_Frozen):
    """One unit of provider reasoning, replayed byte-for-byte or not at all.

    `payload` is the provider's own block: an Anthropic `thinking` /
    `redacted_thinking` block, one OpenAI-compatible `reasoning_details`
    entry, or `{"text": ...}` for a `reasoning_content` string.
    """

    kind: ReasoningKind
    payload: dict[str, Any]
    origin: ModelOrigin


class ToolCall(_Frozen):
    """A tool call requested by the model."""

    id: str
    name: str
    input: dict[str, Any]
    # Gemini signs function calls; the signature is only valid sent back to
    # Gemini. Raw bytes from the SDK, or a string.
    thought_signature: str | bytes | None = None


class SystemMessage(_Frozen):
    role: Literal["system"] = "system"
    content: str


class UserMessage(_Frozen):
    role: Literal["user"] = "user"
    content: str


class AIMessage(_Frozen):
    """An assistant turn.

    `text` is None when the provider returned no text at all, which some
    OpenAI-compatible providers require to stay null on replay.
    """

    role: Literal["assistant"] = "assistant"
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: list[ReasoningPart] = Field(default_factory=list)
    origin: ModelOrigin | None = None


class ToolMessage(_Frozen):
    """The result of one tool call."""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


ChatMessage = Annotated[
    SystemMessage | UserMessage | AIMessage | ToolMessage,
    Field(discriminator="role"),
]

_CALLER_ROLES = ("system", "user", "assistant")


def coerce_message(message: ChatMessage | Mapping[str, Any]) -> ChatMessage:
    """Convert one caller-supplied message into a `ChatMessage`.

    Callers pass `{"role": ..., "content": str}` dicts for system, user and
    assistant turns. Anything else (a provider-native block list, a tool
    result) has no single meaning across providers, so it is rejected rather
    than guessed at.
    """
    if isinstance(message, SystemMessage | UserMessage | AIMessage | ToolMessage):
        return message
    role = message.get("role")
    content = message.get("content")
    extra = set(message) - {"role", "content"}
    if role not in _CALLER_ROLES or not isinstance(content, str) or extra:
        raise ValidationException(
            "LLM messages must be {'role': 'system' | 'user' | 'assistant', "
            + "'content': str}; got role "
            + repr(role)
            + f" with {type(content).__name__} content"
            + (f" and extra keys {sorted(extra)}" if extra else "")
        )
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return UserMessage(content=content)
    return AIMessage(text=content)


def coerce_messages(
    messages: Sequence[ChatMessage | Mapping[str, Any]],
) -> list[ChatMessage]:
    """Convert caller-supplied messages into typed history; see `coerce_message`."""
    return [coerce_message(message) for message in messages]


__all__ = [
    "AIMessage",
    "ChatMessage",
    "ModelOrigin",
    "ReasoningKind",
    "ReasoningPart",
    "SystemMessage",
    "ToolCall",
    "ToolMessage",
    "UserMessage",
    "coerce_message",
    "coerce_messages",
]
