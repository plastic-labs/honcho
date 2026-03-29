from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass(slots=True)
class ToolCallResult:
    """Normalized tool call from any provider."""

    id: str
    name: str
    input: dict[str, Any]
    thought_signature: str | None = None


@dataclass(slots=True)
class CompletionResult:
    """Normalized completion result returned by provider backends."""

    content: Any = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    thinking_content: str | None = None
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None


@dataclass(slots=True)
class StreamChunk:
    """A single chunk in a streaming response."""

    content: str = ""
    is_done: bool = False
    finish_reason: str | None = None
    output_tokens: int | None = None


@runtime_checkable
class ProviderBackend(Protocol):
    """Transport-agnostic interface for LLM providers."""

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
    ) -> CompletionResult: ...

    def stream(
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
    ) -> AsyncIterator[StreamChunk]: ...
