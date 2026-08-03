from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from src.config import coerce_provider_timeout
from src.exceptions import ValidationException


@dataclass(slots=True)
class ToolCallResult:
    """Normalized tool call from any provider."""

    id: str
    name: str
    input: dict[str, Any]
    # Gemini returns this as raw bytes; other providers omit it.
    thought_signature: str | bytes | None = None


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


def request_timeout_from_extra_params(
    extra_params: dict[str, Any] | None,
) -> float | None:
    """Return a validated per-request provider timeout from extra params.

    Config-sourced timeouts are already validated and normalized at config
    load (`coerce_provider_timeout` in src.config); this guards extra_params
    passed programmatically at call time.
    """
    if not extra_params or "timeout" not in extra_params:
        return None

    try:
        return coerce_provider_timeout(extra_params["timeout"])
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc


@runtime_checkable
class ProviderBackend(Protocol):
    """Transport-agnostic interface for LLM providers.

    Credentials are baked into the underlying SDK client at backend construction
    time (see src/llm/registry.py), so these method signatures deliberately do
    not accept api_key / api_base.
    """

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
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
