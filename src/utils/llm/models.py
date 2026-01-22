"""
Shared models for Honcho LLM client orchestration.

This module contains the response/container types that are shared across providers
and orchestration layers (retry/failover, tool loop, streaming).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


@dataclass
class IterationData:
    """Data passed to iteration callbacks after each tool execution loop iteration."""

    iteration: int
    """1-indexed iteration number."""

    tool_calls: list[str]
    """List of tool names called in this iteration."""

    input_tokens: int
    """Input tokens used in this iteration's LLM call."""

    output_tokens: int
    """Output tokens generated in this iteration's LLM call."""

    cache_read_tokens: int = 0
    """Tokens read from cache in this iteration."""

    cache_creation_tokens: int = 0
    """Tokens written to cache in this iteration."""


# Callback invoked after each tool-loop iteration.
IterationCallback = Callable[[IterationData], None]


class HonchoLLMCallResponse(BaseModel, Generic[T]):
    """
    Response object for LLM calls.

    Args:
        content: The response content. When a response_model is provided, this will be
                the parsed object of that type. Otherwise, it will be a string.
        input_tokens: Total number of input tokens (including cached).
        output_tokens: Number of tokens generated in the response.
        cache_creation_input_tokens: Number of tokens written to cache.
        cache_read_input_tokens: Number of tokens read from cache.
        finish_reasons: List of finish reasons for the response.
        tool_calls_made: Optional list of all tool calls executed during the request.
        messages: Full conversation history including tool calls and results (for two-phase dialectic).

    Note:
        Uncached input tokens = input_tokens - cache_read_input_tokens + cache_creation_input_tokens
        (cache_creation costs 25% more, cache_read costs 90% less)
    """

    content: T
    input_tokens: int = 0
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    finish_reasons: list[str]
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    """Number of LLM calls made in the tool execution loop (1 = single response, 2+ = tool iterations plus final synthesis)."""

    thinking_content: str | None = None
    """Normalized reasoning/thinking content for telemetry/debugging."""

    thinking_blocks: list[dict[str, Any]] = Field(default_factory=list)
    """Full thinking blocks with signatures for multi-turn conversation replay (Anthropic only)."""

    reasoning_details: list[dict[str, Any]] = Field(default_factory=list)
    """OpenRouter reasoning_details for Gemini models - must be preserved across turns."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    """Full conversation history for two-phase dialectic (search -> synthesis)."""


class HonchoLLMCallStreamChunk(BaseModel):
    """
    A single chunk in a streaming LLM response.

    Args:
        content: The text content for this chunk. Empty for chunks that only contain metadata.
        is_done: Whether this is the final chunk in the stream.
        finish_reasons: List of finish reasons if the stream is complete.
        output_tokens: Number of tokens generated in the response. Only set on the final chunk.
    """

    content: str
    is_done: bool = False
    finish_reasons: list[str] = Field(default_factory=list)
    output_tokens: int | None = None


class StreamingResponseWithMetadata:
    """
    Wrapper for streaming responses that includes metadata from the tool execution phase.

    This allows callers to access tool call counts, token usage, and thinking content
    from the tool loop while still streaming the final response.
    """

    _stream: AsyncIterator[HonchoLLMCallStreamChunk]
    tool_calls_made: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    thinking_content: str | None
    iterations: int
    messages: list[dict[str, Any]]

    def __init__(
        self,
        stream: AsyncIterator[HonchoLLMCallStreamChunk],
        tool_calls_made: list[dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
        thinking_content: str | None = None,
        iterations: int = 0,
        messages: list[dict[str, Any]] | None = None,
    ):
        self._stream = stream
        self.tool_calls_made = tool_calls_made
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.thinking_content = thinking_content
        self.iterations = iterations
        self.messages = messages or []

    def __aiter__(self) -> AsyncIterator[HonchoLLMCallStreamChunk]:
        return self._stream.__aiter__()

    async def __anext__(self) -> HonchoLLMCallStreamChunk:
        return await self._stream.__anext__()
