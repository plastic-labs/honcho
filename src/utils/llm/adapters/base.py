"""
Provider adapter interfaces for the Honcho LLM layer.

Adapters encapsulate provider-specific request shaping, response parsing, and
tool-calling message formats. The orchestrator (retry/failover/tool-loop) stays
provider-agnostic by delegating those details to an adapter.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from src.utils.llm.models import HonchoLLMCallResponse, HonchoLLMCallStreamChunk
from src.utils.types import SupportedProviders


@runtime_checkable
class ProviderAdapter(Protocol):
    """Interface implemented by per-provider adapters."""

    provider: SupportedProviders

    async def call(
        self,
        *,
        client: Any,
        provider: SupportedProviders,
        model: str,
        prompt: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None,
        json_mode: bool,
        temperature: float | None,
        stop_seqs: list[str] | None,
        reasoning_effort: str | None,
        verbosity: str | None,
        thinking_budget_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> HonchoLLMCallResponse[Any]:
        """
        Perform a non-streaming LLM call and return a normalized response.

        Implementations must preserve all provider idiosyncrasies currently relied
        upon by callers (token accounting, tool call parsing, reasoning extraction,
        structured output handling, etc.).
        """
        ...

    def stream(
        self,
        *,
        client: Any,
        provider: SupportedProviders,
        model: str,
        prompt: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None,
        json_mode: bool,
        temperature: float | None,
        reasoning_effort: str | None,
        verbosity: str | None,
        thinking_budget_tokens: int | None,
    ) -> AsyncIterator[HonchoLLMCallStreamChunk]:
        """Perform a streaming LLM call and yield normalized stream chunks."""
        ...

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert tool definitions to the provider-specific schema.

        The input is in the Anthropic-style schema used by Honcho tool definitions.
        """
        ...

    def format_assistant_tool_message(
        self,
        *,
        content: Any,
        tool_calls: list[dict[str, Any]],
        thinking_blocks: list[dict[str, Any]] | None = None,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Format an assistant message containing tool calls for this provider."""
        ...

    def append_tool_results(
        self,
        *,
        tool_results: list[dict[str, Any]],
        conversation_messages: list[dict[str, Any]],
    ) -> None:
        """Append tool results to `conversation_messages` in provider format."""
        ...


def openai_tool_calls_from_tool_calls(tool_calls: list[dict[str, Any]]) -> list[Any]:
    """
    Convert normalized Honcho tool calls to OpenAI tool_calls entries.

    This helper is shared by OpenAI-compatible adapters when formatting
    multi-turn tool calling messages.
    """
    openai_tool_calls: list[Any] = []
    for tool_call in tool_calls:
        openai_tool_calls.append(
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call["input"]),
                },
            }
        )
    return openai_tool_calls
