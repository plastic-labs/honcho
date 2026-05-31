from __future__ import annotations

import copy
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from pydantic import BaseModel, ValidationError

from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.structured_output import repair_response_model_json

# Effort levels accepted by ``output_config.effort`` (ordered low -> high).
# Honcho's ThinkingEffortLevel additionally allows "none"/"minimal", which the
# Messages API does not accept, so those are mapped onto supported values below.
_ANTHROPIC_EFFORTS: frozenset[str] = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)
_EFFORT_ALIASES: dict[str, str] = {"minimal": "low"}

# claude-<tier>-<major>-<minor>[-<date/suffix>]; only the version prefix matters
# for picking the thinking format.
_MODEL_VERSION_RE = re.compile(r"^claude-(opus|sonnet|haiku)-(\d+)-(\d+)")

# Per-tier (major, minor) at/above which Anthropic removed the legacy
# ``thinking: {"type": "enabled", "budget_tokens": N}`` format and *requires*
# adaptive thinking. Sending the legacy shape to one of these models returns
# HTTP 400:
#   "thinking.type.enabled" is not supported for this model. Use
#   "thinking.type.adaptive" and "output_config.effort" to control thinking ...
# Verified against the Anthropic docs (Opus 4.7 and Opus 4.8 are adaptive-only):
# https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
# Opus 4.6 / Sonnet 4.6 still accept the legacy format (deprecated but
# functional), so they intentionally stay on the legacy path.
_ADAPTIVE_THINKING_MIN_VERSION: dict[str, tuple[int, int]] = {"opus": (4, 7)}


def _requires_adaptive_thinking(model: str) -> bool:
    """Return True if ``model`` rejects legacy budget-based thinking (HTTP 400).

    Such models require ``thinking: {"type": "adaptive"}`` with
    ``output_config.effort`` instead of ``{"type": "enabled", "budget_tokens"}``.
    """
    match = _MODEL_VERSION_RE.match(model)
    if match is None:
        return False
    minimum = _ADAPTIVE_THINKING_MIN_VERSION.get(match.group(1))
    if minimum is None:
        return False
    return (int(match.group(2)), int(match.group(3))) >= minimum


def _budget_to_effort(thinking_budget_tokens: int) -> str:
    """Bucket a legacy thinking-token budget into an adaptive effort level.

    There is no exact mapping from ``budget_tokens`` to ``effort``; these
    buckets keep budget-configured models at a comparable thinking depth (e.g.
    the deriver's 16000-token dream budget maps to ``high``, 32000 to ``xhigh``).
    """
    if thinking_budget_tokens < 4096:
        return "low"
    if thinking_budget_tokens < 16000:
        return "medium"
    if thinking_budget_tokens < 32000:
        return "high"
    return "xhigh"


def _adaptive_effort(
    thinking_effort: str | None, thinking_budget_tokens: int | None
) -> str | None:
    """Resolve ``output_config.effort`` for an adaptive-thinking request.

    An explicit ``thinking_effort`` wins; otherwise the legacy
    ``thinking_budget_tokens`` is bucketed so existing budget-based configs keep
    a comparable thinking depth. Returns None to fall back to the API default
    (``high``).
    """
    if thinking_effort and thinking_effort != "none":
        normalized = _EFFORT_ALIASES.get(thinking_effort, thinking_effort)
        if normalized in _ANTHROPIC_EFFORTS:
            return normalized
    if thinking_budget_tokens:
        return _budget_to_effort(thinking_budget_tokens)
    return None


def _build_thinking_params(
    model: str,
    thinking_budget_tokens: int | None,
    thinking_effort: str | None,
) -> dict[str, Any]:
    """Build the ``thinking`` (+ ``output_config``) request params for a model.

    Opus 4.7+ reject the legacy ``{"type": "enabled", "budget_tokens": N}`` shape
    with HTTP 400 and require adaptive thinking; older models keep the legacy
    shape unchanged. Returns an empty dict when thinking is not requested.
    """
    if not _requires_adaptive_thinking(model):
        if thinking_budget_tokens:
            return {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
            }
        return {}

    effort = _adaptive_effort(thinking_effort, thinking_budget_tokens)
    if not thinking_budget_tokens and effort is None:
        return {}
    params: dict[str, Any] = {"thinking": {"type": "adaptive"}}
    if effort is not None:
        params["output_config"] = {"effort": effort}
    return params


class AnthropicBackend:
    """Provider backend wrapping the native Anthropic SDK."""

    def __init__(self, client: Any) -> None:
        self._client: Any = client

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
    ) -> CompletionResult:
        del max_output_tokens

        request_messages, system_messages = self._extract_system(messages)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": request_messages,
        }

        thinking_params = _build_thinking_params(
            model, thinking_budget_tokens, thinking_effort
        )
        params.update(thinking_params)

        if temperature is not None:
            params["temperature"] = temperature
        if stop:
            params["stop_sequences"] = stop
        if system_messages:
            params["system"] = [
                {
                    "type": "text",
                    "text": "\n\n".join(system_messages),
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            params["tools"] = tools
            converted_tool_choice = self._convert_tool_choice(tool_choice)
            if converted_tool_choice is not None:
                params["tool_choice"] = converted_tool_choice
        if extra_params:
            for key in ("top_p", "top_k"):
                if key in extra_params:
                    params[key] = extra_params[key]

        use_json_prefill = (
            bool(response_format or self._json_mode(extra_params))
            and "thinking" not in thinking_params
            and self._supports_assistant_prefill(model)
        )
        if use_json_prefill and params["messages"]:
            if response_format and isinstance(response_format, type):
                schema_json = json.dumps(response_format.model_json_schema(), indent=2)
                self._append_text_to_last_message(
                    params["messages"],
                    f"\n\nRespond with valid JSON matching this schema:\n{schema_json}",
                )
            params["messages"].append({"role": "assistant", "content": "{"})
        elif (
            response_format and isinstance(response_format, type) and params["messages"]
        ):
            schema_json = json.dumps(response_format.model_json_schema(), indent=2)
            self._append_text_to_last_message(
                params["messages"],
                f"\n\nRespond with valid JSON matching this schema:\n{schema_json}",
            )

        response = await self._client.messages.create(**params)
        return self._normalize_response(
            response=response,
            response_format=response_format
            if isinstance(response_format, type)
            else None,
            prefilled_json=use_json_prefill,
            model_name=model,
        )

    async def stream(
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
    ) -> AsyncIterator[StreamChunk]:
        is_json_mode = self._json_mode(extra_params)
        del max_output_tokens

        request_messages, system_messages = self._extract_system(messages)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": request_messages,
        }
        thinking_params = _build_thinking_params(
            model, thinking_budget_tokens, thinking_effort
        )
        params.update(thinking_params)
        if temperature is not None:
            params["temperature"] = temperature
        if stop:
            params["stop_sequences"] = stop
        if tools:
            params["tools"] = tools
            converted_tool_choice = self._convert_tool_choice(tool_choice)
            if converted_tool_choice is not None:
                params["tool_choice"] = converted_tool_choice
        if system_messages:
            params["system"] = [
                {
                    "type": "text",
                    "text": "\n\n".join(system_messages),
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if extra_params:
            for key in ("top_p", "top_k"):
                if key in extra_params:
                    params[key] = extra_params[key]
        use_json_prefill = (
            bool(response_format or is_json_mode)
            and "thinking" not in thinking_params
            and self._supports_assistant_prefill(model)
        )
        if use_json_prefill and params["messages"]:
            if response_format and isinstance(response_format, type):
                schema_json = json.dumps(response_format.model_json_schema(), indent=2)
                self._append_text_to_last_message(
                    params["messages"],
                    f"\n\nRespond with valid JSON matching this schema:\n{schema_json}",
                )
            params["messages"].append({"role": "assistant", "content": "{"})
        elif (
            response_format and isinstance(response_format, type) and params["messages"]
        ):
            schema_json = json.dumps(response_format.model_json_schema(), indent=2)
            self._append_text_to_last_message(
                params["messages"],
                f"\n\nRespond with valid JSON matching this schema:\n{schema_json}",
            )

        async with self._client.messages.stream(**params) as stream:
            async for chunk in stream:
                if (
                    chunk.type == "content_block_delta"
                    and hasattr(chunk, "delta")
                    and hasattr(chunk.delta, "text")
                ):
                    yield StreamChunk(content=getattr(chunk.delta, "text", ""))

            final_message = await stream.get_final_message()
            output_tokens = (
                final_message.usage.output_tokens if final_message.usage else None
            )
            yield StreamChunk(
                is_done=True,
                finish_reason=final_message.stop_reason,
                output_tokens=output_tokens,
            )

    def _normalize_response(
        self,
        *,
        response: Any,
        response_format: type[BaseModel] | None,
        prefilled_json: bool,
        model_name: str,
    ) -> CompletionResult:
        text_blocks: list[str] = []
        thinking_text_blocks: list[str] = []
        thinking_full_blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCallResult] = []

        for block in response.content:
            if isinstance(block, TextBlock):
                text_blocks.append(block.text)
            elif isinstance(block, ThinkingBlock):
                thinking_text_blocks.append(block.thinking)
                thinking_full_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    }
                )
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    ToolCallResult(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input),
                    )
                )

        usage = response.usage
        cache_creation_tokens = (
            getattr(usage, "cache_creation_input_tokens", 0) or 0 if usage else 0
        )
        cache_read_tokens = (
            getattr(usage, "cache_read_input_tokens", 0) or 0 if usage else 0
        )
        uncached_tokens = usage.input_tokens if usage else 0
        total_input_tokens = uncached_tokens + cache_creation_tokens + cache_read_tokens

        text_content = "\n".join(text_blocks)
        thinking_content = (
            "\n".join(thinking_text_blocks) if thinking_text_blocks else None
        )

        content: Any = text_content
        if response_format is not None:
            raw_content = f"{{{text_content}" if prefilled_json else text_content
            try:
                if prefilled_json:
                    parsed_json = json.loads(raw_content)
                    content = response_format.model_validate(parsed_json)
                else:
                    content = response_format.model_validate_json(raw_content)
            except (json.JSONDecodeError, ValidationError, ValueError):
                content = repair_response_model_json(
                    raw_content,
                    response_format,
                    model_name,
                )

        return CompletionResult(
            content=content,
            input_tokens=total_input_tokens,
            output_tokens=usage.output_tokens if usage else 0,
            cache_creation_input_tokens=cache_creation_tokens,
            cache_read_input_tokens=cache_read_tokens,
            finish_reason=response.stop_reason or "stop",
            tool_calls=tool_calls,
            thinking_content=thinking_content,
            thinking_blocks=thinking_full_blocks,
            raw_response=response,
        )

    @staticmethod
    def _supports_assistant_prefill(model: str) -> bool:
        # Claude 4-class models reject assistant-prefill and require the
        # conversation to end with a user message.
        return not model.startswith(
            (
                "claude-opus-4",
                "claude-sonnet-4",
                "claude-haiku-4",
            )
        )

    @staticmethod
    def _extract_system(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        system_messages: list[str] = []
        non_system_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "system" and isinstance(
                message.get("content"),
                str,
            ):
                system_messages.append(message["content"])
            else:
                non_system_messages.append(copy.deepcopy(message))
        return non_system_messages, system_messages

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, dict):
            return tool_choice
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice in {"any", "required"}:
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        return {"type": "tool", "name": tool_choice}

    @staticmethod
    def _append_text_to_last_message(
        messages: list[dict[str, Any]], suffix: str
    ) -> None:
        """Append text to the last message, handling both string and list content."""
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = content + suffix
        elif isinstance(content, list):
            # Content block list — append to the last text block or add one
            blocks: list[dict[str, Any]] = content  # pyright: ignore[reportUnknownVariableType]
            for block in reversed(blocks):
                if block.get("type") == "text":
                    block["text"] = block["text"] + suffix
                    return
            blocks.append({"type": "text", "text": suffix})

    @staticmethod
    def _json_mode(extra_params: dict[str, Any] | None) -> bool:
        return bool(extra_params and extra_params.get("json_mode"))
