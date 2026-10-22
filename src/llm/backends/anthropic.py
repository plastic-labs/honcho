from __future__ import annotations

import copy
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from anthropic.types import (
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from pydantic import BaseModel, ValidationError

from src.exceptions import ValidationException
from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.capabilities import anthropic_capabilities
from src.llm.request_builder import (
    apply_sdk_passthroughs,
    request_timeout_from_extra_params,
    thinking_tool_choice_conflict_mode,
)
from src.llm.structured_output import (
    StructuredOutputError,
    repair_response_model_json,
    schema_instruction,
    validate_structured_output,
)

logger = logging.getLogger(__name__)

# output_config.effort levels. Honcho's "minimal" has no Anthropic equivalent
# and maps to the lowest one; "none" can't turn thinking off on adaptive
# models, so it sends no effort.
_ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _anthropic_effort(thinking_effort: str | None) -> str | None:
    if thinking_effort == "minimal":
        return "low"
    return thinking_effort if thinking_effort in _ANTHROPIC_EFFORTS else None


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
            # Operator escape hatch: forward Anthropic SDK passthrough kwargs
            # from ModelConfig.provider_params. Shallow merge with operator-wins.
            apply_sdk_passthroughs(params, extra_params)
        self._apply_thinking(
            params, model, thinking_budget_tokens, thinking_effort, extra_params
        )

        timeout = request_timeout_from_extra_params(extra_params)
        if timeout is not None:
            params["timeout"] = timeout

        # The '{' prefill forces a JSON-first response, which suppresses
        # tool_use blocks — skip it when tools are available and rely on the
        # conditional instruction + repair fallback instead.
        use_json_prefill = (
            bool(response_format or self._json_mode(extra_params))
            and not thinking_budget_tokens
            and not tools
            and self._supports_assistant_prefill(model)
        )
        if use_json_prefill and params["messages"]:
            if response_format and isinstance(response_format, type):
                self._append_text_to_last_message(
                    params["messages"],
                    schema_instruction(response_format, tools_present=False),
                )
            params["messages"].append({"role": "assistant", "content": "{"})
        elif (
            response_format and isinstance(response_format, type) and params["messages"]
        ):
            self._add_schema_instruction(
                params, schema_instruction(response_format, tools_present=bool(tools))
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
            # Operator escape hatch: forward Anthropic SDK passthrough kwargs
            # from ModelConfig.provider_params. Shallow merge with operator-wins.
            apply_sdk_passthroughs(params, extra_params)

        timeout = request_timeout_from_extra_params(extra_params)
        if timeout is not None:
            params["timeout"] = timeout

        # See complete(): no '{' prefill when tools are available, so
        # tool_use blocks stay reachable on the streamed path too.
        use_json_prefill = (
            bool(response_format or is_json_mode)
            and not thinking_budget_tokens
            and not tools
            and self._supports_assistant_prefill(model)
        )
        if use_json_prefill and params["messages"]:
            if response_format and isinstance(response_format, type):
                self._append_text_to_last_message(
                    params["messages"],
                    schema_instruction(response_format, tools_present=False),
                )
            params["messages"].append({"role": "assistant", "content": "{"})
        elif (
            response_format and isinstance(response_format, type) and params["messages"]
        ):
            self._add_schema_instruction(
                params, schema_instruction(response_format, tools_present=bool(tools))
            )
        self._apply_thinking(
            params, model, thinking_budget_tokens, thinking_effort, extra_params
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
            elif isinstance(block, RedactedThinkingBlock):
                # No readable text, but it must be replayed with the turn.
                thinking_full_blocks.append(
                    {"type": "redacted_thinking", "data": block.data}
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
        # Tool-call turns carry no consumable content
        if response_format is not None and not tool_calls:
            raw_content = f"{{{text_content}" if prefilled_json else text_content
            try:
                content = validate_structured_output(raw_content, response_format)
            except (
                json.JSONDecodeError,
                StructuredOutputError,
                ValidationError,
                ValueError,
            ):
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
    def _apply_thinking(
        params: dict[str, Any],
        model: str,
        thinking_budget_tokens: int | None,
        thinking_effort: str | None,
        extra_params: dict[str, Any] | None,
    ) -> None:
        """Configure thinking for `model` and reconcile it with tool_choice.

        Runs after the provider_params passthroughs, so a `thinking` supplied
        through `extra_body` counts as requested thinking too.

        A budget model gets `{"type": "enabled", "budget_tokens": N}`. An
        adaptive model gets `{"type": "adaptive"}` when thinking is requested,
        plus `output_config.effort` from `thinking_effort`; it rejects a
        budget, so the budget only switches thinking on. A forced tool_choice
        either conflicts with thinking (older models: resolved by the
        `thinking_tool_choice_conflict` mode), is accepted, or is rejected
        outright, in which case it is relaxed to `auto`.

        https://platform.claude.com/docs/en/build-with-claude/thinking#thinking-with-tool-use
        """
        caps = anthropic_capabilities(model)
        tool_choice = params.get("tool_choice")
        forced = tool_choice is not None and tool_choice.get("type") in {"any", "tool"}

        if forced and caps.forced_tool_choice == "rejected":
            logger.warning(
                "Relaxing tool_choice %s to auto: model %s rejects forced tool use",
                tool_choice,
                model,
            )
            params["tool_choice"] = {"type": "auto"}
            forced = False

        effort = (
            _anthropic_effort(thinking_effort) if caps.thinking == "adaptive" else None
        )
        extra_body: dict[str, Any] = params.get("extra_body") or {}
        operator_thinking_config = extra_body.get("thinking")
        operator_thinking = (
            isinstance(operator_thinking_config, dict)
            and cast(dict[str, Any], operator_thinking_config).get("type") != "disabled"
        )
        if not (thinking_budget_tokens or effort or operator_thinking):
            return

        if forced and caps.forced_tool_choice == "conflicts_with_thinking":
            mode = thinking_tool_choice_conflict_mode(extra_params, "throw")
            conflict = f"model {model}: tool_choice {tool_choice} forces tool use, "
            conflict += "which Anthropic does not allow alongside extended thinking"
            if mode == "throw":
                raise ValidationException(f"Cannot send request to {conflict}")
            if mode == "override_tool":
                logger.warning("Relaxing tool_choice to auto for %s", conflict)
                params["tool_choice"] = {"type": "auto"}
            else:
                logger.warning("Dropping extended thinking for %s", conflict)
                if operator_thinking:
                    remaining = {k: v for k, v in extra_body.items() if k != "thinking"}
                    if remaining:
                        params["extra_body"] = remaining
                    else:
                        params.pop("extra_body")
                return

        if operator_thinking:
            # extra_body.thinking replaces whatever Honcho would send.
            return
        if caps.thinking == "budget":
            if thinking_budget_tokens:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
            return
        params["thinking"] = {"type": "adaptive"}
        if effort:
            params["output_config"] = {"effort": effort}

    @staticmethod
    def _supports_assistant_prefill(model: str) -> bool:
        return anthropic_capabilities(model).supports_prefill

    @classmethod
    def _add_schema_instruction(cls, params: dict[str, Any], instruction: str) -> None:
        """Add the response-schema instruction to the request.

        With tools, the request is one round of a tool loop that sends the same
        history again next round, so the instruction goes into the system
        prompt, which stays the same every round. Suffixed onto the last
        message instead, it would vanish from that message on the next round
        and change the prefix that replayed thinking blocks are bound to.
        Without tools, it is appended to the last message.
        """
        if not params.get("tools"):
            cls._append_text_to_last_message(params["messages"], instruction)
            return
        system = params.get("system")
        if system:
            system[-1]["text"] = f"{system[-1]['text']}\n\n{instruction}"
        else:
            params["system"] = [
                {
                    "type": "text",
                    "text": instruction,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

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
