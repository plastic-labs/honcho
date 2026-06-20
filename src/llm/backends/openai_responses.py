"""OpenAI Responses API backend for ChatGPT Plus / Codex OAuth.

Calls https://chatgpt.com/backend-api/codex/responses using an OAuth bearer
token from an OAuthTokenManager.  The endpoint differs from the standard
OpenAI Chat Completions API in three ways:
  - Always streams (stream=True is required)
  - Stateless (store=False required)
  - Uses the Responses API wire format (instructions + input instead of messages)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel

from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message / tool format translation helpers
# ---------------------------------------------------------------------------


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _split_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert Chat Completions messages to Responses API (instructions, input).

    - system → instructions (concatenated)
    - assistant with tool_calls → function_call input items
    - tool results → function_call_output input items
    - user/assistant text → kept as role+content items
    """
    instruction_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            text = _content_to_str(content)
            if text:
                instruction_parts.append(text)

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                if content:
                    text = _content_to_str(content)
                    if text:
                        input_items.append({"role": "assistant", "content": text})
                for tc in tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    )
            else:
                input_items.append(
                    {"role": "assistant", "content": _content_to_str(content)}
                )

        elif role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": _content_to_str(content),
                }
            )

        else:
            input_items.append(
                {"role": role, "content": _content_to_str(content)}
            )

    return "\n\n".join(instruction_parts), input_items


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Honcho / Chat Completions tool defs to Responses API format.

    Responses API puts name/description/parameters at the top level of the
    tool object, unlike Chat Completions which nests them under "function".
    """
    result: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", {})
            result.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        else:
            # Anthropic-style (name/description/input_schema at top level)
            result.append(
                {
                    "type": "function",
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                }
            )
    return result


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class OpenAIResponsesBackend:
    """Provider backend for the ChatGPT Plus / Codex Responses API.

    Requires an OAuthTokenManager for bearer-token refresh.  Makes direct
    httpx calls so it controls the exact wire format; the standard OpenAI SDK
    is not used here because /chat/completions is Cloudflare-protected on the
    chatgpt.com host while /responses is not.
    """

    def __init__(self, token_manager: Any, base_url: str) -> None:
        self._token_manager = token_manager
        self._url = base_url.rstrip("/") + "/responses"

    # ------------------------------------------------------------------
    # ProviderBackend interface
    # ------------------------------------------------------------------

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
        body = self._build_body(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_effort=thinking_effort,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCallResult] = []
        input_tokens = 0
        output_tokens = 0
        finish_reason = "stop"
        pending_fcs: dict[int, dict[str, Any]] = {}

        async for event in self._iter_events(body):
            etype = event.get("type", "")

            if etype == "response.output_text.delta":
                text_parts.append(event.get("delta", ""))

            elif etype == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    idx = event.get("output_index", 0)
                    pending_fcs[idx] = {
                        "id": item.get("call_id") or item.get("id", ""),
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    }

            elif etype == "response.function_call_arguments.delta":
                idx = event.get("output_index", 0)
                if idx in pending_fcs:
                    pending_fcs[idx]["arguments"] += event.get("delta", "")

            elif etype == "response.output_item.done":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    idx = event.get("output_index", 0)
                    fc = pending_fcs.pop(idx, None)
                    if fc:
                        args_str = fc["arguments"] or item.get("arguments", "")
                        try:
                            args: dict[str, Any] = (
                                json.loads(args_str) if args_str else {}
                            )
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append(
                            ToolCallResult(id=fc["id"], name=fc["name"], input=args)
                        )
                        finish_reason = "tool_calls"

            elif etype == "response.completed":
                usage = (event.get("response") or {}).get("usage") or {}
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

            elif etype == "response.failed":
                error = (event.get("response") or {}).get("error") or {}
                raise RuntimeError(
                    f"Codex Responses API error: {error.get('message', event)}"
                )

        content: Any = "".join(text_parts)
        if response_format is not None and isinstance(response_format, type):
            try:
                content = response_format.model_validate_json(content)
            except Exception:
                pass  # caller's repair logic handles raw string

        return CompletionResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
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
        body = self._build_body(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_effort=thinking_effort,
        )

        output_tokens = 0

        async for event in self._iter_events(body):
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                yield StreamChunk(content=event.get("delta", ""))
            elif etype == "response.completed":
                usage = (event.get("response") or {}).get("usage") or {}
                output_tokens = usage.get("output_tokens", 0)
            elif etype == "response.failed":
                error = (event.get("response") or {}).get("error") or {}
                raise RuntimeError(
                    f"Codex Responses API error: {error.get('message', event)}"
                )

        yield StreamChunk(is_done=True, finish_reason="stop", output_tokens=output_tokens)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_body(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: type[BaseModel] | dict[str, Any] | None,
        thinking_effort: str | None,
    ) -> dict[str, Any]:
        instructions, input_items = _split_messages(messages)
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
            "stream": True,
            "store": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = _convert_tools(tools)
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if thinking_effort:
            body["reasoning"] = {"effort": thinking_effort}
        if response_format is not None:
            if isinstance(response_format, type):
                body["text"] = {
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_format.__name__,
                            "schema": response_format.model_json_schema(),
                            "strict": True,
                        },
                    }
                }
            elif isinstance(response_format, dict):
                fmt_type = response_format.get("type")
                if fmt_type in ("json_object", "json_schema"):
                    body["text"] = {"format": response_format}
        return body

    async def _iter_events(
        self, body: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        await self._token_manager.refresh_if_needed()
        headers = {
            "Authorization": f"Bearer {self._token_manager.access_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream(
                "POST", self._url, json=body, headers=headers
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
                    raise RuntimeError(
                        f"Codex Responses API {resp.status_code}: {resp.text[:500]}"
                    )
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Responses API: invalid JSON in SSE: %.100s", data_str
                        )
