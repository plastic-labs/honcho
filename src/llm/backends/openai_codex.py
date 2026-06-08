from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
from pydantic import BaseModel

from src.exceptions import ValidationException
from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.structured_output import repair_response_model_json

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
ACCOUNT_ID_CLAIM = "https://api.openai.com/auth.chatgpt_account_id"
NESTED_AUTH_CLAIM = "https://api.openai.com/auth"


@dataclass
class OpenAICodexClient:
    """Small runtime handle for the ChatGPT Codex Responses endpoint."""

    api_key: str
    base_url: str | None = None
    refresh_token: str | None = None
    token_url: str = CODEX_OAUTH_TOKEN_URL
    client_id: str = CODEX_OAUTH_CLIENT_ID
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    http_client: httpx.AsyncClient | None = None
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        raise ValidationException("OpenAI Codex API key must be a JWT")
    try:
        payload = json.loads(_b64url_decode(parts[1]))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValidationException("OpenAI Codex API key is not a valid JWT") from exc
    if not isinstance(payload, dict):
        raise ValidationException("OpenAI Codex API key JWT payload is invalid")
    return cast(dict[str, Any], payload)


def _jwt_expiring(token: str, skew_seconds: int) -> bool:
    try:
        exp = _jwt_payload(token).get("exp")
    except ValidationException:
        return False
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= time.time() + max(0, skew_seconds)


def extract_chatgpt_account_id(token: str) -> str:
    payload = _jwt_payload(token)
    account_id = payload.get(ACCOUNT_ID_CLAIM)
    if not isinstance(account_id, str) or not account_id:
        nested_auth = payload.get(NESTED_AUTH_CLAIM)
        if isinstance(nested_auth, dict):
            nested_auth_dict = cast(dict[str, Any], nested_auth)
            account_id = nested_auth_dict.get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id:
        raise ValidationException(
            "OpenAI Codex API key JWT is missing chatgpt account id"
        )
    return account_id


def normalize_codex_url(base_url: str | None) -> str:
    base = (base_url or DEFAULT_CODEX_BASE_URL).rstrip("/")
    if base.endswith("/codex/responses"):
        return base
    if base.endswith("/codex"):
        return f"{base}/responses"
    return f"{base}/codex/responses"


def _split_tool_call_id(tool_call_id: str) -> tuple[str, str | None]:
    if "|" not in tool_call_id:
        return tool_call_id, None
    call_id, item_id = tool_call_id.split("|", 1)
    return call_id, item_id or None


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if role == "user":
            text = content if isinstance(content, str) else json.dumps(content)
            converted.append(
                {"role": role, "content": [{"type": "input_text", "text": text}]}
            )
        elif role == "assistant":
            if isinstance(content, str) and content:
                converted.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                )
            tool_calls = cast(list[dict[str, Any]], message.get("tool_calls") or [])
            for tool_call in tool_calls:
                function = cast(dict[str, Any], tool_call.get("function") or {})
                call_id, item_id = _split_tool_call_id(str(tool_call.get("id", "")))
                if not item_id:
                    item_id = f"fc_{call_id}"
                converted.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    }
                )
        elif role == "tool":
            call_id, _item_id = _split_tool_call_id(str(message.get("tool_call_id", "")))
            output = content if isinstance(content, str) else json.dumps(content)
            converted.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )
    return converted


def _instructions_from_messages(messages: list[dict[str, Any]]) -> str:
    system_parts: list[str] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            system_parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n\n".join(system_parts) or "You are a helpful assistant."


def _append_structured_output_instruction(
    messages: list[dict[str, Any]], response_format: type[BaseModel]
) -> list[dict[str, Any]]:
    """Codex currently ignores json_mode; make structured output explicit."""
    instruction = (
        "Return ONLY valid JSON matching this JSON schema. "
        "Do not include markdown, bullets, code fences, or explanatory prose.\n"
        f"Schema: {json.dumps(response_format.model_json_schema())}"
    )
    if not messages:
        return [{"role": "user", "content": instruction}]

    updated = [dict(message) for message in messages]
    last = updated[-1]
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = f"{content}\n\n{instruction}"
    else:
        updated.append({"role": "user", "content": instruction})
    return updated


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make Pydantic JSON Schema acceptable for Responses structured outputs."""
    schema = dict(schema)
    if schema.get("type") == "object":
        properties = cast(dict[str, Any], schema.get("properties") or {})
        schema["additionalProperties"] = False
        schema["required"] = list(properties.keys())
        schema["properties"] = {
            key: _strict_json_schema(cast(dict[str, Any], value))
            for key, value in properties.items()
        }
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        schema["items"] = _strict_json_schema(cast(dict[str, Any], schema["items"]))
    if "$defs" in schema and isinstance(schema["$defs"], dict):
        defs = cast(dict[str, Any], schema["$defs"])
        schema["$defs"] = {
            key: _strict_json_schema(cast(dict[str, Any], value))
            for key, value in defs.items()
        }
    return schema


def _codex_text_options(
    *,
    verbosity: str,
    response_format: type[BaseModel] | dict[str, Any] | None,
    json_mode: bool,
) -> dict[str, Any]:
    text: dict[str, Any] = {"verbosity": verbosity}
    if isinstance(response_format, type):
        text["format"] = {
            "type": "json_schema",
            "name": response_format.__name__,
            "strict": True,
            "schema": _strict_json_schema(response_format.model_json_schema()),
        }
    elif isinstance(response_format, dict):
        text["format"] = response_format
    elif json_mode:
        text["format"] = {"type": "json_object"}
    return text


def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            fn = tool["function"]
            converted.append(
                {
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        else:
            converted.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                }
            )
    return converted


class OpenAICodexBackend:
    """SSE-only backend for ChatGPT's Codex Responses endpoint."""

    def __init__(self, client: OpenAICodexClient) -> None:
        self._client: OpenAICodexClient = client
        if not client.api_key:
            raise ValidationException("Missing API key for openai_codex model config")
        self._account_id: str = extract_chatgpt_account_id(client.api_key)
        self._url: str = normalize_codex_url(client.base_url)

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
        if thinking_budget_tokens is not None:
            raise ValidationException(
                "OpenAI Codex backend does not support thinking_budget_tokens; use thinking_effort instead"
            )
        request_messages = messages
        if isinstance(response_format, type):
            request_messages = _append_structured_output_instruction(
                messages, response_format
            )
        result = await self._request(
            model=model,
            messages=request_messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
        )
        if isinstance(response_format, type):
            result.content = repair_response_model_json(
                str(result.content), response_format, model
            )
        return result

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
        if thinking_budget_tokens is not None:
            raise ValidationException(
                "OpenAI Codex backend does not support thinking_budget_tokens; use thinking_effort instead"
            )
        request_messages = messages
        if isinstance(response_format, type):
            request_messages = _append_structured_output_instruction(
                messages, response_format
            )
        result = CompletionResult()
        async for chunk in self._request_stream(
            result,
            model=model,
            messages=request_messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
            yield_text=True,
        ):
            yield chunk
        yield StreamChunk(
            is_done=True,
            finish_reason=result.finish_reason,
            output_tokens=result.output_tokens,
        )

    async def _request(self, **kwargs: Any) -> CompletionResult:
        result = CompletionResult()
        async for _chunk in self._request_stream(result, yield_text=False, **kwargs):
            pass
        return result

    async def _request_stream(
        self,
        result: CompletionResult,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: type[BaseModel] | dict[str, Any] | None,
        thinking_effort: str | None,
        extra_params: dict[str, Any] | None,
        yield_text: bool,
    ) -> AsyncIterator[StreamChunk]:
        await self._refresh_access_token_if_needed()
        body = self._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking_effort=thinking_effort,
            extra_params=extra_params,
        )
        owns_client = self._client.http_client is None
        http_client = self._client.http_client or httpx.AsyncClient(timeout=600.0)
        try:
            try:
                async with http_client.stream(
                    "POST", self._url, headers=self._headers(), json=body
                ) as response:
                    response.raise_for_status()
                    async for event in _iter_sse_events(response):
                        chunk = self._handle_event(result, event)
                        if yield_text and chunk.content:
                            yield chunk
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 401 or not self._client.refresh_token:
                    raise
                await self._refresh_access_token(force=True)
                async with http_client.stream(
                    "POST", self._url, headers=self._headers(), json=body
                ) as response:
                    response.raise_for_status()
                    async for event in _iter_sse_events(response):
                        chunk = self._handle_event(result, event)
                        if yield_text and chunk.content:
                            yield chunk
        finally:
            if owns_client:
                await http_client.aclose()

    async def _refresh_access_token_if_needed(self) -> None:
        if not self._client.refresh_token:
            return
        if _jwt_expiring(self._client.api_key, self._client.refresh_skew_seconds):
            await self._refresh_access_token(force=False)

    async def _refresh_access_token(self, *, force: bool = False) -> None:
        async with self._client.refresh_lock:
            if not force and not _jwt_expiring(
                self._client.api_key, self._client.refresh_skew_seconds
            ):
                return
            refresh_token = self._client.refresh_token
            if not refresh_token:
                raise ValidationException(
                    "OpenAI Codex refresh requested but no refresh_token is configured"
                )
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self._client.client_id,
            }
            owns_client = self._client.http_client is None
            http_client = self._client.http_client or httpx.AsyncClient(timeout=60.0)
            try:
                response = await http_client.post(
                    self._client.token_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data=payload,
                )
                response.raise_for_status()
                token_payload = response.json()
            except httpx.HTTPStatusError as exc:
                raise ValidationException(
                    f"OpenAI Codex token refresh failed with status {exc.response.status_code}"
                ) from exc
            finally:
                if owns_client:
                    await http_client.aclose()
            access_token = token_payload.get("access_token")
            if not isinstance(access_token, str) or not access_token.strip():
                raise ValidationException(
                    "OpenAI Codex token refresh response was missing access_token"
                )
            self._client.api_key = access_token.strip()
            next_refresh = token_payload.get("refresh_token")
            if isinstance(next_refresh, str) and next_refresh.strip():
                self._client.refresh_token = next_refresh.strip()
            self._account_id = extract_chatgpt_account_id(self._client.api_key)

    def _build_body(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: type[BaseModel] | dict[str, Any] | None,
        thinking_effort: str | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        verbosity = (extra_params or {}).get("verbosity") or "low"
        body: dict[str, Any] = {
            "model": model,
            "store": False,
            "stream": True,
            "instructions": _instructions_from_messages(messages),
            "input": convert_messages(messages),
            "text": _codex_text_options(
                verbosity=verbosity,
                response_format=response_format,
                json_mode=bool((extra_params or {}).get("json_mode")),
            ),
            "include": ["reasoning.encrypted_content"],
            "tool_choice": tool_choice or "auto",
            "parallel_tool_calls": True,
        }
        _ = max_tokens  # Codex currently rejects max_output_tokens.
        if temperature is not None:
            body["temperature"] = temperature
        if stop:
            body["stop"] = stop
        converted_tools = convert_tools(tools)
        if converted_tools:
            body["tools"] = converted_tools
        if thinking_effort and thinking_effort != "none":
            body["reasoning"] = {"effort": thinking_effort, "summary": "auto"}
        if extra_params:
            for key in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
                if key in extra_params:
                    body[key] = extra_params[key]
        return body

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._client.api_key}",
            "chatgpt-account-id": self._account_id,
            "originator": "honcho",
            "OpenAI-Beta": "responses=experimental",
            "accept": "text/event-stream",
            "content-type": "application/json",
        }

    def _handle_event(self, result: CompletionResult, event: dict[str, Any]) -> StreamChunk:
        event_type = event.get("type") or event.get("event")
        if event_type == "response.output_text.delta":
            delta = str(event.get("delta") or "")
            result.content = f"{result.content}{delta}"
            return StreamChunk(content=delta)
        if event_type == "response.function_call_arguments.delta":
            item_id = str(event.get("item_id") or event.get("id") or "")
            if item_id:
                accumulated = _accumulator(result).setdefault(item_id, {})
                accumulated["arguments"] = str(accumulated.get("arguments") or "") + str(
                    event.get("delta") or ""
                )
        if event_type == "response.function_call_arguments.done":
            item_id = str(event.get("item_id") or event.get("id") or "")
            if item_id and event.get("arguments") is not None:
                _accumulator(result).setdefault(item_id, {})["arguments"] = str(
                    event.get("arguments")
                )
        if event_type == "response.output_item.added":
            item = cast(dict[str, Any], event.get("item") or {})
            if item.get("type") == "function_call":
                accumulator_key = str(item.get("id") or item.get("call_id") or "")
                if accumulator_key:
                    _accumulator(result)[accumulator_key] = dict(item)
        if event_type == "response.output_item.done":
            item = cast(dict[str, Any], event.get("item") or {})
            if item.get("type") == "function_call":
                self._add_tool_call(result, item)
        if event_type in {"response.completed", "response.done", "response.incomplete"}:
            response = event.get("response") or event
            self._extract_usage(result, response.get("usage") or {})
            if event_type == "response.incomplete":
                result.finish_reason = "length"
            elif result.tool_calls:
                result.finish_reason = "tool_calls"
            else:
                result.finish_reason = "stop"
        if event_type in {"response.failed", "error"}:
            error = event.get("error") or event
            raise ValidationException(f"OpenAI Codex request failed: {error}")
        return StreamChunk()

    def _add_tool_call(self, result: CompletionResult, item: dict[str, Any]) -> None:
        call_id = str(item.get("call_id") or item.get("id") or "")
        item_id = str(item.get("id") or f"fc_{call_id}")
        accumulated = _accumulator(result).get(item_id, {})
        merged_item = {**accumulated, **item}
        args_text = merged_item.get("arguments", "{}")
        try:
            parsed_args = json.loads(args_text or "{}")
        except (TypeError, json.JSONDecodeError):
            parsed_args = {}
        args = cast(dict[str, Any], parsed_args if isinstance(parsed_args, dict) else {})
        result.tool_calls.append(
            ToolCallResult(
                id=f"{call_id}|{item_id}",
                name=str(merged_item.get("name") or ""),
                input=args,
            )
        )
        result.finish_reason = "tool_calls"

    @staticmethod
    def _extract_usage(result: CompletionResult, usage: dict[str, Any]) -> None:
        result.input_tokens = int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        )
        result.output_tokens = int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
        raw_details: Any = (
            usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        )
        if isinstance(raw_details, dict):
            details = cast(dict[str, Any], raw_details)
            result.cache_read_input_tokens = int(details.get("cached_tokens") or 0)


def _accumulator(result: CompletionResult) -> dict[str, dict[str, Any]]:
    raw_response: Any = result.raw_response
    if not isinstance(raw_response, dict):
        raw_response = {"function_calls": {}}
        result.raw_response = raw_response
    raw = cast(dict[str, Any], raw_response)
    function_calls = raw.setdefault("function_calls", {})
    return cast(dict[str, dict[str, Any]], function_calls)


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                event = _parse_sse_event(buffer)
                buffer = []
                if event is not None:
                    yield event
            continue
        buffer.append(line)
    if buffer:
        event = _parse_sse_event(buffer)
        if event is not None:
            yield event


def _parse_sse_event(lines: list[str]) -> dict[str, Any] | None:
    event_name: str | None = None
    data_parts: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].strip())
    if not data_parts:
        return None
    data = "\n".join(data_parts)
    if data == "[DONE]":
        return {"type": "response.done"}
    payload = json.loads(data)
    if event_name and "type" not in payload:
        payload["type"] = event_name
    return payload
