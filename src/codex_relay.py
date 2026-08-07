"""Local OpenAI Chat Completions to ChatGPT Codex Responses relay.

The relay deliberately speaks only the subset Honcho uses. It always sends a
streaming Responses request to Codex (the Codex endpoint requires this shape),
then either forwards translated SSE or aggregates it into a Chat Completions
response. It is a local compatibility process, not a production proxy.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("codex_relay")
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_AUTH_PATH = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "auth.json"


class RelayError(RuntimeError):
    """An error that can be represented as an OpenAI-compatible error body."""

    def __init__(self, message: str, *, status_code: int = 502, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class CredentialError(RelayError):
    """A failure resolving or validating the local Codex credential source."""


class ProviderStreamError(RelayError):
    """The provider sent a terminal error inside an otherwise successful SSE stream."""


TokenProvider = Callable[[bool], Awaitable[str]]


@dataclass(slots=True)
class AggregatedResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    response_id: str | None = None
    model: str | None = None
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning: str = ""


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _responses_content(content: Any, *, assistant: bool = False) -> Any:
    if not isinstance(content, list):
        return _as_text(content)
    parts: list[dict[str, Any]] = []
    text_type = "output_text" if assistant else "input_text"
    for part in content:
        if isinstance(part, str):
            parts.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "").lower()
        if part_type in {"text", "input_text", "output_text"}:
            parts.append({"type": text_type, "text": _as_text(part.get("text"))})
        elif part_type in {"image_url", "input_image"}:
            image = part.get("image_url")
            detail = part.get("detail")
            if isinstance(image, dict):
                detail = image.get("detail", detail)
                image = image.get("url")
            if isinstance(image, str) and image:
                item: dict[str, Any] = {"type": "input_image", "image_url": image}
                if isinstance(detail, str) and detail:
                    item["detail"] = detail
                parts.append(item)
    return parts


def _assistant_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        converted.append(
            {
                "type": "function_call",
                "call_id": str(call.get("id") or f"call_{uuid.uuid4().hex[:16]}"),
                "name": name,
                "arguments": arguments or "{}",
            }
        )
    return converted


def _response_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise RelayError("tool must be an object", status_code=400)
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RelayError("function tool name must be a non-empty string", status_code=400)
        parameters = function.get("parameters", {"type": "object"})
        if not isinstance(parameters, dict):
            raise RelayError("function tool parameters must be an object", status_code=400)
        description = function.get("description", "")
        if description is not None and not isinstance(description, str):
            raise RelayError("function tool description must be a string", status_code=400)
        strict = function.get("strict", False)
        if not isinstance(strict, bool):
            raise RelayError("function tool strict must be a boolean", status_code=400)
        return {
            "type": "function",
            "name": name,
            "description": description or "",
            "parameters": parameters,
            "strict": strict,
        }
    # Honcho's provider-independent tool shape is also accepted directly.
    name = tool.get("name")
    if isinstance(name, str) and name:
        parameters = tool.get("input_schema") or tool.get("parameters") or {"type": "object"}
        if not isinstance(parameters, dict):
            raise RelayError("tool parameters must be an object", status_code=400)
        strict = tool.get("strict", False)
        if not isinstance(strict, bool):
            raise RelayError("tool strict must be a boolean", status_code=400)
        return {
            "type": "function",
            "name": name,
            "description": str(tool.get("description") or ""),
            "parameters": parameters,
            "strict": strict,
        }
    raise RelayError("tool must contain a named function", status_code=400)


def _response_format(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        raise RelayError("response_format must be an object", status_code=400)
    kind = value.get("type")
    if kind == "json_object":
        return {"format": {"type": "json_object"}}
    if kind == "json_schema":
        schema = value.get("json_schema")
        if not isinstance(schema, dict):
            raise RelayError("response_format.json_schema must be an object", status_code=400)
        name = schema.get("name", "response")
        shape = schema.get("schema", {})
        strict = schema.get("strict", True)
        if not isinstance(name, str) or not name:
            raise RelayError("response_format.json_schema.name must be a non-empty string", status_code=400)
        if not isinstance(shape, dict):
            raise RelayError("response_format.json_schema.schema must be an object", status_code=400)
        if not isinstance(strict, bool):
            raise RelayError("response_format.json_schema.strict must be a boolean", status_code=400)
        fmt: dict[str, Any] = {"type": "json_schema", "name": name, "schema": shape, "strict": strict}
        return {"format": fmt}
    raise RelayError("response_format has an unsupported type", status_code=400)


def _responses_tool_choice(value: Any) -> str | dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value in {"none", "auto", "required"}:
            return value
        raise RelayError("tool_choice must be none, auto, or required, or a named function", status_code=400)
    if isinstance(value, dict):
        function = value.get("function")
        if value.get("type") == "function" and isinstance(function, dict) and isinstance(function.get("name"), str):
            return {"type": "function", "name": function["name"]}
        if value.get("type") == "function" and isinstance(value.get("name"), str):
            return {"type": "function", "name": value["name"]}
    raise RelayError("tool_choice has an unsupported shape", status_code=400)


_CHAT_REQUEST_FIELDS = {
    "model", "messages", "stream", "tools", "tool_choice", "parallel_tool_calls",
    "reasoning_effort", "response_format", "max_tokens", "max_completion_tokens",
    "stream_options",
}

_UNSUPPORTED_CHAT_FIELDS = {
    "temperature", "top_p", "stop", "presence_penalty", "frequency_penalty",
    "seed", "logprobs", "top_logprobs", "n", "user", "verbosity", "reasoning",
    "extra_body", "extra_headers", "extra_query",
}


def build_responses_request(chat_request: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions request to Codex Responses JSON."""
    messages = chat_request.get("messages")
    if not isinstance(messages, list):
        raise RelayError("messages must be a list", status_code=400)
    unknown = sorted(set(chat_request) - _CHAT_REQUEST_FIELDS - _UNSUPPORTED_CHAT_FIELDS)
    if unknown:
        raise RelayError(f"unsupported Chat Completions parameter: {unknown[0]}", status_code=400)
    unsupported = sorted(key for key in _UNSUPPORTED_CHAT_FIELDS if key in chat_request)
    if unsupported:
        raise RelayError(f"unsupported Chat Completions parameter: {unsupported[0]}", status_code=400)
    if "stream" in chat_request and not isinstance(chat_request["stream"], bool):
        raise RelayError("stream must be a boolean", status_code=400)
    if "stream_options" in chat_request:
        options = chat_request["stream_options"]
        if not isinstance(options, dict) or set(options) - {"include_usage"}:
            raise RelayError("stream_options only supports include_usage", status_code=400)
        if "include_usage" in options and not isinstance(options["include_usage"], bool):
            raise RelayError("stream_options.include_usage must be a boolean", status_code=400)
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise RelayError("each message must be an object", status_code=400)
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if role in {"system", "developer"}:
            text = _as_text(content)
            if text:
                instructions.append(text)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _as_text(content),
                    }
                )
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        if role == "assistant":
            calls = _assistant_tool_calls(message)
            if calls:
                input_items.extend(calls)
            if content not in (None, "", []):
                input_items.append(
                    {"role": "assistant", "content": _responses_content(content, assistant=True)}
                )
        else:
            input_items.append({"role": role, "content": _responses_content(content)})

    request: dict[str, Any] = {
        "model": str(chat_request.get("model") or "gpt-5.6-luna"),
        "instructions": "\n\n".join(instructions),
        "input": input_items,
        "store": False,
        "stream": True,
    }
    effort = chat_request.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or not effort):
        raise RelayError("reasoning_effort must be a non-empty string", status_code=400)
    if effort:
        request["reasoning"] = {"effort": effort, "summary": "auto"}
    raw_tools = chat_request.get("tools")
    if raw_tools is not None and not isinstance(raw_tools, list):
        raise RelayError("tools must be a list", status_code=400)
    request_tools = [_response_tool(tool) for tool in (raw_tools or [])]
    choice = _responses_tool_choice(chat_request["tool_choice"]) if "tool_choice" in chat_request else None
    if request_tools:
        request["tools"] = request_tools
        if choice is not None:
            request["tool_choice"] = choice
    elif choice not in (None, "none"):
        raise RelayError("tool_choice requires tools", status_code=400)
    if "parallel_tool_calls" in chat_request:
        if not isinstance(chat_request["parallel_tool_calls"], bool):
            raise RelayError("parallel_tool_calls must be a boolean", status_code=400)
        request["parallel_tool_calls"] = chat_request["parallel_tool_calls"]
    limits = [chat_request.get("max_completion_tokens"), chat_request.get("max_tokens")]
    if limits[0] is not None and limits[1] is not None and limits[0] != limits[1]:
        raise RelayError("max_completion_tokens and max_tokens disagree", status_code=400)
    limit = limits[0] if limits[0] is not None else limits[1]
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise RelayError("output token limit must be a positive integer", status_code=400)
        request["max_output_tokens"] = limit
    response_format = _response_format(chat_request["response_format"]) if "response_format" in chat_request else None
    if response_format:
        request["text"] = response_format
    return request


def _sse_events(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> dict[str, Any] | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        raw = "\n".join(data_lines).strip()
        name = event_name
        event_name = None
        data_lines = []
        if not raw or raw == "[DONE]":
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderStreamError("Codex returned malformed stream data", status_code=502) from exc
        if isinstance(payload, dict) and name and "type" not in payload:
            payload["type"] = name
        if not isinstance(payload, dict):
            raise ProviderStreamError("Codex returned a non-object stream event", status_code=502)
        return payload

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        if line == "":
            payload = flush()
            if payload is not None:
                yield payload
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    payload = flush()
    if payload is not None:
        yield payload


def _nested(payload: Any, key: str) -> Any:
    return payload.get(key) if isinstance(payload, dict) else None


def _usage(value: Any, *, required: bool = False) -> dict[str, int]:
    if value is None:
        if required:
            raise ProviderStreamError("Codex returned malformed usage data", status_code=502)
        return {}
    if not isinstance(value, dict):
        raise ProviderStreamError("Codex returned malformed usage data", status_code=502)

    def count(*keys: str) -> int:
        raw = next((value[key] for key in keys if key in value), None)
        if raw is None and required:
            raise ProviderStreamError("Codex returned malformed usage data", status_code=502)
        if raw is None:
            raw = 0
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ProviderStreamError("Codex returned malformed usage data", status_code=502)
        return raw

    prompt = count("input_tokens", "prompt_tokens")
    completion = count("output_tokens", "completion_tokens")
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _terminal_response(
    event: dict[str, Any], *, require_usage: bool = False
) -> tuple[dict[str, Any], dict[str, int]]:
    response = event.get("response")
    if not isinstance(response, dict):
        raise ProviderStreamError("Codex returned malformed terminal data", status_code=502)
    for key in ("id", "model"):
        if key in response and response[key] is not None and not isinstance(response[key], str):
            raise ProviderStreamError("Codex returned malformed terminal data", status_code=502)
    if "incomplete_details" in response:
        details = response["incomplete_details"]
        if details is not None and not isinstance(details, dict):
            raise ProviderStreamError("Codex returned malformed terminal data", status_code=502)
        if isinstance(details, dict) and "reason" in details and not isinstance(details["reason"], str):
            raise ProviderStreamError("Codex returned malformed terminal data", status_code=502)
    return response, _usage(response.get("usage"), required=require_usage)


def _provider_error(payload: dict[str, Any]) -> ProviderStreamError:
    nested = payload.get("error")
    error: dict[str, Any] = nested if isinstance(nested, dict) else payload
    raw_code = str(error.get("code") or error.get("type") or "provider_error")
    code = raw_code if raw_code.isascii() and raw_code.replace("_", "").replace("-", "").isalnum() else "provider_error"
    status = 429 if "rate" in code or "quota" in code else 400 if "invalid" in code else 502
    return ProviderStreamError("Codex provider returned an error", status_code=status, payload={"error": {"message": "Codex provider request failed", "code": code}})


def _aggregate_event(
    result: AggregatedResponse,
    arguments: dict[str, list[str]],
    event: dict[str, Any],
    *,
    require_usage: bool = False,
) -> bool:
    event_type = str(event.get("type") or "")
    if event_type in {"error", "response.failed"}:
        raise _provider_error(event)
    if "output_text.delta" in event_type:
        result.content += str(event.get("delta") or "")
    elif "reasoning" in event_type and "delta" in event_type:
        result.reasoning += str(event.get("delta") or "")
    elif "function_call_arguments.delta" in event_type:
        item_id = str(event.get("item_id") or event.get("call_id") or "")
        arguments.setdefault(item_id, []).append(str(event.get("delta") or ""))
    elif event_type == "response.output_item.done":
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or f"call_{len(result.tool_calls)}")
            raw_args = item.get("arguments")
            if not isinstance(raw_args, str):
                raw_args = "".join(arguments.get(str(item.get("id") or call_id), []))
            result.tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": raw_args or "{}",
                    },
                }
            )
    elif event_type in {"response.completed", "response.incomplete"}:
        response, usage = _terminal_response(event, require_usage=require_usage)
        result.response_id = response.get("id")
        result.model = response.get("model")
        result.usage = usage
        if event_type == "response.incomplete":
            details = response.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            result.finish_reason = "content_filter" if reason in {"content_filter", "content_filtering"} else "length"
        return True
    return False


def _finish_aggregate(result: AggregatedResponse) -> AggregatedResponse:
    if result.tool_calls:
        result.finish_reason = "tool_calls"
    return result


def aggregate_codex_sse(raw: str | Iterable[str], *, require_usage: bool = False) -> AggregatedResponse:
    """Aggregate Codex SSE events into the fields used by Chat Completions."""
    result = AggregatedResponse()
    arguments: dict[str, list[str]] = {}
    terminal = False
    for event in _sse_events(raw.splitlines() if isinstance(raw, str) else raw):
        if _aggregate_event(result, arguments, event, require_usage=require_usage):
            terminal = True
            break
    if not terminal:
        raise ProviderStreamError("Codex stream ended without a terminal response", status_code=502)
    return _finish_aggregate(result)


def _chat_response(result: AggregatedResponse, requested_model: str) -> dict[str, Any]:
    response_id = result.response_id or f"chatcmpl-relay-{uuid.uuid4().hex}"
    message: dict[str, Any] = {"role": "assistant", "content": result.content or None}
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model or requested_model,
        "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason}],
        "usage": result.usage,
    }


def _error_response(error: RelayError) -> JSONResponse:
    if isinstance(error, CredentialError):
        payload = {"error": {"message": "Codex credential source unavailable", "code": "credential_unavailable"}}
        status_code = 503
    elif isinstance(error.payload, dict):
        payload = error.payload
        status_code = error.status_code
    elif error.status_code == 400:
        # Input-validation details are safe and useful to the caller. Provider,
        # credential, and filesystem failures may contain secrets or local paths.
        payload = {"error": {"message": str(error), "code": "invalid_request"}}
        status_code = error.status_code
    else:
        payload = {"error": {"message": "Codex provider request failed", "code": "provider_error"}}
        status_code = error.status_code
    return JSONResponse(payload, status_code=status_code)


class AuthStore:
    """Read-only adapter for a Hermes auth document.

    Hermes owns refresh rotation, locking, profile selection, and persistence. The
    relay never writes this file; callers that need rotation inject a token provider.
    """

    def __init__(self, path: Path = DEFAULT_AUTH_PATH) -> None:
        self.path = path

    def _read(self) -> dict[str, Any]:
        try:
            with self.path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CredentialError("Hermes credential source is unavailable", status_code=503) from exc
        if not isinstance(value, dict):
            raise CredentialError("Hermes credential source is invalid", status_code=503)
        return value

    def _candidate(self, state: dict[str, Any]) -> str:
        providers = state.get("providers")
        provider = providers.get("openai-codex") if isinstance(providers, dict) else None
        tokens = provider.get("tokens") if isinstance(provider, dict) else None
        token = tokens.get("access_token") if isinstance(tokens, dict) else None
        if isinstance(token, str) and token.strip():
            return token
        raise CredentialError("No Codex access token is available", status_code=503)

    async def token(self, *, force_refresh: bool = False) -> str:
        if force_refresh:
            raise CredentialError("credential refresh is owned by Hermes, not the relay", status_code=503)
        return self._candidate(self._read())


def _account_id_from_token(token: str) -> str | None:
    """Return only the canonical account id claim; malformed/opaque tokens are safe."""
    try:
        encoded = token.split(".")[1].encode("ascii")
        encoded += b"=" * (-len(encoded) % 4)
        claims = json.loads(base64.b64decode(encoded, altchars=b"-_", validate=True))
        auth = claims.get("https://api.openai.com/auth") if isinstance(claims, dict) else None
        account_id = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
        return account_id if isinstance(account_id, str) and account_id else None
    except (
        IndexError,
        TypeError,
        ValueError,
        binascii.Error,
        json.JSONDecodeError,
        UnicodeDecodeError,
        UnicodeEncodeError,
    ):
        return None


class CodexRelay:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        access_token: str | None = None,
        token_provider: TokenProvider | None = None,
        auth_path: Path = DEFAULT_AUTH_PATH,
        upstream_url: str = CODEX_RESPONSES_URL,
    ) -> None:
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0))
        self._owns_client = client is None
        self.access_token = access_token
        self.token_provider = token_provider
        self.auth = AuthStore(auth_path)
        self.upstream_url = upstream_url

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _send(self, payload: dict[str, Any], *, force_refresh: bool = False) -> httpx.Response:
        try:
            if self.access_token is not None:
                token = self.access_token
            elif self.token_provider is not None:
                try:
                    token = await self.token_provider(force_refresh)
                except Exception as exc:
                    raise CredentialError("Codex credential source is unavailable", status_code=503) from exc
            else:
                token = await self.auth.token(force_refresh=force_refresh)
        except CredentialError:
            raise
        except RelayError as exc:
            raise CredentialError("Codex credential source is unavailable", status_code=503) from exc
        if not isinstance(token, str) or not token.strip():
            raise CredentialError("Codex credential source is unavailable", status_code=503)
        try:
            token.encode("ascii")
        except UnicodeEncodeError as exc:
            raise CredentialError("Codex credential source is unavailable", status_code=503) from exc
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "codex_cli_rs/0.0.0 (Honcho relay)",
            "originator": "codex_cli_rs",
        }
        account_id = _account_id_from_token(token)
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        request = self.client.build_request("POST", self.upstream_url, json=payload, headers=headers)
        return await self.client.send(request, stream=True)

    async def open_upstream(self, chat_request: dict[str, Any]) -> httpx.Response:
        payload = build_responses_request(chat_request)
        response = await self._send(payload)
        if response.status_code == 401 and self.token_provider is not None:
            await response.aclose()
            response = await self._send(payload, force_refresh=True)
        return response

    async def complete(self, chat_request: dict[str, Any]) -> Response:
        response = await self.open_upstream(chat_request)
        include_usage = (
            isinstance(chat_request.get("stream_options"), dict)
            and chat_request["stream_options"].get("include_usage") is True
        )
        result = AggregatedResponse()
        arguments: dict[str, list[str]] = {}
        event_name: str | None = None
        data_lines: list[str] = []
        terminal = False

        def event_payload() -> dict[str, Any] | None:
            raw = "\n".join(data_lines).strip()
            if not raw or raw == "[DONE]":
                return None
            try:
                event: Any = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ProviderStreamError("Codex returned malformed stream data", status_code=502) from exc
            if isinstance(event, dict) and event_name and "type" not in event:
                event["type"] = event_name
            if not isinstance(event, dict):
                raise ProviderStreamError("Codex returned a non-object stream event", status_code=502)
            return event

        try:
            if response.status_code >= 400:
                status = response.status_code if response.status_code < 600 else 502
                return _error_response(RelayError("Codex provider request failed", status_code=status))
            async for line in response.aiter_lines():
                if line == "":
                    event = event_payload()
                    event_name = None
                    data_lines = []
                    if event is not None and _aggregate_event(
                        result, arguments, event, require_usage=include_usage
                    ):
                        terminal = True
                        break
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not terminal:
                event = event_payload()
                if event is not None:
                    terminal = _aggregate_event(
                        result, arguments, event, require_usage=include_usage
                    )
            if not terminal:
                raise ProviderStreamError("Codex stream ended without a terminal response", status_code=502)
            return JSONResponse(_chat_response(_finish_aggregate(result), str(chat_request.get("model") or "")))
        except RelayError as exc:
            return _error_response(exc)
        finally:
            await response.aclose()

    async def stream_chat(
        self, response: httpx.Response, model: str, *, include_usage: bool = False
    ) -> AsyncIterator[bytes]:
        response_id = f"chatcmpl-relay-{uuid.uuid4().hex}"
        created = int(time.time())
        first_delta = True
        tool_calls: list[dict[str, Any]] = []
        terminal = False

        def event_payload(event_name: str | None, data_lines: list[str]) -> dict[str, Any] | None:
            raw = "\n".join(data_lines).strip()
            if not raw or raw == "[DONE]":
                return None
            try:
                event: Any = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ProviderStreamError("Codex returned malformed stream data", status_code=502) from exc
            if isinstance(event, dict) and event_name and "type" not in event:
                event["type"] = event_name
            if not isinstance(event, dict):
                raise ProviderStreamError("Codex returned a non-object stream event", status_code=502)
            return event

        async def translate(event: dict[str, Any]) -> AsyncIterator[bytes]:
            nonlocal first_delta, terminal
            if terminal:
                return
            event_type = str(event.get("type") or "")
            if event_type in {"error", "response.failed"}:
                raise _provider_error(event)
            if "output_text.delta" in event_type:
                delta = str(event.get("delta") or "")
                if delta:
                    yield _sse_line({"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant" if first_delta else None, "content": delta}, "finish_reason": None}]})
                    first_delta = False
            elif event_type == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    call = {"index": len(tool_calls), "id": item.get("call_id") or item.get("id"), "type": "function", "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")} }
                    tool_calls.append(call)
                    yield _sse_line({"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]})
            elif event_type in {"response.completed", "response.incomplete"}:
                terminal_response, usage = _terminal_response(event, require_usage=include_usage)
                terminal = True
                details = terminal_response.get("incomplete_details")
                reason = details.get("reason") if isinstance(details, dict) else None
                if event_type == "response.incomplete" and reason in {"content_filter", "content_filtering"}:
                    finish = "content_filter"
                else:
                    finish = "tool_calls" if tool_calls else "length" if event_type == "response.incomplete" else "stop"
                yield _sse_line({"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                if include_usage:
                    yield _sse_line({"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage})
                yield b"data: [DONE]\n\n"

        event_name: str | None = None
        data_lines: list[str] = []
        try:
            async for line in response.aiter_lines():
                if line == "":
                    event = event_payload(event_name, data_lines)
                    event_name = None
                    data_lines = []
                    if event is not None:
                        async for chunk in translate(event):
                            yield chunk
                        if terminal:
                            break
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not terminal:
                event = event_payload(event_name, data_lines)
                if event is not None:
                    async for chunk in translate(event):
                        yield chunk
            if not terminal:
                raise ProviderStreamError("Codex stream ended without a terminal response", status_code=502)
        except (RelayError, httpx.HTTPError):
            if not terminal:
                yield _sse_line({"error": {"message": "Codex provider stream failed", "code": "provider_stream_error"}})
                yield b"data: [DONE]\n\n"
        finally:
            await response.aclose()


def _sse_line(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def create_app(relay: CodexRelay | None = None, *, inbound_key: str | None = None, bind_host: str = "127.0.0.1") -> FastAPI:
    relay = relay or CodexRelay()
    inbound_key = inbound_key or None
    if not _is_loopback(bind_host) and not inbound_key:
        raise ValueError("inbound_key is required for non-loopback binds")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await relay.close()

    app = FastAPI(title="Honcho Codex relay", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        if inbound_key is not None:
            supplied = request.headers.get("authorization", "")
            candidate = supplied[7:] if supplied.lower().startswith("bearer ") else ""
            candidate_bytes = candidate.encode("utf-8")
            key_bytes = inbound_key.encode("utf-8")
            if not candidate_bytes or not hmac.compare_digest(candidate_bytes, key_bytes):
                return JSONResponse({"error": {"message": "Invalid relay authentication", "code": "relay_unauthorized"}}, status_code=401)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise RelayError("request body must be an object", status_code=400)
            if body.get("stream") is True:
                upstream = await relay.open_upstream(body)
                if upstream.status_code >= 400:
                    try:
                        status = upstream.status_code if upstream.status_code < 600 else 502
                        return _error_response(RelayError("Codex provider request failed", status_code=status))
                    finally:
                        await upstream.aclose()
                options = body.get("stream_options") or {}
                include_usage = bool(options.get("include_usage")) if isinstance(options, dict) else False
                return StreamingResponse(
                    relay.stream_chat(upstream, str(body.get("model") or ""), include_usage=include_usage),
                    media_type="text/event-stream",
                )
            return await relay.complete(body)
        except RelayError as exc:
            return _error_response(exc)
        except (httpx.HTTPError, OSError):
            return _error_response(RelayError("Codex provider request failed", status_code=502))
        except UnicodeError:
            return _error_response(RelayError("Codex credential source is unavailable", status_code=503))
        except (json.JSONDecodeError, ValueError):
            return _error_response(RelayError("Invalid JSON request body", status_code=400))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--auth-path", type=Path, default=DEFAULT_AUTH_PATH)
    parser.add_argument("--upstream-url", default=CODEX_RESPONSES_URL)
    parser.add_argument("--inbound-key", default=os.environ.get("CODEX_RELAY_INBOUND_KEY"), help="shared inbound API key (also CODEX_RELAY_INBOUND_KEY)")
    args = parser.parse_args()
    import uvicorn

    if not _is_loopback(args.host) and not args.inbound_key:
        parser.error("--inbound-key or CODEX_RELAY_INBOUND_KEY is required for non-loopback binds")
    logger.info("Starting local Codex relay on %s:%s", args.host, args.port)
    uvicorn.run(create_app(CodexRelay(auth_path=args.auth_path, upstream_url=args.upstream_url), inbound_key=args.inbound_key, bind_host=args.host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
