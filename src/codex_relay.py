"""Local OpenAI Chat Completions to ChatGPT Codex Responses relay.

The relay deliberately speaks only the subset Honcho uses. It always sends a
streaming Responses request to Codex (the Codex endpoint requires this shape),
then either forwards translated SSE or aggregates it into a Chat Completions
response. It is a local compatibility process, not a production proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("codex_relay")
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_AUTH_PATH = Path.home() / ".hermes" / "auth.json"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


class RelayError(RuntimeError):
    """An error that can be represented as an OpenAI-compatible error body."""

    def __init__(self, message: str, *, status_code: int = 502, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class ProviderStreamError(RelayError):
    """The provider sent a terminal error inside an otherwise successful SSE stream."""


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


def _response_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            return None
        return {
            "type": "function",
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": function.get("parameters") or {"type": "object"},
            "strict": bool(function.get("strict", False)),
        }
    # Honcho's provider-independent tool shape is also accepted directly.
    name = tool.get("name")
    if isinstance(name, str) and name:
        return {
            "type": "function",
            "name": name,
            "description": str(tool.get("description") or ""),
            "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object"},
            "strict": bool(tool.get("strict", False)),
        }
    return None


def _response_format(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    kind = value.get("type")
    if kind == "json_object":
        return {"format": {"type": "json_object"}}
    if kind == "json_schema":
        schema = value.get("json_schema")
        if isinstance(schema, dict):
            fmt: dict[str, Any] = {
                "type": "json_schema",
                "name": str(schema.get("name") or "response"),
                "schema": schema.get("schema") or {},
                "strict": bool(schema.get("strict", True)),
            }
            return {"format": fmt}
    return None


def build_responses_request(chat_request: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions request to Codex Responses JSON."""
    messages = chat_request.get("messages")
    if not isinstance(messages, list):
        raise RelayError("messages must be a list", status_code=400)
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
    if isinstance(effort, str) and effort:
        request["reasoning"] = {"effort": effort, "summary": "auto"}
    tools = [_response_tool(tool) for tool in (chat_request.get("tools") or [])]
    request_tools = [tool for tool in tools if tool is not None]
    if request_tools:
        request["tools"] = request_tools
        choice = chat_request.get("tool_choice")
        if choice not in (None, "none"):
            request["tool_choice"] = choice if choice is not None else "auto"
        request["parallel_tool_calls"] = True
    response_format = _response_format(chat_request.get("response_format"))
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
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and name and "type" not in payload:
            payload["type"] = name
        return payload if isinstance(payload, dict) else None

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


def _usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    prompt = int(value.get("input_tokens") or value.get("prompt_tokens") or 0)
    completion = int(value.get("output_tokens") or value.get("completion_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _provider_error(payload: dict[str, Any]) -> ProviderStreamError:
    nested = payload.get("error")
    error: dict[str, Any] = nested if isinstance(nested, dict) else payload
    code = str(error.get("code") or error.get("type") or "provider_error")
    message = str(error.get("message") or "Codex Responses request failed")
    status = 429 if "rate" in code or "quota" in code else 400 if "invalid" in code else 502
    return ProviderStreamError(f"{code}: {message}", status_code=status, payload={"error": error})


def aggregate_codex_sse(raw: str | Iterable[str]) -> AggregatedResponse:
    """Aggregate Codex SSE events into the fields used by Chat Completions."""
    result = AggregatedResponse()
    arguments: dict[str, list[str]] = {}
    for event in _sse_events(raw.splitlines() if isinstance(raw, str) else raw):
        event_type = str(event.get("type") or "")
        if event_type == "error":
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
        elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
            response = event.get("response")
            if isinstance(response, dict):
                result.response_id = response.get("id")
                result.model = response.get("model")
                result.usage = _usage(response.get("usage"))
                if event_type == "response.incomplete":
                    result.finish_reason = "length"
                elif event_type == "response.failed":
                    raise _provider_error(response)
    if result.tool_calls:
        result.finish_reason = "tool_calls"
    return result


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
    payload = error.payload if isinstance(error.payload, dict) else {"error": {"message": str(error)}}
    return JSONResponse(payload, status_code=error.status_code)


class AuthStore:
    """Read Hermes auth state on demand and refresh an expiring Codex token."""

    def __init__(self, path: Path = DEFAULT_AUTH_PATH) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    def _read(self) -> dict[str, Any]:
        with self.path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise RelayError("Hermes auth store is not an object", status_code=503)
        return value

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Serialize refresh/rotation with other relay processes."""
        import fcntl

        lock_path = self.path.with_name(f".{self.path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _expiry(token: str) -> float | None:
        try:
            encoded = token.split(".")[1]
            encoded += "=" * (-len(encoded) % 4)
            claims = json.loads(base64.urlsafe_b64decode(encoded))
            return float(claims["exp"])
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _candidate(self, state: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        provider = state.get("providers", {}).get("openai-codex", {})
        tokens = provider.get("tokens") if isinstance(provider, dict) else None
        if isinstance(tokens, dict) and tokens.get("access_token"):
            return str(tokens["access_token"]), tokens
        pool = state.get("credential_pool", {}).get("openai-codex", [])
        if isinstance(pool, list):
            for entry in sorted(pool, key=lambda item: item.get("priority", 0) if isinstance(item, dict) else 0):
                if isinstance(entry, dict) and entry.get("access_token"):
                    return str(entry["access_token"]), entry
        raise RelayError("No openai-codex OAuth access token in Hermes auth.json", status_code=503)

    async def token(self, *, force_refresh: bool = False) -> str:
        async with self._lock:
            with self._file_lock():
                state = self._read()
                access, token_record = self._candidate(state)
                refresh = token_record.get("refresh_token") if token_record else None
                expiry = self._expiry(access)
                needs_refresh = force_refresh or (expiry is not None and expiry <= time.time() + 120)
                if not needs_refresh or not isinstance(refresh, str) or not refresh:
                    return access
                response = await self._refresh(refresh)
                new_access = response.get("access_token")
                if not isinstance(new_access, str) or not new_access:
                    raise RelayError("Codex token refresh returned no access_token", status_code=503)
                new_refresh = response.get("refresh_token") or refresh
                self._update(state, access, new_access, str(new_refresh))
                self._atomic_write(state)
                return new_access

    async def _refresh(self, refresh_token: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Accept": "application/json", "User-Agent": "honcho-codex-relay"},
            )
        if response.status_code != 200:
            raise RelayError(
                f"Codex token refresh failed with status {response.status_code}",
                status_code=response.status_code,
                payload={"error": {"message": response.text, "code": "codex_refresh_failed"}},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RelayError("Codex token refresh returned invalid JSON", status_code=503) from exc
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _update(state: dict[str, Any], old: str, access: str, refresh: str) -> None:
        provider = state.get("providers", {}).get("openai-codex", {})
        records: list[dict[str, Any]] = []
        if isinstance(provider, dict) and isinstance(provider.get("tokens"), dict):
            records.append(provider["tokens"])
        pool = state.get("credential_pool", {}).get("openai-codex", [])
        if isinstance(pool, list):
            records.extend(record for record in pool if isinstance(record, dict) and record.get("access_token") == old)
        for record in records:
            record["access_token"] = access
            record["refresh_token"] = refresh
            record["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _atomic_write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class CodexRelay:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        access_token: str | None = None,
        auth_path: Path = DEFAULT_AUTH_PATH,
        upstream_url: str = CODEX_RESPONSES_URL,
    ) -> None:
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0))
        self._owns_client = client is None
        self.access_token = access_token
        self.auth = AuthStore(auth_path)
        self.upstream_url = upstream_url

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _send(self, payload: dict[str, Any], *, force_refresh: bool = False) -> httpx.Response:
        token = self.access_token or await self.auth.token(force_refresh=force_refresh)
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "codex_cli_rs/0.0.0 (Honcho relay)",
            "originator": "codex_cli_rs",
        }
        request = self.client.build_request("POST", self.upstream_url, json=payload, headers=headers)
        return await self.client.send(request, stream=True)

    async def open_upstream(self, chat_request: dict[str, Any]) -> httpx.Response:
        payload = build_responses_request(chat_request)
        response = await self._send(payload)
        if response.status_code == 401 and not self.access_token:
            await response.aclose()
            response = await self._send(payload, force_refresh=True)
        return response

    async def complete(self, chat_request: dict[str, Any]) -> Response:
        response = await self.open_upstream(chat_request)
        try:
            raw = await response.aread()
            if response.status_code >= 400:
                return Response(content=raw, status_code=response.status_code, media_type=response.headers.get("content-type"))
            try:
                result = aggregate_codex_sse(raw.decode("utf-8", errors="replace"))
            except RelayError as exc:
                return _error_response(exc)
            return JSONResponse(_chat_response(result, str(chat_request.get("model") or "")))
        finally:
            await response.aclose()

    async def stream_chat(self, response: httpx.Response, model: str) -> AsyncIterator[bytes]:
        response_id = f"chatcmpl-relay-{uuid.uuid4().hex}"
        created = int(time.time())
        first_delta = True
        tool_calls: list[dict[str, Any]] = []

        def event_payload(event_name: str | None, data_lines: list[str]) -> dict[str, Any] | None:
            raw = "\n".join(data_lines).strip()
            if not raw or raw == "[DONE]":
                return None
            try:
                event: Any = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if isinstance(event, dict) and event_name and "type" not in event:
                event["type"] = event_name
            return event if isinstance(event, dict) else None

        async def translate(event: dict[str, Any]) -> AsyncIterator[bytes]:
            nonlocal first_delta
            event_type = str(event.get("type") or "")
            if event_type == "error":
                yield _sse_line({"error": event})
                yield b"data: [DONE]\n\n"
                return
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
            elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                if event_type == "response.failed":
                    yield _sse_line({"error": event.get("response") or event})
                finish = "tool_calls" if tool_calls else "length" if event_type == "response.incomplete" else "stop"
                yield _sse_line({"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
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
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            event = event_payload(event_name, data_lines)
            if event is not None:
                async for chunk in translate(event):
                    yield chunk
        finally:
            await response.aclose()


def _sse_line(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


def create_app(relay: CodexRelay | None = None) -> FastAPI:
    relay = relay or CodexRelay()
    app = FastAPI(title="Honcho Codex relay", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise RelayError("request body must be an object", status_code=400)
            if body.get("stream") is True:
                upstream = await relay.open_upstream(body)
                if upstream.status_code >= 400:
                    raw = await upstream.aread()
                    await upstream.aclose()
                    return Response(content=raw, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))
                return StreamingResponse(relay.stream_chat(upstream, str(body.get("model") or "")), media_type="text/event-stream")
            return await relay.complete(body)
        except RelayError as exc:
            return _error_response(exc)
        except (json.JSONDecodeError, ValueError) as exc:
            return _error_response(RelayError(str(exc), status_code=400))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--auth-path", type=Path, default=DEFAULT_AUTH_PATH)
    parser.add_argument("--upstream-url", default=CODEX_RESPONSES_URL)
    args = parser.parse_args()
    import uvicorn

    logger.info("Starting local Codex relay on %s:%s", args.host, args.port)
    uvicorn.run(create_app(CodexRelay(auth_path=args.auth_path, upstream_url=args.upstream_url)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
