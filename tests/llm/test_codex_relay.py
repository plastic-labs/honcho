from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.codex_relay import (
    AuthStore,
    CodexRelay,
    aggregate_codex_sse,
    build_responses_request,
    create_app,
)

TEXT_SSE = """event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello "}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"world"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":12,"output_tokens":4}}}

"""

TOOL_SSE = """event: response.output_item.added
data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_1","name":"lookup","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"city\\":\\"Paris\\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\\"city\\":\\"Paris\\"}","status":"completed"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_2","status":"completed","usage":{"input_tokens":20,"output_tokens":7}}}

"""


def test_build_responses_request_translates_chat_contract() -> None:
    request = build_responses_request(
        {
            "model": "gpt-5.6-luna",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup", "description": "Lookup weather", "parameters": {"type": "object"}}}],
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
            "max_tokens": 200,
        }
    )

    assert request["model"] == "gpt-5.6-luna"
    assert request["instructions"] == "Be concise."
    assert request["store"] is False
    assert request["stream"] is True
    assert request["reasoning"] == {"effort": "high", "summary": "auto"}
    assert request["text"] == {"format": {"type": "json_object"}}
    assert "max_output_tokens" not in request
    assert request["input"][-2:] == [
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    ]
    assert request["tools"] == [{"type": "function", "name": "lookup", "description": "Lookup weather", "parameters": {"type": "object"}, "strict": False}]


def test_aggregate_codex_sse_text_response() -> None:
    result = aggregate_codex_sse(TEXT_SSE)
    assert result.content == "Hello world"
    assert result.tool_calls == []
    assert result.response_id == "resp_1"
    assert result.usage == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    assert result.finish_reason == "stop"


def test_aggregate_codex_sse_tool_call_response() -> None:
    result = aggregate_codex_sse(TOOL_SSE)
    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"city":"Paris"}'}}]


@pytest.mark.asyncio
async def test_non_stream_completion_aggregates_upstream_sse() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        assert request.url.path == "/backend-api/codex/responses"
        return httpx.Response(200, text=TEXT_SSE, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token="test-token")
    response = await relay.complete({"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "Hi"}]})
    await client.aclose()

    assert response.status_code == 200
    body = json.loads(bytes(response.body).decode())
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["usage"]["total_tokens"] == 16
    assert seen["body"]["stream"] is True  # type: ignore[index]


@pytest.mark.asyncio
async def test_stream_completion_translates_codex_sse() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=TEXT_SSE, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token="test-token")
    upstream = await relay.open_upstream({"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "Hi"}], "stream": True})
    chunks = [chunk async for chunk in relay.stream_chat(upstream, "gpt-5.6-luna")]
    await client.aclose()

    streamed = b"".join(chunks)
    assert b'"content":"Hello "' in streamed
    assert b'"finish_reason":"stop"' in streamed
    assert streamed.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_upstream_http_error_status_and_body_are_preserved() -> None:
    error_body = '{"error":{"type":"invalid_request_error","message":"bad model"}}'

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=error_body, headers={"content-type": "application/json"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token="test-token")
    response = await relay.complete({"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "Hi"}]})
    await client.aclose()

    assert response.status_code == 401
    assert bytes(response.body).decode() == error_body


def test_health_smoke() -> None:
    client = TestClient(create_app(CodexRelay(access_token="test-token")))
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_auth_store_refreshes_and_persists_rotated_pool_token(tmp_path: Any) -> None:
    def encoded(value: dict[str, Any]) -> str:
        raw = json.dumps(value).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    old = f"header.{encoded({'exp': time.time() - 1})}.signature"
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps({"credential_pool": {"openai-codex": [{"access_token": old, "refresh_token": "refresh-1"}]}})
    )
    store = AuthStore(path)

    async def fake_refresh(refresh_token: str) -> dict[str, Any]:
        assert refresh_token == "refresh-1"
        return {"access_token": "new-access", "refresh_token": "refresh-2"}

    store._refresh = fake_refresh  # type: ignore[method-assign]
    assert await store.token() == "new-access"
    persisted = json.loads(path.read_text())
    assert persisted["credential_pool"]["openai-codex"][0]["access_token"] == "new-access"
    assert persisted["credential_pool"]["openai-codex"][0]["refresh_token"] == "refresh-2"
