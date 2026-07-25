from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.codex_relay import (
    AuthStore,
    CodexRelay,
    RelayError,
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
    assert request["max_output_tokens"] == 200
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


def test_auth_store_is_read_only_and_reports_missing_source(tmp_path: Any) -> None:
    path = tmp_path / "auth.json"
    store = AuthStore(path)
    with pytest.raises(RelayError, match="credential source"):
        import asyncio
        asyncio.run(store.token())


def test_auth_store_reads_only_canonical_provider_token(tmp_path: Any) -> None:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"providers": {"openai-codex": {"tokens": {"access_token": "synthetic"}}}}))
    store = AuthStore(path)
    import asyncio
    assert asyncio.run(store.token()) == "synthetic"
    assert list(tmp_path.iterdir()) == [path]


def test_tool_controls_are_translated_or_rejected() -> None:
    request = build_responses_request({
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "parallel_tool_calls": False,
        "max_completion_tokens": 20,
    })
    assert request["tool_choice"] == {"type": "function", "name": "lookup"}
    assert request["parallel_tool_calls"] is False
    assert request["max_output_tokens"] == 20
    with pytest.raises(RelayError, match="unsupported"):
        build_responses_request({"messages": [], "temperature": 0.2})


def test_aggregate_requires_terminal_and_ignores_post_terminal_data() -> None:
    raw = """event: response.output_text.delta
data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}

event: response.completed
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\"}}

event: response.output_text.delta
data: {\"type\":\"response.output_text.delta\",\"delta\":\"bad\"}

"""
    result = aggregate_codex_sse(raw)
    assert result.content == "ok"
    with pytest.raises(RelayError, match="without a terminal"):
        aggregate_codex_sse('event: response.output_text.delta\ndata: {"delta":"partial"}\n\n')


def test_aggregate_provider_and_malformed_events_are_sanitized() -> None:
    with pytest.raises(RelayError, match="provider") as provider:
        aggregate_codex_sse('event: error\ndata: {"message":"SECRET_TOKEN"}\n\n')
    assert "SECRET_TOKEN" not in str(provider.value)
    with pytest.raises(RelayError, match="malformed"):
        aggregate_codex_sse("event: response.completed\ndata: {not-json}\n\n")


def test_non_loopback_requires_key_and_rejects_before_upstream() -> None:
    with pytest.raises(ValueError, match="inbound_key"):
        create_app(CodexRelay(access_token="synthetic"), bind_host="0.0.0.0")


def test_relay_auth_rejects_missing_wrong_and_malformed_before_upstream() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=TEXT_SSE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(CodexRelay(client=client, access_token="synthetic"), inbound_key="relay-secret", bind_host="192.0.2.10")
    with TestClient(app) as test_client:
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic relay-secret"}):
            response = test_client.post("/v1/chat/completions", headers=headers, json={"messages": []})
            assert response.status_code == 401
        response = test_client.post("/v1/chat/completions", headers={"Authorization": "Bearer relay-secret"}, json={"messages": []})
        assert response.status_code == 200
    assert calls == 1


@pytest.mark.asyncio
async def test_account_claim_header_and_opaque_token_are_safe() -> None:
    import base64

    def enc(value: Any) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    token = f"h.{enc({'https://api.openai.com/auth': {'chatgpt_account_id': 'acct-synthetic'}})}.s"
    seen: dict[str, str | None] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["account"] = request.headers.get("ChatGPT-Account-ID")
        return httpx.Response(200, text=TEXT_SSE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token=token)
    response = await relay.open_upstream({"messages": [{"role": "user", "content": "x"}]})
    await response.aclose()
    assert seen["account"] == "acct-synthetic"
    relay.access_token = "opaque-token"
    response = await relay.open_upstream({"messages": [{"role": "user", "content": "x"}]})
    await response.aclose()
    assert seen["account"] is None
    await client.aclose()


@pytest.mark.asyncio
async def test_injected_token_provider_retries_once_and_sanitizes_failure() -> None:
    calls = 0
    refreshes: list[bool] = []

    async def provider(force_refresh: bool) -> str:
        refreshes.append(force_refresh)
        return "synthetic-refreshed" if force_refresh else "synthetic-stale"

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401 if calls == 1 else 200, text=TEXT_SSE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, token_provider=provider)
    response = await relay.open_upstream({"messages": [{"role": "user", "content": "x"}]})
    await response.aclose()
    assert calls == 2
    assert refreshes == [False, True]

    async def broken(_: bool) -> str:
        raise RelayError("provider secret", status_code=503)

    broken_app = create_app(CodexRelay(client=client, token_provider=broken), bind_host="127.0.0.1")
    with TestClient(broken_app) as test_client:
        result = test_client.post("/v1/chat/completions", json={"messages": []})
    assert result.status_code == 503
    assert "provider secret" not in result.text
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_failed_and_eof_emit_only_sanitized_error_done() -> None:
    failed = "event: response.output_text.delta\ndata: {\"delta\":\"partial\"}\n\nevent: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"SECRET\"}}}\n\n"
    eof = "event: response.output_text.delta\ndata: {\"delta\":\"partial\"}\n\n"

    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=failed if calls == 1 else eof)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token="synthetic")
    for request_body in ({"messages": [{"role": "user", "content": "failed"}]}, {"messages": [{"role": "user", "content": "eof"}]}):
        response = await relay.open_upstream(request_body)
        chunks = b"".join([chunk async for chunk in relay.stream_chat(response, "gpt-5.6-luna")])
        assert chunks.endswith(b"data: [DONE]\n\n")
        assert chunks.count(b"provider_stream_error") == 1
        assert b'"finish_reason":"stop"' not in chunks
        assert b"SECRET" not in chunks
    await client.aclose()


@pytest.mark.asyncio
async def test_current_openai_sdk_and_backend_in_process_paths() -> None:
    pytest.importorskip("openai")
    from openai import AsyncOpenAI

    try:
        from src.llm.backends.openai import OpenAIBackend
    except ImportError as exc:
        pytest.skip(f"upstream backend dependencies unavailable: {exc}")

    seen: list[dict[str, Any]] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, text=TEXT_SSE, headers={"content-type": "text/event-stream"})

    relay_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    relay = CodexRelay(client=relay_client, access_token="synthetic")
    sdk_transport = httpx.ASGITransport(app=create_app(relay))
    sdk_http = httpx.AsyncClient(transport=sdk_transport, base_url="http://relay")
    sdk = AsyncOpenAI(api_key="synthetic-relay-key", base_url="http://relay/v1", http_client=sdk_http)

    normal = await sdk.chat.completions.create(model="gpt-5.6-luna", messages=[{"role": "user", "content": "hi"}])
    structured = await sdk.chat.completions.create(model="gpt-5.6-luna", messages=[{"role": "user", "content": "json"}], response_format={"type": "json_object"})
    tool = await sdk.chat.completions.create(model="gpt-5.6-luna", messages=[{"role": "user", "content": "lookup"}], tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}], tool_choice="auto", parallel_tool_calls=False)
    stream = await sdk.chat.completions.create(model="gpt-5.6-luna", messages=[{"role": "user", "content": "stream"}], stream=True)
    streamed = [chunk async for chunk in stream]
    backend = OpenAIBackend(sdk)
    backend_result = await backend.complete(model="gpt-5.6-luna", messages=[{"role": "user", "content": "backend"}], max_tokens=20)

    assert normal.choices[0].message.content == "Hello world"
    assert structured.choices[0].message.content == "Hello world"
    assert tool.choices[0].finish_reason == "stop"
    assert streamed[-1].choices[0].finish_reason == "stop"
    assert backend_result.content == "Hello world"
    assert seen[2]["parallel_tool_calls"] is False
    assert seen[1]["text"] == {"format": {"type": "json_object"}}
    await sdk_http.aclose()
    await relay_client.aclose()
