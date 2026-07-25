from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.codex_relay import (
    AuthStore,
    CodexRelay,
    CredentialError,
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
async def test_upstream_http_error_is_sanitized() -> None:
    error_body = '{"error":{"type":"invalid_request_error","message":"bad model"}}'

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=error_body, headers={"content-type": "application/json"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token="test-token")
    response = await relay.complete({"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "Hi"}]})
    await client.aclose()

    assert response.status_code == 401
    assert json.loads(bytes(response.body).decode()) == {"error": {"message": "Codex provider request failed", "code": "provider_error"}}


@pytest.mark.asyncio
async def test_provider_http_503_keeps_provider_classification_for_non_stream() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="PROVIDER_OUTAGE_DETAIL")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token="test-token")
    response = await relay.complete({"messages": []})
    await client.aclose()

    assert response.status_code == 503
    assert json.loads(bytes(response.body).decode()) == {
        "error": {"message": "Codex provider request failed", "code": "provider_error"}
    }
    assert "PROVIDER_OUTAGE_DETAIL" not in bytes(response.body).decode()


def test_provider_http_503_keeps_provider_classification_for_stream() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="PROVIDER_OUTAGE_DETAIL")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(CodexRelay(client=client, access_token="test-token"))
    with TestClient(app) as test_client:
        response = test_client.post("/v1/chat/completions", json={"messages": [], "stream": True})
    assert response.status_code == 503
    assert response.json() == {
        "error": {"message": "Codex provider request failed", "code": "provider_error"}
    }
    assert "PROVIDER_OUTAGE_DETAIL" not in response.text


def test_stream_upstream_http_error_is_sanitized() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="PROVIDER_PRIVATE_DETAIL", headers={"content-type": "text/plain"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(CodexRelay(client=client, access_token="test-token"))
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5.6-luna", "messages": [], "stream": True},
        )
    assert response.status_code == 500
    assert response.json() == {"error": {"message": "Codex provider request failed", "code": "provider_error"}}
    assert "PROVIDER_PRIVATE_DETAIL" not in response.text


def test_health_smoke() -> None:
    client = TestClient(create_app(CodexRelay(access_token="test-token")))
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_store_is_read_only_and_reports_missing_source(tmp_path: Any) -> None:
    path = tmp_path / "auth.json"
    store = AuthStore(path)
    with pytest.raises(RelayError, match="credential source"):
        asyncio.run(store.token())


def test_auth_store_reads_only_canonical_provider_token(tmp_path: Any) -> None:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"providers": {"openai-codex": {"tokens": {"access_token": "synthetic"}}}}))
    store = AuthStore(path)
    assert asyncio.run(store.token()) == "synthetic"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "document",
    [
        {"providers": []},
        {"providers": {"openai-codex": []}},
        {"providers": {"openai-codex": {"tokens": {"access_token": ""}}}},
        {"providers": {"openai-codex": {"tokens": {"access_token": 123}}}},
    ],
)
def test_malformed_auth_documents_are_sanitized_at_route(tmp_path: Any, document: dict[str, Any]) -> None:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(document))
    app = create_app(CodexRelay(auth_path=path))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 503
    assert response.json() == {"error": {"message": "Codex credential source unavailable", "code": "credential_unavailable"}}


def test_corrupt_auth_document_and_non_ascii_token_are_sanitized(tmp_path: Any) -> None:
    path = tmp_path / "auth.json"
    path.write_text("{not-json")
    app = create_app(CodexRelay(auth_path=path))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 503
    assert "not-json" not in response.text

    non_ascii = create_app(CodexRelay(access_token="x.☃.y"))
    with TestClient(non_ascii) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 503
    assert "☃" not in response.text

    async def unauthorized(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"error":{"message":"private auth detail"}}')

    client = httpx.AsyncClient(transport=httpx.MockTransport(unauthorized))
    malformed = create_app(CodexRelay(client=client, access_token="x.%%%.y"))
    with TestClient(malformed) as test_client:
        response = test_client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 401
    assert "private auth detail" not in response.text
    assert "%%" not in response.text
    asyncio.run(client.aclose())


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


@pytest.mark.parametrize("field", ["verbosity", "reasoning", "extra_body", "unknown_control"])
def test_request_allowlist_rejects_untranslated_top_level_controls(field: str) -> None:
    with pytest.raises(RelayError, match="unsupported Chat Completions parameter"):
        build_responses_request({"messages": [], field: {"value": 1}})


def test_stream_options_include_usage_is_supported_and_validated() -> None:
    request = build_responses_request({"messages": [], "stream_options": {"include_usage": True}})
    assert "stream_options" not in request
    with pytest.raises(RelayError, match="stream_options"):
        build_responses_request({"messages": [], "stream_options": {"include_usage": "yes"}})
    with pytest.raises(RelayError, match="stream_options"):
        build_responses_request({"messages": [], "stream_options": {"other": True}})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tools", ["not-an-object"], "tool"),
        ("tool_choice", {"type": "function"}, "tool_choice"),
        ("parallel_tool_calls", "yes", "parallel_tool_calls"),
        ("response_format", {"type": "xml"}, "response_format"),
    ],
)
def test_malformed_supported_fields_are_rejected(field: str, value: Any, message: str) -> None:
    with pytest.raises(RelayError, match=message):
        build_responses_request({"messages": [], field: value})


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


def test_aggregate_does_not_request_another_input_after_terminal() -> None:
    def fail_if_read() -> Any:
        yield "event: response.completed"
        yield 'data: {"response":{"id":"r","usage":{"input_tokens":1,"output_tokens":2}}}'
        yield ""
        raise AssertionError("aggregate requested input after terminal event")

    result = aggregate_codex_sse(fail_if_read())
    assert result.content == ""
    assert result.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def test_aggregate_provider_and_malformed_events_are_sanitized() -> None:
    with pytest.raises(RelayError, match="provider") as provider:
        aggregate_codex_sse('event: error\ndata: {"message":"SECRET_TOKEN"}\n\n')
    assert "SECRET_TOKEN" not in str(provider.value)
    with pytest.raises(RelayError, match="malformed"):
        aggregate_codex_sse("event: response.completed\ndata: {not-json}\n\n")
    with pytest.raises(RelayError, match="object"):
        aggregate_codex_sse("event: response.completed\ndata: [1,2,3]\n\n")


def test_aggregate_rejects_malformed_usage_and_terminal_shapes() -> None:
    with pytest.raises(RelayError, match="usage"):
        aggregate_codex_sse(
            'event: response.completed\ndata: {"response":{"usage":{"input_tokens":"bad"}}}\n\n'
        )
    with pytest.raises(RelayError, match="terminal"):
        aggregate_codex_sse('event: response.completed\ndata: {"response":[]}\n\n')


@pytest.mark.asyncio
async def test_stream_stops_without_reading_after_terminal_and_rejects_non_object() -> None:
    class BlockingAfterTerminal(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.read_after_terminal = False
            self.closed = False
            self._release = asyncio.Event()

        async def __aiter__(self):
            yield b"event: response.completed\n"
            yield b'data: {"response":{"id":"r","usage":{"input_tokens":1,"output_tokens":1}}}\n'
            yield b"\n"
            self.read_after_terminal = True
            await self._release.wait()

        async def aclose(self) -> None:
            self.closed = True
            self._release.set()

    stream = BlockingAfterTerminal()
    response = httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    relay = CodexRelay(access_token="synthetic")
    chunks = await asyncio.wait_for(
        _collect(relay.stream_chat(response, "gpt-5.6-luna")), timeout=1
    )
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert stream.read_after_terminal is False
    assert stream.closed is True

    async def non_object(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: [1,2,3]\n\n", headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(non_object))
    relay = CodexRelay(client=client, access_token="synthetic")
    upstream = await relay.open_upstream({"messages": []})
    streamed = b"".join([chunk async for chunk in relay.stream_chat(upstream, "gpt-5.6-luna")])
    assert b"provider_stream_error" in streamed
    await client.aclose()


@pytest.mark.asyncio
async def test_non_stream_completion_stops_without_reading_after_terminal() -> None:
    class BlockingAfterTerminal(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.read_after_terminal = False
            self.closed = False
            self._release = asyncio.Event()

        async def __aiter__(self):
            yield b"event: response.output_text.delta\n"
            yield b'data: {"delta":"ok"}\n'
            yield b"\n"
            yield b"event: response.completed\n"
            yield b'data: {"response":{"id":"r","usage":{"input_tokens":1,"output_tokens":2}}}\n'
            yield b"\n"
            self.read_after_terminal = True
            await self._release.wait()

        async def aclose(self) -> None:
            self.closed = True
            self._release.set()

    stream = BlockingAfterTerminal()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token="synthetic")
    response = await asyncio.wait_for(
        relay.complete({"messages": [{"role": "user", "content": "hi"}]}), timeout=1
    )
    assert response.status_code == 200
    body = json.loads(bytes(response.body).decode())
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["usage"]["total_tokens"] == 3
    assert stream.read_after_terminal is False
    assert stream.closed is True
    await client.aclose()


@pytest.mark.asyncio
async def test_include_usage_rejects_null_terminal_usage() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='event: response.completed\ndata: {"response":{"id":"r","usage":null}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    relay = CodexRelay(client=client, access_token="synthetic")
    upstream = await relay.open_upstream({"messages": []})
    streamed = b"".join([chunk async for chunk in relay.stream_chat(upstream, "gpt-5.6-luna", include_usage=True)])
    assert streamed.count(b"provider_stream_error") == 1
    assert b'"finish_reason"' not in streamed
    assert streamed.endswith(b"data: [DONE]\n\n")
    await client.aclose()


async def _collect(iterator: Any) -> list[bytes]:
    return [chunk async for chunk in iterator]


def test_non_loopback_requires_key_and_rejects_before_upstream() -> None:
    with pytest.raises(ValueError, match="inbound_key"):
        create_app(CodexRelay(access_token="synthetic"), bind_host="0.0.0.0")


def test_empty_inbound_key_is_unconfigured_on_loopback_but_required_off_host() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=TEXT_SSE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(CodexRelay(client=client, access_token="synthetic"), inbound_key="")
    with TestClient(app) as test_client:
        response = test_client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello world"

    with pytest.raises(ValueError, match="inbound_key"):
        create_app(CodexRelay(access_token="synthetic"), inbound_key="", bind_host="0.0.0.0")


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


def test_non_ascii_bearer_candidate_returns_normal_unauthorized_response() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=TEXT_SSE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        CodexRelay(client=client, access_token="synthetic"),
        inbound_key="relay-secret",
        bind_host="192.0.2.10",
    )
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            headers={b"Authorization": b"Bearer \xc3\xa9"},
            json={"messages": []},
        )
    assert response.status_code == 401
    assert response.json() == {
        "error": {"message": "Invalid relay authentication", "code": "relay_unauthorized"}
    }
    assert calls == 0


@pytest.mark.asyncio
async def test_account_claim_header_and_opaque_token_are_safe() -> None:
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


@pytest.mark.parametrize(
    "provider_error",
    [
        OSError("OS_PRIVATE_DETAIL"),
        ValueError("VALUE_PRIVATE_DETAIL"),
        RuntimeError("RUNTIME_PRIVATE_DETAIL"),
        httpx.ConnectError("HTTP_PRIVATE_DETAIL"),
        UnicodeError("UNICODE_PRIVATE_DETAIL"),
        Exception("GENERIC_PRIVATE_DETAIL"),
    ],
    ids=["oserror", "value", "runtime", "http", "unicode", "generic"],
)
def test_injected_token_provider_exception_types_are_sanitized_at_route(provider_error: Exception) -> None:
    calls: list[bool] = []

    async def provider(force_refresh: bool) -> str:
        calls.append(force_refresh)
        raise provider_error

    app = create_app(CodexRelay(token_provider=provider))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 503
    assert response.json() == {
        "error": {"message": "Codex credential source unavailable", "code": "credential_unavailable"}
    }
    assert str(provider_error) not in response.text
    assert calls == [False]


def test_credential_error_payload_is_never_reflected_at_route() -> None:
    private_payload = {"error": {"message": "PRIVATE_CREDENTIAL_PAYLOAD", "code": "private_code"}}

    async def provider(_: bool) -> str:
        raise CredentialError("PRIVATE_CREDENTIAL_EXCEPTION", status_code=418, payload=private_payload)

    app = create_app(CodexRelay(token_provider=provider))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 503
    assert response.json() == {
        "error": {"message": "Codex credential source unavailable", "code": "credential_unavailable"}
    }
    assert "PRIVATE_CREDENTIAL" not in response.text
    assert "private_code" not in response.text


@pytest.mark.asyncio
async def test_initial_401_is_closed_when_forced_refresh_provider_fails() -> None:
    class ClosingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed: bool = False

        async def __aiter__(self):
            yield b""

        async def aclose(self) -> None:
            self.closed = True

    initial_stream = ClosingStream()
    refreshes: list[bool] = []

    async def provider(force_refresh: bool) -> str:
        refreshes.append(force_refresh)
        if force_refresh:
            raise RuntimeError("REFRESH_PRIVATE_DETAIL")
        return "synthetic-stale"

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=initial_stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(CodexRelay(client=client, token_provider=provider))

    route_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://relay")
    response = await route_client.post("/v1/chat/completions", json={"messages": []})
    await route_client.aclose()
    await client.aclose()

    assert response.status_code == 503
    assert response.json() == {
        "error": {"message": "Codex credential source unavailable", "code": "credential_unavailable"}
    }
    assert "REFRESH_PRIVATE_DETAIL" not in response.text
    assert refreshes == [False, True]
    assert initial_stream.closed is True


@pytest.mark.asyncio
async def test_injected_token_provider_does_not_catch_cancellation() -> None:
    async def provider(_: bool) -> str:
        raise asyncio.CancelledError

    relay = CodexRelay(token_provider=provider)
    with pytest.raises(asyncio.CancelledError):
        await relay.open_upstream({"messages": []})


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
    assert result.json() == {
        "error": {"message": "Codex credential source unavailable", "code": "credential_unavailable"}
    }
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
    responses: list[httpx.Response] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        stream_text = (
            'event: response.completed\ndata: {"response":{"id":"r","usage":null}}\n\n'
            if "null usage" in json.dumps(body)
            else TEXT_SSE
        )
        response = httpx.Response(200, text=stream_text, headers={"content-type": "text/event-stream"})
        responses.append(response)
        return response

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
    backend_stream = backend.stream(model="gpt-5.6-luna", messages=[{"role": "user", "content": "backend stream"}], max_tokens=20)
    backend_chunks = [chunk async for chunk in backend_stream]

    null_stream = await sdk.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "null usage"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    sdk_null_error: Exception | None = None
    try:
        sdk_null_chunks = [chunk async for chunk in null_stream]
    except Exception as exc:
        sdk_null_chunks = []
        sdk_null_error = exc
    assert sdk_null_error is not None
    assert "Codex provider stream failed" in str(sdk_null_error)
    assert not any(
        chunk.choices and chunk.choices[0].finish_reason for chunk in sdk_null_chunks
    )

    backend_null_stream = backend.stream(
        model="gpt-5.6-luna", messages=[{"role": "user", "content": "null usage"}], max_tokens=20
    )
    backend_null_error: Exception | None = None
    try:
        backend_null_chunks = [chunk async for chunk in backend_null_stream]
    except Exception as exc:
        backend_null_chunks = []
        backend_null_error = exc
    assert backend_null_error is not None
    assert "Codex provider stream failed" in str(backend_null_error)
    assert not any(chunk.is_done and chunk.output_tokens is None for chunk in backend_null_chunks)

    from openai import BadRequestError
    with pytest.raises(BadRequestError, match="verbosity"):
        unsupported_stream = backend.stream(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "verbosity"}],
            max_tokens=20,
            extra_params={"verbosity": "high"},
        )
        [chunk async for chunk in unsupported_stream]
    with pytest.raises(BadRequestError, match="reasoning"):
        unsupported_reasoning = backend.stream(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "reasoning"}],
            max_tokens=20,
            extra_params={"extra_body": {"reasoning": {"max_tokens": 256}}},
        )
        [chunk async for chunk in unsupported_reasoning]
    backend_result = await backend.complete(model="gpt-5.6-luna", messages=[{"role": "user", "content": "backend"}], max_tokens=20)

    assert normal.choices[0].message.content == "Hello world"
    assert structured.choices[0].message.content == "Hello world"
    assert tool.choices[0].finish_reason == "stop"
    assert streamed[-1].choices[0].finish_reason == "stop"
    assert "".join(chunk.content for chunk in backend_chunks if chunk.content) == "Hello world"
    assert backend_chunks[-1].is_done is True
    assert backend_chunks[-1].finish_reason == "stop"
    assert backend_chunks[-1].output_tokens == 4
    assert backend_result.content == "Hello world"
    assert seen[2]["parallel_tool_calls"] is False
    assert seen[1]["text"] == {"format": {"type": "json_object"}}
    backend_stream_request = next(body for body in seen if "backend stream" in json.dumps(body))
    assert backend_stream_request["stream"] is True
    assert "stream_options" not in backend_stream_request
    assert all(response.is_closed for response in responses)
    await sdk_http.aclose()
    await relay_client.aclose()
