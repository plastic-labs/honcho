import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import settings
from src.utils.queue_payload import WebhookPayload
from src.webhooks import webhook_delivery


class FakeAsyncClient:
    def __init__(self, responses: dict[str, httpx.Response | Exception]):
        self._responses: dict[str, httpx.Response | Exception] = responses
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(
        self, exc_type: object, exc: object, tb: object
    ) -> None:  # pragma: no cover - required async CM signature
        _ = (exc_type, exc, tb)

    async def post(
        self, *, url: str, content: str, headers: dict[str, str]
    ) -> httpx.Response:
        self.calls.append({"url": url, "content": content, "headers": headers})
        result = self._responses[url]
        if isinstance(result, Exception):
            raise result
        return result


def test_generate_webhook_signature_uses_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.WEBHOOK, "SECRET", "unit-test-secret")
    payload = '{"key":"value"}'

    expected = hmac.new(
        b"unit-test-secret", payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    actual = webhook_delivery._generate_webhook_signature(payload)  # pyright: ignore[reportPrivateUsage]

    assert actual == expected


def test_generate_webhook_signature_raises_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.WEBHOOK, "SECRET", "")

    with pytest.raises(ValueError, match="WEBHOOK_SECRET not found"):
        webhook_delivery._generate_webhook_signature("{}")  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_webhook_urls_returns_all_endpoint_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = object()
    list_endpoints = AsyncMock(return_value=query)
    monkeypatch.setattr(webhook_delivery, "list_webhook_endpoints", list_endpoints)

    endpoint_a = SimpleNamespace(url="https://a.example.com/hook")
    endpoint_b = SimpleNamespace(url="https://b.example.com/hook")
    execute_result = MagicMock()
    execute_result.scalars.return_value.all.return_value = [endpoint_a, endpoint_b]

    db = AsyncMock()
    db.execute.return_value = execute_result

    urls = await webhook_delivery._get_webhook_urls(  # pyright: ignore[reportPrivateUsage]
        db, "workspace-a"
    )

    assert urls == [endpoint_a.url, endpoint_b.url]
    list_endpoints.assert_awaited_once_with("workspace-a")
    db.execute.assert_awaited_once_with(query)


@pytest.mark.asyncio
async def test_get_webhook_urls_returns_empty_list_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        webhook_delivery,
        "list_webhook_endpoints",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    db = AsyncMock()

    urls = await webhook_delivery._get_webhook_urls(  # pyright: ignore[reportPrivateUsage]
        db, "workspace-a"
    )

    assert urls == []
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_deliver_webhook_skips_when_no_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeAsyncClient({})

    def async_client_factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        _ = (args, kwargs)
        return fake_client

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)
    monkeypatch.setattr(
        webhook_delivery,
        "_get_webhook_urls",
        AsyncMock(return_value=[]),
    )

    payload = WebhookPayload(event_type="peer.created", data={"id": "p_123"})
    await webhook_delivery.deliver_webhook(AsyncMock(), payload, "workspace-a")

    assert fake_client.calls == []


@pytest.mark.asyncio
async def test_deliver_webhook_posts_signed_payload_to_each_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.WEBHOOK, "SECRET", "delivery-secret")
    monkeypatch.setattr(webhook_delivery, "utc_now_iso", lambda: "2026-02-13T00:00:00Z")

    urls = [
        "https://a.example.com/hook",
        "https://b.example.com/hook",
    ]
    monkeypatch.setattr(
        webhook_delivery,
        "_get_webhook_urls",
        AsyncMock(return_value=urls),
    )

    fake_client = FakeAsyncClient(
        {
            urls[0]: httpx.Response(
                status_code=202, request=httpx.Request("POST", urls[0])
            ),
            urls[1]: httpx.ConnectError("connection failed"),
        }
    )

    def async_client_factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        _ = (args, kwargs)
        return fake_client

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    payload = WebhookPayload(
        event_type="message.created",
        data={"id": "m_1", "workspace": "workspace-a"},
    )
    await webhook_delivery.deliver_webhook(AsyncMock(), payload, "workspace-a")

    expected_event_json = json.dumps(
        {
            "type": payload.event_type,
            "data": payload.data,
            "timestamp": "2026-02-13T00:00:00Z",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    expected_signature = webhook_delivery._generate_webhook_signature(  # pyright: ignore[reportPrivateUsage]
        expected_event_json
    )

    assert len(fake_client.calls) == 2
    for call in fake_client.calls:
        assert call["content"] == expected_event_json
        assert call["headers"]["Content-Type"] == "application/json"
        assert call["headers"]["X-Honcho-Signature"] == expected_signature


@pytest.mark.asyncio
async def test_deliver_webhook_handles_signature_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.WEBHOOK, "SECRET", "")
    monkeypatch.setattr(
        webhook_delivery,
        "_get_webhook_urls",
        AsyncMock(return_value=["https://a.example.com/hook"]),
    )
    fake_client = FakeAsyncClient(
        {
            "https://a.example.com/hook": httpx.Response(
                status_code=200,
                request=httpx.Request("POST", "https://a.example.com/hook"),
            )
        }
    )

    def async_client_factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        _ = (args, kwargs)
        return fake_client

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    payload = WebhookPayload(event_type="workspace.updated", data={"id": "ws_1"})
    await webhook_delivery.deliver_webhook(AsyncMock(), payload, "workspace-a")

    assert fake_client.calls == []


@pytest.mark.asyncio
async def test_deliver_webhook_catches_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.WEBHOOK, "SECRET", "delivery-secret")
    monkeypatch.setattr(
        webhook_delivery,
        "_get_webhook_urls",
        AsyncMock(side_effect=httpx.RequestError("network issue")),
    )

    def async_client_factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        _ = (args, kwargs)
        return FakeAsyncClient({})

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    payload = WebhookPayload(event_type="workspace.updated", data={"id": "ws_1"})
    await webhook_delivery.deliver_webhook(AsyncMock(), payload, "workspace-a")
