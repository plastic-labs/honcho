import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db import tenant_context
from src.deriver.consumer import process_item
from src.exceptions import WebhookTenantUnresolved
from src.models import QueueItem
from src.utils.queue_payload import WebhookPayload
from src.utils.work_unit import tenant_id_for_work_unit_key
from src.webhooks import webhook_delivery
from src.webhooks.events import QueueEmptyEvent, publish_webhook_event


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
    await webhook_delivery.deliver_webhook(payload, "workspace-a")

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
    await webhook_delivery.deliver_webhook(payload, "workspace-a")

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
async def test_deliver_webhook_flag_off_body_is_byte_identical_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSS single-tenant invariant: flag-off body and signature are
    byte-identical to the single-tenant wire format regardless of
    `tenant_id`."""
    # region ai
    # Pins the literal JSON rather than a round-trip comparison (like the
    # dynamic-comparison test above) so a future change to key order,
    # separators, or shape is caught here even if it happened to also
    # correctly update the reconstruction.
    # endregion
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    monkeypatch.setattr(settings.WEBHOOK, "SECRET", "delivery-secret")
    monkeypatch.setattr(webhook_delivery, "utc_now_iso", lambda: "2026-02-13T00:00:00Z")

    url = "https://a.example.com/hook"
    monkeypatch.setattr(
        webhook_delivery, "_get_webhook_urls", AsyncMock(return_value=[url])
    )

    fake_client = FakeAsyncClient(
        {url: httpx.Response(status_code=200, request=httpx.Request("POST", url))}
    )

    def async_client_factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        _ = (args, kwargs)
        return fake_client

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    payload = WebhookPayload(
        event_type="message.created",
        data={"id": "m_1", "workspace": "workspace-a"},
    )
    # tenant_id is supplied but must be ignored entirely flag-off.
    await webhook_delivery.deliver_webhook(payload, "workspace-a", tenant_id="t1")

    expected_body = (
        '{"data":{"id":"m_1","workspace":"workspace-a"},'
        '"timestamp":"2026-02-13T00:00:00Z","type":"message.created"}'
    )
    expected_signature = hmac.new(
        b"delivery-secret", expected_body.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["content"] == expected_body
    assert call["headers"]["X-Honcho-Signature"] == expected_signature


@pytest.mark.asyncio
async def test_deliver_webhook_flag_on_adds_tenant_id_to_signed_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-on with a tenant present: tenant_id joins type/data/timestamp as a
    top-level key, data is untouched, and the signature verifies over the
    exact bytes sent (the existing HMAC needs no change since tenant_id is
    inside the signed body)."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings.WEBHOOK, "SECRET", "delivery-secret")
    monkeypatch.setattr(webhook_delivery, "utc_now_iso", lambda: "2026-02-13T00:00:00Z")

    url = "https://a.example.com/hook"
    monkeypatch.setattr(
        webhook_delivery, "_get_webhook_urls", AsyncMock(return_value=[url])
    )

    fake_client = FakeAsyncClient(
        {url: httpx.Response(status_code=200, request=httpx.Request("POST", url))}
    )

    def async_client_factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        _ = (args, kwargs)
        return fake_client

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    payload = WebhookPayload(
        event_type="message.created",
        data={"id": "m_1", "workspace": "workspace-a"},
    )
    await webhook_delivery.deliver_webhook(
        payload, "workspace-a", tenant_id="tenant-xyz"
    )

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    sent_body: str = call["content"]

    assert json.loads(sent_body) == {
        "type": "message.created",
        "data": {"id": "m_1", "workspace": "workspace-a"},
        "timestamp": "2026-02-13T00:00:00Z",
        "tenant_id": "tenant-xyz",
    }
    assert '"tenant_id":"tenant-xyz"' in sent_body

    # region ai
    # Literal pin (in addition to the round-trip check above) so a future
    # sort_keys=False regression -- which json.loads == would not catch,
    # since dict equality ignores key order -- is caught directly here.
    # endregion
    expected_body = (
        '{"data":{"id":"m_1","workspace":"workspace-a"},'
        '"tenant_id":"tenant-xyz","timestamp":"2026-02-13T00:00:00Z",'
        '"type":"message.created"}'
    )
    assert sent_body == expected_body

    expected_signature = hmac.new(
        b"delivery-secret", sent_body.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert call["headers"]["X-Honcho-Signature"] == expected_signature


@pytest.mark.asyncio
async def test_deliver_webhook_flag_on_without_tenant_raises_and_makes_no_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-on with no tenant resolvable: fail closed before any DB session
    opens or any HTTP call is attempted."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)

    get_urls = AsyncMock()
    monkeypatch.setattr(webhook_delivery, "_get_webhook_urls", get_urls)

    fake_client = FakeAsyncClient({})

    def async_client_factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        _ = (args, kwargs)
        return fake_client

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    payload = WebhookPayload(event_type="message.created", data={"id": "m_1"})

    with pytest.raises(WebhookTenantUnresolved):
        await webhook_delivery.deliver_webhook(payload, "workspace-a", tenant_id=None)

    assert fake_client.calls == []
    # _get_webhook_urls is the only thing that opens a DB session
    # (tracked_db), so asserting it was never awaited stands in for "no DB
    # session was opened".
    get_urls.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_webhook_flag_on_with_empty_tenant_raises_and_makes_no_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-on with an empty-string tenant_id: fail closed before any DB
    session opens or any HTTP call is attempted, same as no tenant at all."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)

    get_urls = AsyncMock()
    monkeypatch.setattr(webhook_delivery, "_get_webhook_urls", get_urls)

    fake_client = FakeAsyncClient({})

    def async_client_factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        _ = (args, kwargs)
        return fake_client

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)

    payload = WebhookPayload(event_type="message.created", data={"id": "m_1"})

    with pytest.raises(WebhookTenantUnresolved):
        await webhook_delivery.deliver_webhook(payload, "workspace-a", tenant_id="")

    assert fake_client.calls == []
    # _get_webhook_urls is the only thing that opens a DB session
    # (tracked_db), so asserting it was never awaited stands in for "no DB
    # session was opened".
    get_urls.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_item_webhook_threads_queue_item_tenant_to_deliver_webhook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The webhook branch of `process_item` threads `QueueItem.tenant_id` to `deliver_webhook` explicitly."""
    # region ai
    # Why not tenant_context.get(): QueueItem.tenant_id is the authoritative
    # attribution column, so threading it explicitly here -- rather than
    # relying on delivery to read ambient tenant_context.get() itself --
    # keeps the value auditable at the one call site instead of ambient,
    # mirroring publish_webhook_event's own explicit-threading pattern for
    # the same tenant.
    # endregion
    deliver_mock = AsyncMock()
    monkeypatch.setattr(webhook_delivery, "deliver_webhook", deliver_mock)

    queue_item = QueueItem(
        task_type="webhook",
        work_unit_key="tenant-abc:webhook:workspace-a",
        payload={
            "task_type": "webhook",
            "event_type": "message.created",
            "data": {"id": "m_1"},
        },
        processed=False,
        workspace_name="workspace-a",
        tenant_id="tenant-abc",
    )

    await process_item(queue_item)

    deliver_mock.assert_awaited_once()
    _, call_kwargs = deliver_mock.await_args
    assert call_kwargs["tenant_id"] == "tenant-abc"


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
    await webhook_delivery.deliver_webhook(payload, "workspace-a")

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
    await webhook_delivery.deliver_webhook(payload, "workspace-a")


async def _webhook_keys(db: AsyncSession, workspace_id: str) -> list[str]:
    rows = await db.execute(
        select(QueueItem.work_unit_key).where(
            QueueItem.task_type == "webhook",
            QueueItem.workspace_name == workspace_id,
        )
    )
    return list(rows.scalars().all())


@pytest.mark.asyncio
async def test_publish_webhook_event_namespaces_key_with_explicit_tenant(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T5: queue.empty is published from the deriver's teardown, AFTER tenant_context
    # is reset, so the tenant must be threaded in explicitly. With it, the enqueued
    # QueueItem's key is tenant-namespaced — two tenants sharing a workspace_name no
    # longer collide onto one webhook work unit.
    monkeypatch.setattr(settings, "MULTI_TENANT", True)

    clear = tenant_context.set(None)  # no ambient tenant, like the real call site
    try:
        await publish_webhook_event(
            QueueEmptyEvent(workspace_id="ws-shared", queue_type="representation"),
            tenant_id="tenant-a",
        )
    finally:
        tenant_context.reset(clear)

    assert await _webhook_keys(db_session, "ws-shared") == [
        "tenant-a:webhook:ws-shared"
    ]


@pytest.mark.asyncio
async def test_publish_webhook_event_enqueues_nothing_without_tenant(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pre-fix bug: flag on with no tenant in scope, the webhook was published on
    # a non-namespaced (cross-tenant-colliding) key. Now the invariant guard in
    # construct_work_unit_key raises, publish_webhook_event swallows it, and nothing
    # is enqueued — a dropped webhook beats a cross-tenant one.
    monkeypatch.setattr(settings, "MULTI_TENANT", True)

    clear = tenant_context.set(None)
    try:
        await publish_webhook_event(
            QueueEmptyEvent(workspace_id="ws-orphan", queue_type="representation"),
        )
    finally:
        tenant_context.reset(clear)

    assert await _webhook_keys(db_session, "ws-orphan") == []


@pytest.mark.asyncio
async def test_publish_webhook_event_uses_ambient_tenant_when_omitted(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A6 regression: the /test route (src/routers/webhooks.py) calls
    # publish_webhook_event with no tenant_id, unlike the deriver's queue-drain
    # caller. Under MULTI_TENANT with an ambient tenant (set by auth on the
    # request), tracked_db and construct_work_unit_key both already fell back to
    # tenant_context internally, so the work_unit_key got a tenant prefix — but
    # QueueItem(tenant_id=tenant_id, ...) used the raw None, so the persisted row's
    # column disagreed with its own key's prefix. The fix resolves the ambient
    # tenant once, up front, so every use downstream agrees.
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    ambient_tenant_id = f"tenant-{generate_nanoid()}"

    token = tenant_context.set(ambient_tenant_id)
    try:
        await publish_webhook_event(
            QueueEmptyEvent(workspace_id="ws-ambient", queue_type="representation"),
        )
    finally:
        tenant_context.reset(token)

    result = await db_session.execute(
        select(QueueItem).where(
            QueueItem.task_type == "webhook",
            QueueItem.workspace_name == "ws-ambient",
        )
    )
    queue_item = result.scalar_one()
    assert queue_item.tenant_id == ambient_tenant_id
    assert tenant_id_for_work_unit_key(queue_item.work_unit_key) == ambient_tenant_id
