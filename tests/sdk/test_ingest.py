"""The SDK's ingest surface: auto workspaces, batched ingest, and derive later."""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from pydantic import ValidationError

from sdks.python.src.honcho.api_types import IngestRecord, PeerConfig
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.http import AsyncHonchoHTTPClient
from sdks.python.src.honcho.http.exceptions import ConflictError, NotFoundError


def _name() -> str:
    return f"sdk-ingest-{generate_nanoid(size=10).replace('_', 'a').lower()}"


def _sdk(client: TestClient, workspace_id: str, *, auto: bool = False) -> Honcho:
    return Honcho(
        workspace_id=workspace_id,
        base_url=str(client.base_url),
        http_client=httpx.Client(
            transport=client._transport,  # pyright: ignore
            base_url=str(client.base_url),
            headers=client.headers,
        ),
        auto=auto,
    )


def _records(count: int) -> Iterator[dict[str, Any]]:
    """A generator, so the SDK has to read it lazily."""
    for n in range(count):
        yield {
            "source": "discord",
            "source_instance": "guild1",
            "source_id": f"msg-{n}",
            "created_at": f"2026-11-04T{10 + n // 60:02d}:{n % 60:02d}:00Z",
            "container": {"id": f"chan-{n % 3}", "name": f"channel {n % 3}"},
            "author": {"id": f"user-{n % 4}", "name": f"Person {n % 4}"},
            "content": f"message {n}",
        }


def _workspace(client: TestClient, workspace_id: str) -> dict[str, Any] | None:
    """The workspace as the server has it, read without creating it."""
    response = client.post(
        "/v3/workspaces/list", json={"filters": {"id": workspace_id}}
    )
    assert response.status_code == 200
    items = response.json()["items"]
    return items[0] if items else None


def test_an_auto_client_creates_an_auto_workspace(client: TestClient) -> None:
    name = _name()
    sdk = _sdk(client, name, auto=True)

    sdk.get_metadata()

    workspace = _workspace(client, name)
    assert workspace is not None
    assert workspace["auto"] is True


def test_an_auto_client_refuses_an_existing_regular_workspace(
    client: TestClient,
) -> None:
    name = _name()
    _sdk(client, name).get_metadata()

    with pytest.raises(ConflictError):
        _sdk(client, name, auto=True).get_metadata()


def test_ingest_sends_every_record_in_batches_and_numbers_them_in_order(
    client: TestClient,
) -> None:
    sdk = _sdk(client, _name(), auto=True)

    result = sdk.ingest(_records(250), derive="later")

    assert [r.index for r in result.records] == list(range(250))
    assert {r.outcome for r in result.records} == {"created"}
    assert {r.derivation for r in result.records} == {"deferred"}
    # Each result belongs to the record at its index.
    messages = {r.message_id for r in result.records}
    assert len(messages) == 250

    again = sdk.ingest(_records(250), derive="later")

    assert {r.outcome for r in again.records} == {"duplicate"}
    assert [r.message_id for r in again.records] == [
        r.message_id for r in result.records
    ]


def test_ingest_derive_releases_every_deferred_record(client: TestClient) -> None:
    sdk = _sdk(client, _name(), auto=True)
    sdk.ingest(_records(120), derive="later")

    released = sdk.ingest_derive()

    assert (released.pending, released.remaining) == (120, 0)
    assert (sdk.ingest_derive().pending, sdk.ingest_derive().remaining) == (0, 0)


def test_ingest_without_auto_does_not_create_the_workspace(
    client: TestClient,
) -> None:
    name = _name()

    with pytest.raises(NotFoundError):
        _sdk(client, name).ingest(_records(1))

    assert _workspace(client, name) is None


def test_ingest_into_a_regular_workspace_is_refused(client: TestClient) -> None:
    sdk = _sdk(client, _name())
    sdk.get_metadata()

    with pytest.raises(ConflictError):
        sdk.ingest(_records(1))


def test_a_client_without_auto_can_ingest_into_an_existing_auto_workspace(
    client: TestClient,
) -> None:
    name = _name()
    _sdk(client, name, auto=True).get_metadata()

    result = _sdk(client, name).ingest([IngestRecord.model_validate(next(_records(1)))])

    assert [r.outcome for r in result.records] == ["created"]


def test_a_bad_record_raises_before_its_batch_is_sent(client: TestClient) -> None:
    name = _name()
    sdk = _sdk(client, name, auto=True)
    records = list(_records(3))
    del records[2]["created_at"]

    with pytest.raises(ValidationError):
        sdk.ingest(records)

    # The workspace was created, but none of the batch was written.
    assert sdk.sessions().total == 0


def test_ingest_of_nothing_sends_nothing(client: TestClient) -> None:
    name = _name()

    assert _sdk(client, name).ingest([]).records == []
    assert _workspace(client, name) is None


def _derive_transport(steps: list[tuple[int, int]]) -> httpx.MockTransport:
    """Answers each derive call with the next (pending, remaining) step."""
    calls = iter(steps)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/ingest/derive")
        pending, remaining = next(calls)
        return httpx.Response(
            200, content=json.dumps({"pending": pending, "remaining": remaining})
        )

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        ([(1000, 1500), (1000, 500), (500, 0)], (2500, 0)),
        # Records still arriving deferred: stop once a call releases nothing.
        ([(1000, 40), (0, 40)], (1000, 40)),
    ],
)
def test_ingest_derive_calls_until_nothing_is_left_to_release(
    steps: list[tuple[int, int]], expected: tuple[int, int]
) -> None:
    sdk = Honcho(
        workspace_id="w",
        base_url="http://honcho.test",
        http_client=httpx.Client(
            transport=_derive_transport(steps), base_url="http://honcho.test"
        ),
    )

    released = sdk.ingest_derive()

    assert (released.pending, released.remaining) == expected


@pytest.mark.asyncio
async def test_async_ingest_and_derive(client: TestClient) -> None:
    sdk = _sdk(client, _name(), auto=True)
    async_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=client.app),
        base_url=str(client.base_url),
        headers=client.headers,
    )
    sdk._async_http = AsyncHonchoHTTPClient(  # pyright: ignore[reportPrivateUsage]
        base_url=str(client.base_url), api_key=None, http_client=async_client
    )
    try:
        result = await sdk.aio.ingest(_records(150), derive="later")
        released = await sdk.aio.ingest_derive()
    finally:
        await async_client.aclose()

    assert [r.index for r in result.records] == list(range(150))
    assert {r.outcome for r in result.records} == {"created"}
    assert (released.pending, released.remaining) == (150, 0)


def test_review_an_import_before_deriving_it(client: TestClient) -> None:
    sdk = _sdk(client, _name(), auto=True)
    sdk.ingest(_records(12), derive="later")

    status = sdk.ingest_status()
    people = list(sdk.ingest_people(size=2))
    sessions = list(sdk.ingest_sessions())

    assert (status.records.deferred, status.people, status.sessions) == (12, 4, 3)
    assert status.deferred_tokens > 0
    assert [p.messages for p in people] == [3, 3, 3, 3]
    assert {p.display_name for p in people} == {f"Person {n}" for n in range(4)}
    assert people[0].accounts[0].source == "discord"
    assert {s.display_name for s in sessions} == {f"channel {n}" for n in range(3)}
    assert all(p.modeled for p in people) and all(s.modeled for s in sessions)

    excluded = people[0]
    sdk.peer(excluded.peer_id).set_configuration(PeerConfig(observe_me=False))

    assert sdk.ingest_status().people_not_modeled == 1
    assert {p.peer_id for p in sdk.ingest_people() if not p.modeled} == {
        excluded.peer_id
    }


@pytest.mark.asyncio
async def test_async_review(client: TestClient) -> None:
    sdk = _sdk(client, _name(), auto=True)
    sdk.ingest(_records(6), derive="later")
    async_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=client.app),
        base_url=str(client.base_url),
        headers=client.headers,
    )
    sdk._async_http = AsyncHonchoHTTPClient(  # pyright: ignore[reportPrivateUsage]
        base_url=str(client.base_url), api_key=None, http_client=async_client
    )
    try:
        status = await sdk.aio.ingest_status()
        people = (await sdk.aio.ingest_people()).items
        sessions = (await sdk.aio.ingest_sessions()).items
    finally:
        await async_client.aclose()

    assert (status.records.deferred, len(people), len(sessions)) == (6, 4, 3)
