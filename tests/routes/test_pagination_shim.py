"""Pagination across every list endpoint: cursor mode and the offset shim.

Each endpoint is walked three ways (offset through the shim, plain offset, and
cursor) and must return the same rows in the same order. The shim tests then
pin its own behavior: sequential pages seek instead of scanning, `total` is
cached past page 1, paging past the end is free, and a cache failure falls back
to the plain offset query.
"""

from collections.abc import Callable
from typing import Any, final

import pytest
from fastapi.testclient import TestClient
from fastapi_pagination.cursor import encode_cursor
from nanoid import generate as generate_nanoid

from src.cache.client import cache
from src.config import settings
from src.models import Peer, Workspace
from src.telemetry.prometheus.metrics import (
    PaginationShimOutcomes,
    pagination_offset_shim_counter,
)

SEEDED = 5
SIZE = 2


def _new_id() -> str:
    return str(generate_nanoid())


@final
class _Endpoint:
    """A list endpoint plus a way to seed it with `SEEDED` rows."""

    def __init__(
        self,
        name: str,
        method: str,
        seed: Callable[[TestClient, str, str], tuple[str, set[str]]],
        reversible: bool = True,
    ) -> None:
        self.name = name
        self.method = method
        self.seed = seed
        self.reversible = reversible

    def __repr__(self) -> str:
        return self.name


def _seed_peers(client: TestClient, ws: str, _peer: str) -> tuple[str, set[str]]:
    names = {_new_id() for _ in range(SEEDED)}
    for name in names:
        assert client.post(f"/v3/workspaces/{ws}/peers", json={"id": name}).is_success
    return f"/v3/workspaces/{ws}/peers/list", names


def _seed_sessions(client: TestClient, ws: str, peer: str) -> tuple[str, set[str]]:
    names = {_new_id() for _ in range(SEEDED)}
    for name in names:
        response = client.post(
            f"/v3/workspaces/{ws}/sessions", json={"id": name, "peers": {peer: {}}}
        )
        assert response.is_success, response.text
    return f"/v3/workspaces/{ws}/sessions/list", names


def _seed_peer_sessions(client: TestClient, ws: str, peer: str) -> tuple[str, set[str]]:
    _, names = _seed_sessions(client, ws, peer)
    return f"/v3/workspaces/{ws}/peers/{peer}/sessions", names


def _seed_messages(client: TestClient, ws: str, peer: str) -> tuple[str, set[str]]:
    session = _new_id()
    assert client.post(f"/v3/workspaces/{ws}/sessions", json={"id": session}).is_success
    response = client.post(
        f"/v3/workspaces/{ws}/sessions/{session}/messages",
        json={
            "messages": [
                {"content": f"message {i}", "peer_id": peer} for i in range(SEEDED)
            ]
        },
    )
    assert response.is_success, response.text
    ids = {message["id"] for message in response.json()}
    return f"/v3/workspaces/{ws}/sessions/{session}/messages/list", ids


def _seed_session_peers(
    client: TestClient, ws: str, _peer: str
) -> tuple[str, set[str]]:
    # Added in one request, so every membership shares a joined_at: the walk
    # only stays total because the peer id breaks the tie.
    names = {_new_id() for _ in range(SEEDED)}
    session = _new_id()
    response = client.post(
        f"/v3/workspaces/{ws}/sessions",
        json={"id": session, "peers": {name: {} for name in names}},
    )
    assert response.is_success, response.text
    return f"/v3/workspaces/{ws}/sessions/{session}/peers", names


def _seed_scopes(client: TestClient, ws: str, _peer: str) -> tuple[str, set[str]]:
    names = {_new_id() for _ in range(SEEDED)}
    for name in names:
        assert client.post(f"/v3/workspaces/{ws}/scopes", json={"id": name}).is_success
    return f"/v3/workspaces/{ws}/scopes/list", names


def _seed_scope_sessions(
    client: TestClient, ws: str, _peer: str
) -> tuple[str, set[str]]:
    # One request, so the memberships share a joined_at, which lives on the
    # joined table rather than on the returned sessions.
    scope = _new_id()
    assert client.post(f"/v3/workspaces/{ws}/scopes", json={"id": scope}).is_success
    names = {_new_id() for _ in range(SEEDED)}
    for name in names:
        assert client.post(
            f"/v3/workspaces/{ws}/sessions", json={"id": name}
        ).is_success
    response = client.post(
        f"/v3/workspaces/{ws}/scopes/{scope}/sessions",
        json={"session_ids": sorted(names)},
    )
    assert response.is_success, response.text
    return f"/v3/workspaces/{ws}/scopes/{scope}/sessions/list", names


def _seed_webhooks(client: TestClient, ws: str, _peer: str) -> tuple[str, set[str]]:
    ids: set[str] = set()
    for i in range(SEEDED):
        response = client.post(
            f"/v3/workspaces/{ws}/webhooks",
            json={"url": f"http://example.com/hook-{i}"},
        )
        assert response.is_success, response.text
        ids.add(response.json()["id"])
    return f"/v3/workspaces/{ws}/webhooks", ids


def _seed_workspaces(client: TestClient, _ws: str, _peer: str) -> tuple[str, set[str]]:
    names = {_new_id() for _ in range(SEEDED)}
    for name in names:
        assert client.post("/v3/workspaces", json={"id": name}).is_success
    return "/v3/workspaces/list", names


ENDPOINTS = [
    _Endpoint("peers", "POST", _seed_peers),
    _Endpoint("sessions", "POST", _seed_sessions),
    _Endpoint("peer_sessions", "POST", _seed_peer_sessions),
    _Endpoint("messages", "POST", _seed_messages),
    _Endpoint("session_peers", "GET", _seed_session_peers, reversible=False),
    _Endpoint("scopes", "POST", _seed_scopes),
    _Endpoint("scope_sessions", "POST", _seed_scope_sessions),
    _Endpoint("webhooks", "GET", _seed_webhooks, reversible=False),
    _Endpoint("workspaces", "POST", _seed_workspaces),
]


def _get(
    client: TestClient, endpoint: _Endpoint, url: str, params: dict[str, Any]
) -> dict[str, Any]:
    response = client.request(endpoint.method, url, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _walk_offset(
    client: TestClient, endpoint: _Endpoint, url: str, extra: dict[str, Any]
) -> list[str]:
    ids: list[str] = []
    page = 1
    while True:
        data = _get(client, endpoint, url, {"page": page, "size": SIZE, **extra})
        assert set(data) == {"items", "total", "page", "size", "pages"}
        ids += [item["id"] for item in data["items"]]
        if page >= data["pages"]:
            return ids
        page += 1
        assert page <= 50, "offset walk did not terminate"


def _walk_cursor(
    client: TestClient, endpoint: _Endpoint, url: str, extra: dict[str, Any]
) -> list[str]:
    ids: list[str] = []
    cursor: str | None = ""
    while cursor is not None:
        data = _get(client, endpoint, url, {"cursor": cursor, "size": SIZE, **extra})
        assert data["total"] is None
        ids += [item["id"] for item in data["items"]]
        cursor = data["next_page"]
        assert len(ids) <= 200, "cursor walk did not terminate"
    return ids


@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=repr)
@pytest.mark.parametrize("reverse", [False, True])
def test_every_walk_returns_the_same_rows_in_the_same_order(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    endpoint: _Endpoint,
    reverse: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    if reverse and not endpoint.reversible:
        pytest.skip("endpoint has no reverse parameter")
    workspace, peer = sample_data
    url, seeded = endpoint.seed(client, workspace.name, peer.name)
    extra = {"reverse": reverse} if endpoint.reversible else {}

    shim_ids = _walk_offset(client, endpoint, url, extra)
    with monkeypatch.context() as m:
        m.setattr(settings.CACHE, "PAGINATION_OFFSET_SHIM", False)
        plain_ids = _walk_offset(client, endpoint, url, extra)
    cursor_ids = _walk_cursor(client, endpoint, url, extra)

    assert seeded <= set(plain_ids)
    assert len(plain_ids) == len(set(plain_ids))
    assert shim_ids == plain_ids
    assert cursor_ids == plain_ids


def _shim_counts() -> dict[PaginationShimOutcomes, float]:
    return {
        outcome: pagination_offset_shim_counter.labels(
            outcome=outcome.value
        )._value.get()  # pyright: ignore
        for outcome in PaginationShimOutcomes
    }


def test_sequential_offset_walk_seeks_after_page_one(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    before = _shim_counts()

    _walk_offset(client, endpoint, url, {})  # 5 rows at size 2: pages 1, 2, 3

    after = _shim_counts()
    delta = {outcome: after[outcome] - before[outcome] for outcome in after}
    assert delta == {
        PaginationShimOutcomes.FIRST: 1,
        PaginationShimOutcomes.HIT: 2,
        PaginationShimOutcomes.MISS: 0,
    }


def test_total_is_cached_past_page_one(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    assert _get(client, endpoint, url, {"page": 1, "size": SIZE})["total"] == SEEDED

    _seed_sessions(client, workspace.name, peer.name)

    # Page 2 reuses the count page 1 cached; page 1 always recounts.
    assert _get(client, endpoint, url, {"page": 2, "size": SIZE})["total"] == SEEDED
    assert _get(client, endpoint, url, {"page": 1, "size": SIZE})["total"] == 2 * SEEDED


def test_a_cached_total_does_not_end_a_walk_early(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Rows appended mid-walk land at the end of an oldest-first list; the walk must reach them."""
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    data = _get(client, endpoint, url, {"page": 1, "size": SIZE})
    ids = [item["id"] for item in data["items"]]

    _seed_sessions(client, workspace.name, peer.name)

    page = 2
    while True:
        data = _get(client, endpoint, url, {"page": page, "size": SIZE})
        ids += [item["id"] for item in data["items"]]
        if page >= data["pages"]:
            break
        page += 1
    assert len(ids) == len(set(ids)) == 2 * SEEDED


def test_the_last_page_fills_when_rows_are_appended(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Reaching the end stores nothing, so a later jump to the last page sees new rows."""
    workspace, peer = sample_data
    session_ids = [_new_id() for _ in range(4)]
    for session_id in session_ids:
        assert client.post(
            f"/v3/workspaces/{workspace.name}/sessions", json={"id": session_id}
        ).is_success
    url = f"/v3/workspaces/{workspace.name}/sessions/list"
    endpoint = ENDPOINTS[1]
    _walk_offset(client, endpoint, url, {})  # pages 1 and 2, both full

    _seed_sessions(client, workspace.name, peer.name)
    last = _get(client, endpoint, url, {"page": 1, "size": SIZE})["pages"]
    shim = _get(client, endpoint, url, {"page": last, "size": SIZE})
    monkeypatch.setattr(settings.CACHE, "PAGINATION_OFFSET_SHIM", False)
    plain = _get(client, endpoint, url, {"page": last, "size": SIZE})

    assert shim["items"]
    assert shim == plain


def test_inserts_during_a_newest_first_walk_cause_no_duplicates(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Plain OFFSET shifts later pages when rows land at the head; a seek doesn't."""
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    params = {"size": SIZE, "reverse": True}

    first = _get(client, endpoint, url, {"page": 1, **params})["items"]
    # One new row at the head: plain OFFSET would now repeat page 1's last row.
    response = client.post(
        f"/v3/workspaces/{workspace.name}/sessions", json={"id": _new_id()}
    )
    assert response.is_success
    second = _get(client, endpoint, url, {"page": 2, **params})["items"]

    first_ids = {item["id"] for item in first}
    assert not first_ids & {item["id"] for item in second}


def test_cache_failure_falls_back_to_offset(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    expected = _walk_offset(client, endpoint, url, {})

    async def broken_get(*_args: object, **_kwargs: object):
        raise ConnectionError("cache down")

    async def broken_set(*_args: object, **_kwargs: object):
        raise ConnectionError("cache down")

    monkeypatch.setattr(cache, "get", broken_get)
    monkeypatch.setattr(cache, "set", broken_set)

    assert _walk_offset(client, endpoint, url, {}) == expected


@pytest.mark.parametrize(
    "cursor",
    [
        encode_cursor("not a bookmark"),  # decodes, but isn't a bookmark
        encode_cursor(">s:a\rb"),  # breaks the bookmark's CSV parser
        encode_cursor(">i:1~i:2~i:3"),  # the wrong column count for this query
        encode_cursor(">s:abc"),  # a string where the bigint id is compared
        encode_cursor(">i:99999999999999999999999"),  # out of bigint range
    ],
)
def test_a_cursor_that_does_not_fit_is_a_400(
    client: TestClient, sample_data: tuple[Workspace, Peer], cursor: str
):
    """Not a 500, and not the 404 the messages route maps other ValueErrors to."""
    workspace, peer = sample_data
    url, _ = _seed_messages(client, workspace.name, peer.name)

    response = client.post(url, params={"cursor": cursor})

    assert response.status_code == 400, response.text
