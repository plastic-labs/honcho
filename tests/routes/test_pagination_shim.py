"""Pagination across every list endpoint: cursor mode and the offset shim.

Each endpoint is walked four ways (offset through the shim, plain offset,
cursor, and offset page 1 handing off to cursor mode through `next_cursor`)
and must return the same rows in the same order. The shim tests then
pin its own behavior: sequential pages seek instead of scanning, `total` is
cached past page 1, paging past the end is free, a page that ends the list
needs no count, an empty page after a deleted tail ends the walk, a list
past the count limit reports a full count taken in the background (and skips
the capped count while that is fresh), `total_kind` says how to read every
total, and a cache failure falls back to the plain offset query.
"""

import time
from collections.abc import Callable
from typing import Any, final

import pytest
from fastapi.testclient import TestClient
from fastapi_pagination.cursor import encode_cursor
from nanoid import generate as generate_nanoid
from sqlalchemy import Select, literal, select

from src.cache.client import cache
from src.config import settings
from src.models import Peer, Workspace
from src.telemetry.prometheus.metrics import (
    PaginationShimOutcomes,
    pagination_offset_shim_counter,
)
from src.utils import pagination

SEEDED = 5
SIZE = 2
OFFSET_FIELDS = {
    "items",
    "total",
    "total_kind",
    "page",
    "size",
    "pages",
    "next_cursor",
}


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
        assert set(data) == OFFSET_FIELDS
        ids += [item["id"] for item in data["items"]]
        if page >= data["pages"]:
            assert data["next_cursor"] is None
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


def _walk_offset_then_cursor(
    client: TestClient, endpoint: _Endpoint, url: str, extra: dict[str, Any]
) -> list[str]:
    """Read page 1 in offset mode, then follow its `next_cursor` in cursor mode."""
    data = _get(client, endpoint, url, {"page": 1, "size": SIZE, **extra})
    ids = [item["id"] for item in data["items"]]
    cursor: str | None = data["next_cursor"]
    while cursor is not None:
        data = _get(client, endpoint, url, {"cursor": cursor, "size": SIZE, **extra})
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
    handoff_ids = _walk_offset_then_cursor(client, endpoint, url, extra)

    assert seeded <= set(plain_ids)
    assert len(plain_ids) == len(set(plain_ids))
    assert shim_ids == plain_ids
    assert cursor_ids == plain_ids
    assert handoff_ids == plain_ids


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


@pytest.mark.parametrize("shim", [True, False], ids=["shim", "plain"])
def test_a_capped_total_does_not_end_a_walk_early(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    shim: bool,
):
    """`total` stops one past the limit, but `pages` still carries a walk to the last row."""
    workspace, peer = sample_data
    url, seeded = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    monkeypatch.setattr(settings, "PAGINATION_COUNT_LIMIT", SEEDED - 2)
    monkeypatch.setattr(settings.CACHE, "PAGINATION_OFFSET_SHIM", shim)

    first = _get(client, endpoint, url, {"page": 1, "size": SIZE})
    assert first["total"] == SEEDED - 1
    assert first["pages"] == 2

    ids = _walk_offset(client, endpoint, url, {})
    assert seeded <= set(ids)
    assert len(ids) == len(set(ids))
    last = _get(client, endpoint, url, {"page": -(-len(ids) // SIZE), "size": SIZE})
    assert last["total"] == len(ids)


@pytest.mark.parametrize("shim", [True, False], ids=["shim", "plain"])
def test_a_page_past_the_end_reports_the_real_total(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    shim: bool,
):
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    monkeypatch.setattr(settings.CACHE, "PAGINATION_OFFSET_SHIM", shim)
    expected = _get(client, endpoint, url, {"page": 1, "size": SIZE})["total"]

    far = _get(client, endpoint, url, {"page": 1000, "size": SIZE})

    assert far["items"] == []
    assert far["total"] == expected
    assert far["pages"] == -(-expected // SIZE)


@pytest.mark.parametrize("shim", [True, False], ids=["shim", "plain"])
@pytest.mark.parametrize("seeded", [True, False], ids=["one_page", "empty"])
def test_a_list_that_fits_on_page_one_is_not_counted(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    shim: bool,
    seeded: bool,
):
    """Page 1 with no next page has seen every row, so its total needs no count."""
    workspace, peer = sample_data
    url = f"/v3/workspaces/{workspace.name}/sessions/list"
    expected = 0
    if seeded:
        url, _ = _seed_sessions(client, workspace.name, peer.name)
        expected = SEEDED
    endpoint = ENDPOINTS[1]
    monkeypatch.setattr(settings.CACHE, "PAGINATION_OFFSET_SHIM", shim)

    def no_count(_stmt: Select[Any]) -> Select[Any]:
        raise AssertionError("counted a list that fits on page 1")

    monkeypatch.setattr(pagination, "_capped_count_query", no_count)

    data = _get(client, endpoint, url, {"page": 1, "size": SEEDED})

    assert len(data["items"]) == expected
    assert data["total"] == expected
    assert data["pages"] == (1 if seeded else 0)


def test_a_stale_high_total_does_not_carry_a_walk_past_the_last_page(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """A cached count can be too high (rows deleted after it was taken); the last page still ends the walk."""
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]

    def stale_count(_stmt: Select[Any]) -> Select[Any]:
        return select(literal(100))

    with monkeypatch.context() as patch:
        patch.setattr(pagination, "_capped_count_query", stale_count)
        first = _get(client, endpoint, url, {"page": 1, "size": SIZE})
    assert first["total"] == 100

    visited = [1]
    data = first
    while visited[-1] < data["pages"]:
        visited.append(visited[-1] + 1)
        data = _get(client, endpoint, url, {"page": visited[-1], "size": SIZE})

    assert visited == [1, 2, 3]
    assert data["total"] == SEEDED
    assert data["pages"] == 3


def _eventually_total(
    client: TestClient, endpoint: _Endpoint, url: str, expected: int
) -> dict[str, Any]:
    """Re-read page 1 until the background count lands (it runs after a response)."""
    data: dict[str, Any] = {}
    for _ in range(100):
        data = _get(client, endpoint, url, {"page": 1, "size": SIZE})
        if data["total"] == expected:
            return data
        time.sleep(0.05)
    raise AssertionError(f"total stayed {data.get('total')}, expected {expected}")


def test_a_list_past_the_limit_reports_its_full_count(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Past the limit, `total` is the last full count, taken in the background and refreshed when stale."""
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    monkeypatch.setattr(settings, "PAGINATION_COUNT_LIMIT", SEEDED - 2)

    first = _get(client, endpoint, url, {"page": 1, "size": SIZE})
    assert first["total"] == SEEDED - 1  # nothing fully counted yet
    _eventually_total(client, endpoint, url, SEEDED)

    _seed_sessions(client, workspace.name, peer.name)
    # Within the refresh interval the last full count stands.
    assert _get(client, endpoint, url, {"page": 1, "size": SIZE})["total"] == SEEDED

    monkeypatch.setattr(settings.CACHE, "PAGINATION_ESTIMATE_REFRESH_SECONDS", 0)
    data = _eventually_total(client, endpoint, url, 2 * SEEDED)
    assert data["pages"] == SEEDED  # 10 rows at size 2

    ids = _walk_offset(client, endpoint, url, {})
    assert len(ids) == len(set(ids)) == 2 * SEEDED


def test_without_the_cache_a_list_past_the_limit_stays_capped(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    monkeypatch.setattr(settings, "PAGINATION_COUNT_LIMIT", SEEDED - 2)
    monkeypatch.setattr(settings.CACHE, "ENABLED", False)

    for _ in range(3):
        data = _get(client, endpoint, url, {"page": 1, "size": SIZE})
        assert data["total"] == SEEDED - 1
        time.sleep(0.05)
    assert not pagination._estimate_tasks  # pyright: ignore[reportPrivateUsage]


def _high_count(_stmt: Select[Any]) -> Select[Any]:
    return select(literal(100))


@pytest.mark.parametrize("shim", [True, False], ids=["shim", "plain"])
def test_a_walk_stops_at_a_tail_deleted_mid_walk(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    shim: bool,
):
    """A tail deleted mid-walk empties the next page; a high total must not carry the walk on."""
    workspace, peer = sample_data
    url, names = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    monkeypatch.setattr(settings.CACHE, "PAGINATION_OFFSET_SHIM", shim)
    # Every count is too high, as a stale cached count or a background count would be.
    monkeypatch.setattr(pagination, "_capped_count_query", _high_count)

    pages = [_get(client, endpoint, url, {"page": p, "size": SIZE}) for p in (1, 2)]
    assert pages[1]["total"] == 100
    seen = {item["id"] for page in pages for item in page["items"]}
    for name in names - seen:
        response = client.delete(f"/v3/workspaces/{workspace.name}/sessions/{name}")
        assert response.status_code < 300, response.text

    data = _get(client, endpoint, url, {"page": 3, "size": SIZE})

    assert data["items"] == []
    assert data["total"] == 2 * SIZE
    assert data["pages"] == 2  # a client walking `page < pages` stops here
    assert data["total_kind"] == "at_most"


def test_total_kind_says_how_to_read_every_total(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]

    def page(n: int) -> dict[str, Any]:
        return _get(client, endpoint, url, {"page": n, "size": SIZE})

    # Counted under the limit, cached past page 1, and the page that ends the list.
    assert [(page(n)["total"], page(n)["total_kind"]) for n in (1, 2, 3)] == [
        (SEEDED, "exact")
    ] * 3

    # Past the limit with nothing fully counted yet: the limit plus one.
    monkeypatch.setattr(settings, "PAGINATION_COUNT_LIMIT", SEEDED - 2)
    capped = page(1)
    assert (capped["total"], capped["total_kind"]) == (SEEDED - 1, "at_least")

    # The background full count.
    data = _eventually_total(client, endpoint, url, SEEDED)
    assert data["total_kind"] == "estimate"

    # A count below the rows already seen is raised past them.
    def low_count(_stmt: Select[Any]) -> Select[Any]:
        return select(literal(1))

    monkeypatch.setattr(settings, "PAGINATION_COUNT_LIMIT", 10_000)
    with monkeypatch.context() as patch:
        patch.setattr(settings.CACHE, "PAGINATION_OFFSET_SHIM", False)
        patch.setattr(pagination, "_capped_count_query", low_count)
        low = page(2)
    assert (low["total"], low["total_kind"]) == (2 * SIZE + 1, "at_least")


def test_a_fresh_full_count_past_the_limit_skips_the_capped_count(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    url, _ = _seed_sessions(client, workspace.name, peer.name)
    endpoint = ENDPOINTS[1]
    monkeypatch.setattr(settings, "PAGINATION_COUNT_LIMIT", SEEDED - 2)
    _get(client, endpoint, url, {"page": 1, "size": SIZE})
    _eventually_total(client, endpoint, url, SEEDED)

    def no_count(_stmt: Select[Any]) -> Select[Any]:
        raise AssertionError("ran the capped count with a fresh full count stored")

    monkeypatch.setattr(pagination, "_capped_count_query", no_count)
    for n in (1, 2):
        data = _get(client, endpoint, url, {"page": n, "size": SIZE})
        assert (data["total"], data["total_kind"]) == (SEEDED, "estimate")
