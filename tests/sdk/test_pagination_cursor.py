"""SDK iteration reads past page 1 in cursor mode when the server offers `next_cursor`."""

from typing import Any

import pytest
from pydantic import BaseModel

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.pagination import AsyncPage, SyncPage

SEEDED = 7
SIZE = 2


def _spy_queries(
    honcho_client: Honcho, client_type: str, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """Record the query of every list POST the client sends."""
    queries: list[dict[str, Any]] = []
    http: Any = (
        honcho_client._async_http  # pyright: ignore[reportPrivateUsage]
        if client_type == "async"
        else honcho_client._http  # pyright: ignore[reportPrivateUsage]
    )
    original = http.post

    if client_type == "async":

        async def apost(path: str, **kwargs: Any) -> Any:
            if path.endswith("/list"):
                queries.append(dict(kwargs.get("query") or {}))
            return await original(path, **kwargs)

        monkeypatch.setattr(http, "post", apost)
    else:

        def post(path: str, **kwargs: Any) -> Any:
            if path.endswith("/list"):
                queries.append(dict(kwargs.get("query") or {}))
            return original(path, **kwargs)

        monkeypatch.setattr(http, "post", post)
    return queries


async def _seed_peers(honcho_client: Honcho, client_type: str, prefix: str) -> None:
    for i in range(SEEDED):
        if client_type == "async":
            peer = await honcho_client.aio.peer(id=f"{prefix}-{i}")
            await peer.aio.get_metadata()
        else:
            honcho_client.peer(id=f"{prefix}-{i}").get_metadata()


async def _first_page(
    honcho_client: Honcho, client_type: str, **kwargs: Any
) -> SyncPage[Any, Any] | AsyncPage[Any, Any]:
    if client_type == "async":
        return await honcho_client.aio.peers(size=SIZE, **kwargs)
    return honcho_client.peers(size=SIZE, **kwargs)


async def _iterate(page: SyncPage[Any, Any] | AsyncPage[Any, Any]) -> list[str]:
    if isinstance(page, AsyncPage):
        return [peer.id async for peer in page]
    return [peer.id for peer in page]


async def _manual_offset_walk(
    page: SyncPage[Any, Any] | AsyncPage[Any, Any],
) -> list[str]:
    ids: list[str] = []
    current: SyncPage[Any, Any] | AsyncPage[Any, Any] | None = page
    while current is not None:
        ids += [peer.id for peer in current.items]
        if isinstance(current, AsyncPage):
            current = await current.get_next_page()
        else:
            current = current.get_next_page()
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_iteration_follows_next_cursor(
    client_fixture: tuple[Honcho, str],
    monkeypatch: pytest.MonkeyPatch,
    reverse: bool,
):
    honcho_client, client_type = client_fixture
    await _seed_peers(honcho_client, client_type, f"cursor-iter-{client_type}")
    expected = await _manual_offset_walk(
        await _first_page(honcho_client, client_type, reverse=reverse)
    )
    assert len(expected) >= SEEDED

    queries = _spy_queries(honcho_client, client_type, monkeypatch)
    first = await _first_page(honcho_client, client_type, reverse=reverse)
    assert first.next_cursor is not None
    assert first.total_kind == "exact"
    ids = await _iterate(first)

    assert ids == expected
    assert len(ids) == len(set(ids))
    assert "page" in queries[0]
    assert len(queries) == -(-len(expected) // SIZE)
    for query in queries[1:]:
        assert "page" not in query
        assert query["cursor"]
        assert query["size"] == SIZE
        assert ("reverse" in query) == reverse


@pytest.mark.asyncio
async def test_get_next_page_stays_offset(
    client_fixture: tuple[Honcho, str], monkeypatch: pytest.MonkeyPatch
):
    honcho_client, client_type = client_fixture
    await _seed_peers(honcho_client, client_type, f"cursor-manual-{client_type}")
    queries = _spy_queries(honcho_client, client_type, monkeypatch)

    first = await _first_page(honcho_client, client_type)
    if isinstance(first, AsyncPage):
        second = await first.get_next_page()
    else:
        second = first.get_next_page()

    assert second is not None
    assert second.page == 2
    assert [query.get("page") for query in queries] == [1, 2]
    assert all("cursor" not in query for query in queries)


class _Item(BaseModel):
    id: str


def _offset_data(page: int, pages: int, **extra: Any) -> dict[str, Any]:
    return {
        "items": [{"id": f"p{page}"}],
        "page": page,
        "size": 1,
        "total": pages,
        "pages": pages,
        **extra,
    }


def test_a_server_without_next_cursor_is_walked_by_page_number():
    def fetch_next(page: int) -> SyncPage[_Item, _Item]:
        return SyncPage(_offset_data(page, 3), _Item, None, fetch_next, fetch_cursor)

    def fetch_cursor(_cursor: str) -> dict[str, Any]:
        raise AssertionError("followed a cursor the server never sent")

    first = SyncPage(_offset_data(1, 3), _Item, None, fetch_next, fetch_cursor)

    assert [item.id for item in first] == ["p1", "p2", "p3"]
    assert first.next_cursor is None


@pytest.mark.asyncio
async def test_an_async_server_without_next_cursor_is_walked_by_page_number():
    async def fetch_next(page: int) -> AsyncPage[_Item, _Item]:
        return AsyncPage(_offset_data(page, 3), _Item, None, fetch_next, fetch_cursor)

    async def fetch_cursor(_cursor: str) -> dict[str, Any]:
        raise AssertionError("followed a cursor the server never sent")

    first = AsyncPage(_offset_data(1, 3), _Item, None, fetch_next, fetch_cursor)

    items = [item async for item in first]
    assert [item.id for item in items] == ["p1", "p2", "p3"]


def test_cursor_iteration_ends_at_a_null_next_page_and_keeps_the_transform():
    responses = {
        "c1": {"items": [{"id": "b"}], "total": None, "next_page": "c2"},
        "c2": {"items": [{"id": "c"}], "total": None, "next_page": None},
    }

    def fetch_next(_page: int) -> SyncPage[_Item, str]:
        raise AssertionError("walked by page number past a page with next_cursor")

    first = SyncPage(
        _offset_data(1, 9, next_cursor="c1", total_kind="estimate")
        | {"items": [{"id": "a"}]},
        _Item,
        lambda item: item.id.upper(),
        fetch_next,
        lambda cursor: responses[cursor],
    )

    assert list(first) == ["A", "B", "C"]
    assert first.total_kind == "estimate"


def test_an_offset_answer_to_a_cursor_request_is_an_error():
    first = SyncPage(
        _offset_data(1, 2, next_cursor="c1"),
        _Item,
        None,
        None,
        lambda _cursor: _offset_data(1, 2),
    )

    with pytest.raises(RuntimeError, match="cursor-mode"):
        list(first)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"total": 5}, "exact"),
        ({"total": 100_000}, "exact"),
        ({"total": 100_001}, "at_least"),
        ({"total": 250_000, "total_kind": "estimate"}, "estimate"),
        ({"total": 4, "total_kind": "at_most"}, "at_most"),
        ({"total": 7, "total_kind": "at_least"}, "at_least"),
        ({"total": 250_000, "total_kind": "exact"}, "exact"),
    ],
)
def test_total_kind_reads_the_field_or_infers_it_from_older_servers(
    data: dict[str, Any], expected: str
):
    page = SyncPage({"items": [], **data}, _Item)

    assert page.total_kind == expected


def _ingest_person(peer_id: str) -> dict[str, Any]:
    return {"peer_id": peer_id, "accounts": [], "messages": 1, "modeled": True}


@pytest.mark.asyncio
async def test_ingest_people_iteration_reads_on_by_cursor(
    client_fixture: tuple[Honcho, str], monkeypatch: pytest.MonkeyPatch
):
    """The ingest lists are GETs; iteration repeats them with `cursor` instead of `page`."""
    honcho_client, client_type = client_fixture
    queries: list[dict[str, Any]] = []

    def respond(path: str, query: dict[str, Any] | None) -> dict[str, Any]:
        assert path.endswith("/ingest/people")
        queries.append(dict(query or {}))
        if len(queries) == 1:
            return {
                "items": [_ingest_person("a"), _ingest_person("b")],
                "page": 1,
                "size": 2,
                "total": 3,
                "pages": 2,
                "total_kind": "exact",
                "next_cursor": "c1",
            }
        return {"items": [_ingest_person("c")], "next_page": None}

    if client_type == "async":

        async def aget(
            path: str, *, query: dict[str, Any] | None = None, **_: Any
        ) -> Any:
            return respond(path, query)

        monkeypatch.setattr(honcho_client._async_http, "get", aget)  # pyright: ignore[reportPrivateUsage]
        page = await honcho_client.aio.ingest_people(size=2)
        ids = [person.peer_id async for person in page]
    else:

        def get(path: str, *, query: dict[str, Any] | None = None, **_: Any) -> Any:
            return respond(path, query)

        monkeypatch.setattr(honcho_client._http, "get", get)  # pyright: ignore[reportPrivateUsage]
        ids = [person.peer_id for person in honcho_client.ingest_people(size=2)]

    assert ids == ["a", "b", "c"]
    assert queries == [{"page": 1, "size": 2}, {"cursor": "c1", "size": 2}]
