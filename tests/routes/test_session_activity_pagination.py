"""Activity ordering stays total across offset, cached seeks, and cursor pages."""

import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from fastapi_pagination.cursor import encode_cursor
from nanoid import generate as generate_nanoid
from sqlakeyset import Marker, serialize_bookmark
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.models import Peer, Workspace


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["plain", "shim", "cursor"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("shape", ["mixed", "all_null", "all_active", "empty"])
async def test_activity_pages_keep_nulls_and_timestamp_ties(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    reverse: bool,
    size: int,
    shape: str,
):
    workspace, _ = sample_data
    group = str(generate_nanoid())
    old = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    recent = datetime.datetime(2024, 1, 2, tzinfo=datetime.UTC)
    timestamps = (
        []
        if shape == "empty"
        else [None] * 7
        if shape == "all_null"
        else [recent, old, recent, old, recent, old, old]
        if shape == "all_active"
        else [None, recent, old, None, recent, old, None]
    )
    rows = [
        models.Session(
            id=f"{group[:20]}{i}",
            name=f"activity-{group}-{i}",
            workspace_name=workspace.name,
            last_message_at=timestamp,
            h_metadata={"activity_group": group},
        )
        for i, timestamp in enumerate(timestamps)
    ]
    db_session.add_all(rows)
    await db_session.commit()
    expected = sorted(
        (row for row in rows if row.last_message_at is not None),
        key=lambda row: (row.last_message_at, row.id),
        reverse=reverse,
    ) + sorted(
        (row for row in rows if row.last_message_at is None),
        key=lambda row: row.id,
        reverse=reverse,
    )
    monkeypatch.setattr(settings.CACHE, "ENABLED", True)
    monkeypatch.setattr(settings.CACHE, "PAGINATION_OFFSET_SHIM", mode != "plain")
    url = f"/v3/workspaces/{workspace.name}/sessions/list"
    params: dict[str, Any] = {
        "size": size,
        "sort_by": "last_message_at",
        "reverse": reverse,
    }
    body = {"filters": {"metadata": {"activity_group": group}}}

    def get(extra: dict[str, Any]) -> dict[str, Any]:
        response = client.post(url, params={**params, **extra}, json=body)
        assert response.status_code == 200, response.text
        return response.json()

    pages: list[dict[str, Any]] = []
    if mode == "cursor":
        cursor: str | None = ""
        while cursor is not None:
            data = get({"cursor": cursor})
            assert data["total"] is None
            pages.append(data)
            cursor = data["next_page"]
            assert len(pages) <= len(rows) + 1, "cursor walk did not terminate"
        for index in range(1, len(pages)):
            previous = get({"cursor": pages[index]["previous_page"]})
            assert previous["items"] == pages[index - 1]["items"]
        for data in pages:
            assert get({"cursor": data["current_page"]})["items"] == data["items"]
            backwards = get({"cursor": data["current_page_backwards"]})["items"]
            # Like the upstream paginator, an end-of-list backwards cursor
            # fills the page even when its forward counterpart was partial.
            assert backwards[-len(data["items"]) :] == data["items"]
    else:
        # A cache miss on a random page still has the same order as a walk.
        random_page = get({"page": 3})
        assert [item["id"] for item in random_page["items"]] == [
            row.name for row in expected[2 * size : 3 * size]
        ]
        page = 1
        while True:
            data = get({"page": page})
            assert data["total"] == len(rows)
            pages.append(data)
            if page >= data["pages"]:
                break
            page += 1
            assert page <= len(rows) + 1, "offset walk did not terminate"

    assert [item["id"] for data in pages for item in data["items"]] == [
        row.name for row in expected
    ]


@pytest.mark.parametrize(
    "place",
    [
        (1, "a" * 21),
        (datetime.datetime(2024, 1, 1), "a" * 21),
        (None, 1),
        (None, "short"),
        (None, "a" * 21, 1),
    ],
)
def test_activity_cursor_rejects_invalid_key_types(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    place: tuple[object, ...],
):
    workspace, _ = sample_data
    cursor = encode_cursor(serialize_bookmark(Marker(place)))
    response = client.post(
        f"/v3/workspaces/{workspace.name}/sessions/list",
        params={"sort_by": "last_message_at", "cursor": cursor},
    )
    assert response.status_code == 400, response.text
