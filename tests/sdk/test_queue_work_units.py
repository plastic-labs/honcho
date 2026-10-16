"""Python cursor SDK contract, including an empty final page."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from sdks.python.src.honcho.api_types import QueueWorkUnit
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.pagination import AsyncCursorPage, SyncCursorPage


def item(ident: str) -> dict[str, Any]:
    return {
        "id": ident,
        "task_type": "dream",
        "workspace_id": "progress",
        "session_id": "session",
        "observer": "alice",
        "observers": [],
        "observed": "bob",
        "started_at": None,
        "stage": "deduction",
        "last_activity_at": None,
    }


def test_sync_cursor_preserves_scope_and_handles_empty_tail() -> None:
    client = Honcho(
        workspace_id="progress", api_key="test", base_url="http://localhost"
    )
    pages: list[dict[str, Any]] = [
        {"items": [item("a")], "next_page": "cursor-a"},
        {"items": [item("b")], "next_page": "cursor-b"},
        {"items": [], "next_page": None},
    ]
    with (
        patch.object(client, "_ensure_workspace"),
        patch.object(client._http, "get", side_effect=pages) as get,  # pyright: ignore[reportPrivateUsage]
    ):
        page = client.queue_work_units(
            task_type="dream", observer="alice", sender="bob", session="session", size=1
        )
        assert isinstance(page, SyncCursorPage)
        assert [row.id for row in page] == ["a", "b"]
        assert isinstance(page.items[0], QueueWorkUnit)
        assert page.items[0].started_at is None
        queries = [call.kwargs["query"] for call in get.call_args_list]
        assert [q["cursor"] for q in queries] == ["", "cursor-a", "cursor-b"]
        assert all(
            q
            == {
                "task_type": "dream",
                "observer_id": "alice",
                "sender_id": "bob",
                "session_id": "session",
                "size": 1,
                "cursor": q["cursor"],
            }
            for q in queries
        )


@pytest.mark.asyncio
async def test_async_cursor_preserves_scope_and_stops_at_empty_tail() -> None:
    client = Honcho(
        workspace_id="progress", api_key="test", base_url="http://localhost"
    )
    pages: list[dict[str, Any]] = [
        {"items": [item("a")], "next_page": "next"},
        {"items": [], "next_page": None},
    ]
    with (
        patch.object(client, "_ensure_workspace_async", new_callable=AsyncMock),
        patch.object(
            client._async_http_client,  # pyright: ignore[reportPrivateUsage]
            "get",
            new_callable=AsyncMock,
            side_effect=pages,
        ) as get,
    ):
        page = await client.aio.queue_work_units(
            task_type="dream", observer="alice", size=1
        )
        assert isinstance(page, AsyncCursorPage)
        assert [row.id async for row in page] == ["a"]
        assert [call.kwargs["query"]["cursor"] for call in get.call_args_list] == [
            "",
            "next",
        ]
        assert all(
            call.kwargs["query"]["observer_id"] == "alice"
            for call in get.call_args_list
        )


@pytest.mark.parametrize("size", [0, 101])
def test_page_size_validation(size: int) -> None:
    client = Honcho(
        workspace_id="progress", api_key="test", base_url="http://localhost"
    )
    with pytest.raises(ValueError):
        client.queue_work_units(size=size)
