"""Resource creation telemetry: emitted once per inserted row, implicit paths included."""

from collections.abc import Iterator
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src.telemetry.events import BaseEvent

_EMITTING_MODULES = (
    "src.crud.workspace",
    "src.crud.peer",
    "src.crud.session",
    "src.crud.scope",
)


@pytest.fixture
def emitted() -> Iterator[list[BaseEvent]]:
    events: list[BaseEvent] = []
    with ExitStack() as stack:
        for module in _EMITTING_MODULES:
            stack.enter_context(patch(f"{module}.emit", side_effect=events.append))
        yield events


def _created(events: list[BaseEvent]) -> list[tuple[str, str]]:
    return sorted(
        (e.event_type(), e.get_resource_id())
        for e in events
        if e.event_type().endswith(".created")
    )


def test_implicit_creation_emits_once_per_row(
    client: TestClient, emitted: list[BaseEvent]
):
    """Posting messages and creating a session with scopes report every row they
    insert, including ones created by nested calls; repeats report nothing."""
    workspace = generate_nanoid()
    base = f"/v3/workspaces/{workspace}"

    response = client.post(
        f"{base}/sessions/s1/messages",
        json={
            "messages": [
                {"peer_id": "alice", "content": "hi"},
                {"peer_id": "bob", "content": "hey"},
            ]
        },
    )
    assert response.status_code in (200, 201), response.text
    assert _created(emitted) == [
        ("peer.created", f"{workspace}:alice"),
        ("peer.created", f"{workspace}:bob"),
        ("session.created", f"{workspace}:s1"),
        ("workspace.created", workspace),
    ]

    emitted.clear()
    body: dict[str, Any] = {
        "id": "s2",
        "peers": {"alice": {}, "carol": {}},
        "scopes": ["team"],
    }
    response = client.post(f"{base}/sessions", json=body)
    assert response.status_code == 201, response.text
    assert _created(emitted) == [
        ("peer.created", f"{workspace}:carol"),
        ("scope.created", f"{workspace}:team"),
        ("session.created", f"{workspace}:s2"),
    ]

    emitted.clear()
    response = client.post(f"{base}/sessions", json=body)
    assert response.status_code == 200, response.text
    assert _created(emitted) == []


def test_peer_route_and_clone_emit_created(
    client: TestClient, emitted: list[BaseEvent]
):
    """POST /peers into a new workspace reports the workspace too, and a clone
    reports the session it inserts."""
    workspace = generate_nanoid()
    base = f"/v3/workspaces/{workspace}"

    response = client.post(f"{base}/peers", json={"id": "gus"})
    assert response.status_code == 201, response.text
    assert _created(emitted) == [
        ("peer.created", f"{workspace}:gus"),
        ("workspace.created", workspace),
    ]

    response = client.post(f"{base}/sessions", json={"id": "s1"})
    assert response.status_code == 201, response.text
    emitted.clear()
    response = client.post(f"{base}/sessions/s1/clone")
    assert response.status_code == 201, response.text
    assert _created(emitted) == [
        ("session.created", f"{workspace}:{response.json()['id']}"),
    ]
