"""Reviewing an import before deriving it: the ingest status, people and sessions."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings


def _workspace(client: TestClient, *, auto: bool = True) -> str:
    name = f"ws-{generate_nanoid(size=10).replace('_', 'a')}"
    response = client.post("/v3/workspaces", json={"id": name, "auto": auto})
    assert response.status_code in (200, 201), response.text
    return name


def _record(
    n: int, author: dict[str, Any], channel: str, **extra: Any
) -> dict[str, Any]:
    return {
        "source": "discord",
        "source_instance": "guild1",
        "source_id": f"msg-{n}",
        "created_at": f"2026-11-04T10:{n:02d}:00Z",
        "container": {"id": channel, "name": f"#{channel}"},
        "author": author,
        "content": f"message {n}",
        **extra,
    }


JESS = {"id": "u1", "name": "Jessica"}
MAYA = {"id": "u2", "name": "Maya"}
SENTRY = {"id": "u9", "name": "Sentry", "bot": True}


def _import(client: TestClient, workspace: str) -> None:
    """Jessica writes 3, Maya 2, Sentry 1; #dev holds 4, #random 2. Deferred."""
    records = [
        _record(1, JESS, "dev"),
        _record(2, JESS, "dev"),
        _record(3, MAYA, "dev", participants=[JESS]),
        _record(4, SENTRY, "dev"),
        _record(5, JESS, "random"),
        _record(6, MAYA, "random"),
    ]
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"derive": "later", "records": records},
    )
    assert response.status_code == 200, response.text


def _people(client: TestClient, workspace: str, **params: Any) -> dict[str, Any]:
    response = client.get(f"/v3/workspaces/{workspace}/ingest/people", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _sessions(client: TestClient, workspace: str) -> list[dict[str, Any]]:
    response = client.get(f"/v3/workspaces/{workspace}/ingest/sessions")
    assert response.status_code == 200, response.text
    return response.json()["items"]


def _status(client: TestClient, workspace: str) -> dict[str, Any]:
    response = client.get(f"/v3/workspaces/{workspace}/ingest/status")
    assert response.status_code == 200, response.text
    return response.json()


def test_status_counts_an_import_waiting_to_be_derived(client: TestClient) -> None:
    workspace = _workspace(client)
    _import(client, workspace)

    status = _status(client, workspace)

    assert status["records"] == {"deferred": 6, "pending": 0, "queued": 0}
    assert status["deferred_tokens"] > 0
    assert (status["people"], status["people_not_modeled"]) == (3, 1)
    assert (status["sessions"], status["sessions_not_modeled"]) == (2, 0)


def test_people_list_each_peer_with_its_accounts_most_records_first(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)

    people = _people(client, workspace)["items"]

    assert [(p["display_name"], p["messages"], p["modeled"]) for p in people] == [
        ("Jessica", 3, True),
        ("Maya", 2, True),
        ("Sentry", 1, False),
    ]
    assert people[0]["accounts"] == [
        {
            "source": "discord",
            "source_instance": "guild1",
            "id": "u1",
            "display_name": "Jessica",
        }
    ]
    assert people[0]["peer_id"] != "Jessica"


def test_people_page_in_both_pagination_modes(client: TestClient) -> None:
    workspace = _workspace(client)
    _import(client, workspace)

    offset = [p["display_name"] for p in _people(client, workspace, size=2)["items"]]
    offset += [
        p["display_name"] for p in _people(client, workspace, size=2, page=2)["items"]
    ]
    first = _people(client, workspace, size=2, cursor="")
    cursor = [p["display_name"] for p in first["items"]]
    cursor += [
        p["display_name"]
        for p in _people(client, workspace, size=2, cursor=first["next_page"])["items"]
    ]

    assert offset == cursor == ["Jessica", "Maya", "Sentry"]


def test_sessions_list_each_container_most_records_first(client: TestClient) -> None:
    workspace = _workspace(client)
    _import(client, workspace)

    sessions = _sessions(client, workspace)

    assert [(s["display_name"], s["messages"], s["modeled"]) for s in sessions] == [
        ("#dev", 4, True),
        ("#random", 2, True),
    ]
    assert {(s["source"], s["source_instance"]) for s in sessions} == {
        ("discord", "guild1")
    }


def test_status_shows_edits_made_before_release(client: TestClient) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    people = {
        p["display_name"]: p["peer_id"] for p in _people(client, workspace)["items"]
    }
    sessions = {
        s["display_name"]: s["session_id"] for s in _sessions(client, workspace)
    }

    assert (
        client.put(
            f"/v3/workspaces/{workspace}/peers/{people['Maya']}",
            json={"configuration": {"observe_me": False}},
        ).status_code
        == 200
    )
    assert (
        client.put(
            f"/v3/workspaces/{workspace}/sessions/{sessions['#random']}",
            json={"configuration": {"reasoning": {"enabled": False}}},
        ).status_code
        == 200
    )

    status = _status(client, workspace)
    assert (status["people_not_modeled"], status["sessions_not_modeled"]) == (2, 1)
    modeled = {
        p["display_name"]: p["modeled"] for p in _people(client, workspace)["items"]
    }
    assert modeled == {"Jessica": True, "Maya": False, "Sentry": False}


def test_a_workspace_without_reasoning_models_no_session(client: TestClient) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    assert (
        client.put(
            f"/v3/workspaces/{workspace}",
            json={"configuration": {"reasoning": {"enabled": False}}},
        ).status_code
        == 200
    )

    assert {s["modeled"] for s in _sessions(client, workspace)} == {False}


@pytest.mark.asyncio
async def test_edits_before_release_change_what_is_derived(
    client: TestClient, db_session: AsyncSession
) -> None:
    """An excluded person is observed by no one; an excluded channel gets summaries only."""
    workspace = _workspace(client)
    _import(client, workspace)
    people = {
        p["display_name"]: p["peer_id"] for p in _people(client, workspace)["items"]
    }
    sessions = {
        s["display_name"]: s["session_id"] for s in _sessions(client, workspace)
    }
    client.put(
        f"/v3/workspaces/{workspace}/peers/{people['Maya']}",
        json={"configuration": {"observe_me": False}},
    )
    client.put(
        f"/v3/workspaces/{workspace}/sessions/{sessions['#random']}",
        json={"configuration": {"reasoning": {"enabled": False}}},
    )

    released = client.post(f"/v3/workspaces/{workspace}/ingest/derive").json()
    assert released == {"pending": 6, "remaining": 0}

    rows = (
        await db_session.execute(
            select(models.QueueItem.session_id, models.QueueItem.payload).where(
                models.QueueItem.workspace_name == workspace,
                models.QueueItem.task_type == "representation",
            )
        )
    ).all()
    observed = {payload["observed"] for _, payload in rows}
    session_ids = await db_session.scalars(
        select(models.Session.id).where(
            models.Session.workspace_name == workspace,
            models.Session.name == sessions["#random"],
        )
    )
    random_id = session_ids.one()

    assert observed == {people["Jessica"]}
    assert random_id not in {session_id for session_id, _ in rows}
    assert _status(client, workspace)["records"]["deferred"] == 0


def test_status_needs_an_auto_workspace(client: TestClient) -> None:
    regular = _workspace(client, auto=False)
    for path in ("status", "people", "sessions"):
        assert client.get(f"/v3/workspaces/{regular}/ingest/{path}").status_code == 409
        assert client.get(f"/v3/workspaces/missing-ws/ingest/{path}").status_code == 404


def test_an_empty_auto_workspace_has_an_empty_status(client: TestClient) -> None:
    workspace = _workspace(client)

    assert _status(client, workspace) == {
        "records": {"deferred": 0, "pending": 0, "queued": 0},
        "deferred_tokens": 0,
        "people": 0,
        "people_not_modeled": 0,
        "sessions": 0,
        "sessions_not_modeled": 0,
        "sessions_being_deleted": 0,
        "suggestions": {"rules": {"high": 0, "low": 0}, "refreshing": False},
    }
    assert _people(client, workspace)["items"] == []


def test_records_with_explicit_targets_are_reviewed_too(client: TestClient) -> None:
    workspace = _workspace(client)
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={
            "derive": "later",
            "records": [
                _record(1, JESS, "dev", peer_id="alice", session_id="planning"),
                _record(2, JESS, "dev", peer_id="alice", session_id="planning"),
            ],
        },
    )
    assert response.status_code == 200, response.text

    people = {p["peer_id"]: p for p in _people(client, workspace)["items"]}
    sessions = {s["session_id"]: s for s in _sessions(client, workspace)}

    assert people["alice"]["messages"] == 2
    assert people["alice"]["accounts"] == []
    assert sessions["planning"]["messages"] == 2
    assert (
        sessions["planning"]["source"],
        sessions["planning"]["source_instance"],
    ) == (
        "discord",
        "guild1",
    )


@pytest.mark.asyncio
async def test_an_untrusted_binding_lends_no_accounts(
    client: TestClient, db_session: AsyncSession
) -> None:
    """A binding pointing at a peer it didn't create, unconfirmed, is stale."""
    workspace = _workspace(client)
    _import(client, workspace)
    assert client.post(
        f"/v3/workspaces/{workspace}/peers", json={"id": "someone-else"}
    ).status_code in (200, 201)
    await db_session.execute(
        update(models.IngestPeerBinding)
        .where(
            models.IngestPeerBinding.workspace_name == workspace,
            models.IngestPeerBinding.key_value == "u2",
        )
        .values(peer_name="someone-else")
    )
    await db_session.commit()

    people = {p["peer_id"]: p for p in _people(client, workspace)["items"]}

    assert "someone-else" not in people
    # Maya's records still count under the peer that wrote them.
    maya = next(p for p in people.values() if p["messages"] == 2)
    assert maya["accounts"] == []


@pytest.mark.asyncio
async def test_modeled_follows_how_the_deriver_resolves_reasoning(
    client: TestClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    sessions = {
        s["display_name"]: s["session_id"] for s in _sessions(client, workspace)
    }
    # The legacy key, still honored by the deriver's configuration resolver.
    await db_session.execute(
        update(models.Session)
        .where(
            models.Session.workspace_name == workspace,
            models.Session.name == sessions["#random"],
        )
        .values(configuration={"deriver": {"enabled": False}})
    )
    await db_session.commit()

    modeled = {s["display_name"]: s["modeled"] for s in _sessions(client, workspace)}
    assert modeled == {"#dev": True, "#random": False}

    monkeypatch.setattr(settings.DERIVER, "ENABLED", False)
    assert {s["modeled"] for s in _sessions(client, workspace)} == {False}
    assert _status(client, workspace)["sessions_not_modeled"] == 2
