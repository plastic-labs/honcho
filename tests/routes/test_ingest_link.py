"""Linking imported accounts into one person, applied on the next import."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models

ALEX = {"id": "u1", "name": "Alex"}
ALEX_GITHUB = {"id": "g1", "name": "alexdev"}
SAM = {"id": "u2", "name": "Sam"}


def _workspace(client: TestClient, *, auto: bool = True) -> str:
    name = f"ws-{generate_nanoid(size=10).replace('_', 'a')}"
    response = client.post("/v3/workspaces", json={"id": name, "auto": auto})
    assert response.status_code in (200, 201), response.text
    return name


def _record(n: int, author: dict[str, Any], source: str = "discord") -> dict[str, Any]:
    return {
        "source": source,
        "source_instance": "acme",
        "source_id": f"{source}-{n}",
        "created_at": f"2026-11-04T10:{n:02d}:00Z",
        "container": {"id": f"{source}-chan", "name": f"{source} channel"},
        "author": author,
        "content": f"message {n}",
    }


RECORDS = [
    _record(1, ALEX),
    _record(2, SAM),
    _record(3, ALEX),
    _record(4, ALEX_GITHUB, "github"),
    _record(5, ALEX_GITHUB, "github"),
]


def _import(client: TestClient, workspace: str) -> dict[int, dict[str, Any]]:
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"derive": "later", "records": RECORDS},
    )
    assert response.status_code == 200, response.text
    return {r["index"]: r for r in response.json()["records"]}


def _people(client: TestClient, workspace: str) -> dict[str, dict[str, Any]]:
    response = client.get(f"/v3/workspaces/{workspace}/ingest/people")
    assert response.status_code == 200, response.text
    return {p["display_name"]: p for p in response.json()["items"]}


def _link(client: TestClient, workspace: str, *peers: str):
    return client.post(
        f"/v3/workspaces/{workspace}/ingest/link", json={"peers": list(peers)}
    )


async def _reset(client: TestClient, db: AsyncSession, workspace: str) -> list[str]:
    """Reset, then run the deletions the deriver would run."""
    response = client.post(f"/v3/workspaces/{workspace}/ingest/reset")
    assert response.status_code == 202, response.text
    sessions: list[str] = response.json()["sessions"]
    for name in sessions:
        await crud.delete_session(db, workspace, name)
    await db.commit()
    return sessions


@pytest.mark.asyncio
async def test_reset_link_and_import_put_both_accounts_on_one_person(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    first = _import(client, workspace)
    people = _people(client, workspace)
    alex, alexdev = people["Alex"]["peer_id"], people["alexdev"]["peer_id"]

    # Already imported: nothing moves, so it's refused until a reset.
    refused = _link(client, workspace, alex, alexdev)
    assert refused.status_code == 409, refused.text
    assert "reset" in refused.json()["detail"].lower()

    sessions = await _reset(client, db_session, workspace)
    assert len(sessions) == 2
    status = client.get(f"/v3/workspaces/{workspace}/ingest/status").json()
    assert status["sessions_being_deleted"] == 0
    linked = _link(client, workspace, alex, alexdev)
    assert linked.status_code == 200, linked.text
    assert linked.json()["merged"] == [alexdev]
    # The person's own account first: it names the person.
    assert [(a["source"], a["id"]) for a in linked.json()["accounts"]] == [
        ("discord", "u1"),
        ("github", "g1"),
    ]

    again = _import(client, workspace)

    assert {again[i]["peer_id"] for i in (0, 2, 3, 4)} == {alex}
    assert again[1]["peer_id"] == first[1]["peer_id"]  # Sam is untouched
    people = _people(client, workspace)
    assert "alexdev" not in people
    assert people["Alex"]["messages"] == 4
    assert len(people["Alex"]["accounts"]) == 2


@pytest.mark.asyncio
async def test_unlink_puts_the_account_back_on_its_own_person(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    people = _people(client, workspace)
    alex, alexdev = people["Alex"]["peer_id"], people["alexdev"]["peer_id"]
    await _reset(client, db_session, workspace)
    assert _link(client, workspace, alex, alexdev).status_code == 200
    _import(client, workspace)
    account = {"source": "github", "source_instance": "acme", "id": "g1"}

    refused = client.post(
        f"/v3/workspaces/{workspace}/ingest/unlink",
        json={"peer": alex, "account": account},
    )
    assert refused.status_code == 409, refused.text

    await _reset(client, db_session, workspace)
    unlinked = client.post(
        f"/v3/workspaces/{workspace}/ingest/unlink",
        json={"peer": alex, "account": account},
    )
    assert unlinked.status_code == 200, unlinked.text
    assert unlinked.json()["peer_id"] == alexdev

    again = _import(client, workspace)
    assert {again[i]["peer_id"] for i in (3, 4)} == {alexdev}
    assert {again[i]["peer_id"] for i in (0, 2)} == {alex}


def test_an_own_account_cant_be_unlinked(client: TestClient) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    alex = _people(client, workspace)["Alex"]["peer_id"]

    response = client.post(
        f"/v3/workspaces/{workspace}/ingest/unlink",
        json={
            "peer": alex,
            "account": {"source": "discord", "source_instance": "acme", "id": "u1"},
        },
    )

    assert response.status_code == 409, response.text


@pytest.mark.asyncio
async def test_only_imported_people_can_be_merged(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    alex = _people(client, workspace)["Alex"]["peer_id"]
    await _reset(client, db_session, workspace)
    assert client.post(
        f"/v3/workspaces/{workspace}/peers", json={"id": "hand-made"}
    ).status_code in (200, 201)

    explicit = _link(client, workspace, alex, "hand-made")
    same = _link(client, workspace, alex, alex)
    missing = _link(client, workspace, alex, "nobody")

    assert explicit.status_code == 409, explicit.text
    assert same.status_code in (409, 422), same.text
    assert missing.status_code == 404, missing.text
    # An explicit peer can be the one kept.
    alexdev = _people(client, workspace)["alexdev"]["peer_id"]
    assert _link(client, workspace, "hand-made", alexdev).status_code == 200


@pytest.mark.asyncio
async def test_a_merged_person_cant_be_linked_into(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    people = _people(client, workspace)
    alex, alexdev, sam = (people[n]["peer_id"] for n in ("Alex", "alexdev", "Sam"))
    await _reset(client, db_session, workspace)
    assert _link(client, workspace, alex, alexdev).status_code == 200

    response = _link(client, workspace, alexdev, sam)

    assert response.status_code == 409, response.text


@pytest.mark.asyncio
async def test_derived_state_blocks_a_link(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    people = _people(client, workspace)
    alex, alexdev = people["Alex"]["peer_id"], people["alexdev"]["peer_id"]
    await _reset(client, db_session, workspace)
    db_session.add(
        models.Collection(workspace_name=workspace, observer=alex, observed=alexdev)
    )
    await db_session.commit()

    response = _link(client, workspace, alex, alexdev)

    assert response.status_code == 409, response.text
    assert "derived" in response.json()["detail"]


@pytest.mark.asyncio
async def test_status_drops_a_reset_import_at_once(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)

    response = client.post(f"/v3/workspaces/{workspace}/ingest/reset")
    assert response.status_code == 202, response.text
    status = client.get(f"/v3/workspaces/{workspace}/ingest/status").json()

    # Deletion hasn't run yet; the sessions are already inactive.
    assert status["records"] == {"deferred": 0, "pending": 0, "queued": 0}
    assert status["sessions"] == 0
    assert status["sessions_being_deleted"] == 2
    still_there = await db_session.scalar(
        select(models.IngestRecord.id).where(
            models.IngestRecord.workspace_name == workspace
        )
    )
    assert still_there is not None


def test_link_needs_an_auto_workspace(client: TestClient) -> None:
    regular = _workspace(client, auto=False)
    assert _link(client, regular, "a", "b").status_code == 409
    assert client.post(f"/v3/workspaces/{regular}/ingest/reset").status_code == 409


@pytest.mark.asyncio
async def test_records_cant_be_sent_to_a_merged_person(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    people = _people(client, workspace)
    alex, alexdev = people["Alex"]["peer_id"], people["alexdev"]["peer_id"]
    await _reset(client, db_session, workspace)
    assert _link(client, workspace, alex, alexdev).status_code == 200

    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"records": [{**_record(9, ALEX_GITHUB, "github"), "peer_id": alexdev}]},
    )

    assert response.status_code == 409, response.text


@pytest.mark.asyncio
async def test_a_chained_link_unlinks_back_to_where_the_account_came_from(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    people = _people(client, workspace)
    alex, alexdev, sam = (people[n]["peer_id"] for n in ("Alex", "alexdev", "Sam"))
    await _reset(client, db_session, workspace)
    assert _link(client, workspace, alex, alexdev).status_code == 200
    assert _link(client, workspace, sam, alex).status_code == 200

    unlinked = client.post(
        f"/v3/workspaces/{workspace}/ingest/unlink",
        json={
            "peer": sam,
            "account": {"source": "github", "source_instance": "acme", "id": "g1"},
        },
    )

    assert unlinked.status_code == 200, unlinked.text
    assert unlinked.json()["peer_id"] == alexdev


@pytest.mark.asyncio
async def test_an_account_without_an_id_cant_be_linked(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={
            "derive": "later",
            "records": [_record(1, ALEX), _record(2, {"name": "Anonymous"})],
        },
    )
    assert response.status_code == 200, response.text
    people = _people(client, workspace)
    await _reset(client, db_session, workspace)

    linked = _link(
        client, workspace, people["Alex"]["peer_id"], people["Anonymous"]["peer_id"]
    )

    assert linked.status_code == 409, linked.text
    assert "without an ID" in linked.json()["detail"]


@pytest.mark.asyncio
async def test_a_tombstone_doesnt_stop_an_unlink_after_a_reset(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    people = _people(client, workspace)
    alex, alexdev = people["Alex"]["peer_id"], people["alexdev"]["peer_id"]
    await _reset(client, db_session, workspace)
    assert _link(client, workspace, alex, alexdev).status_code == 200
    _import(client, workspace)
    # The source deleted one of the linked account's records, in the dry run.
    deleted = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={
            "derive": "later",
            "records": [
                {
                    "op": "delete",
                    "source": "github",
                    "source_instance": "acme",
                    "source_id": "github-4",
                }
            ],
        },
    )
    assert deleted.json()["records"][0]["outcome"] == "deleted"

    await _reset(client, db_session, workspace)
    unlinked = client.post(
        f"/v3/workspaces/{workspace}/ingest/unlink",
        json={
            "peer": alex,
            "account": {"source": "github", "source_instance": "acme", "id": "g1"},
        },
    )
    assert unlinked.status_code == 200, unlinked.text
