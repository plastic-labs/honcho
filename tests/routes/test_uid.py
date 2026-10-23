"""Every route that returns a workspace, peer or session exposes a stable ``uid``.

``id`` is the client-chosen name; ``uid`` is the server-assigned primary key. These
tests pin that ``uid`` is present on each response shape, matches the database row,
and stays the same across get-or-create, update and list calls.
"""

from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


def _list_items(response_json: dict[str, object]) -> list[dict[str, object]]:
    items = response_json["items"]
    assert isinstance(items, list)
    return items  # pyright: ignore[reportUnknownVariableType]


async def test_workspace_uid_is_stable(client: TestClient, db_session: AsyncSession):
    name = str(generate_nanoid())

    created = client.post("/v3/workspaces", json={"id": name})
    assert created.status_code in (200, 201)
    uid = created.json()["uid"]

    row = await db_session.scalar(
        select(models.Workspace).where(models.Workspace.name == name)
    )
    assert row is not None
    assert uid == row.id
    assert uid != name

    fetched = client.post("/v3/workspaces", json={"id": name})
    assert fetched.json()["uid"] == uid

    updated = client.put(f"/v3/workspaces/{name}", json={"metadata": {"k": "v"}})
    assert updated.status_code == 200
    assert updated.json()["uid"] == uid

    listed = client.post("/v3/workspaces/list", json={"filters": {"id": name}})
    assert listed.status_code == 200
    assert [w["uid"] for w in _list_items(listed.json())] == [uid]


async def test_peer_uid_is_stable(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    workspace, _ = sample_data
    name = str(generate_nanoid())
    base = f"/v3/workspaces/{workspace.name}/peers"

    created = client.post(base, json={"id": name})
    assert created.status_code in (200, 201)
    uid = created.json()["uid"]

    row = await db_session.scalar(
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace.name)
        .where(models.Peer.name == name)
    )
    assert row is not None
    assert uid == row.id

    # Second get-or-create is served from the cache path; uid must survive it.
    fetched = client.post(base, json={"id": name})
    assert fetched.json()["uid"] == uid

    updated = client.put(f"{base}/{name}", json={"metadata": {"k": "v"}})
    assert updated.status_code == 200
    assert updated.json()["uid"] == uid

    listed = client.post(f"{base}/list", json={"filters": {"id": name}})
    assert listed.status_code == 200
    assert [p["uid"] for p in _list_items(listed.json())] == [uid]


async def test_session_uid_is_stable(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    workspace, peer = sample_data
    name = str(generate_nanoid())
    base = f"/v3/workspaces/{workspace.name}/sessions"

    created = client.post(base, json={"id": name, "peers": {peer.name: {}}})
    assert created.status_code in (200, 201)
    uid = created.json()["uid"]

    row = await db_session.scalar(
        select(models.Session)
        .where(models.Session.workspace_name == workspace.name)
        .where(models.Session.name == name)
    )
    assert row is not None
    assert uid == row.id

    fetched = client.post(base, json={"id": name})
    assert fetched.json()["uid"] == uid

    updated = client.put(f"{base}/{name}", json={"metadata": {"k": "v"}})
    assert updated.status_code == 200
    assert updated.json()["uid"] == uid

    listed = client.post(f"{base}/list", json={"filters": {"id": name}})
    assert listed.status_code == 200
    assert [s["uid"] for s in _list_items(listed.json())] == [uid]

    peer_sessions = client.post(
        f"/v3/workspaces/{workspace.name}/peers/{peer.name}/sessions", json={}
    )
    assert peer_sessions.status_code == 200
    assert uid in [s["uid"] for s in _list_items(peer_sessions.json())]

    other = client.post(
        f"/v3/workspaces/{workspace.name}/peers", json={"id": str(generate_nanoid())}
    ).json()
    added = client.post(f"{base}/{name}/peers", json={other["id"]: {}})
    assert added.status_code == 200
    assert added.json()["uid"] == uid

    session_peers = client.get(f"{base}/{name}/peers")
    assert session_peers.status_code == 200
    assert other["uid"] in [p["uid"] for p in _list_items(session_peers.json())]


async def test_cloned_session_gets_its_own_uid(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    workspace, _ = sample_data
    name = str(generate_nanoid())
    base = f"/v3/workspaces/{workspace.name}/sessions"

    original = client.post(base, json={"id": name}).json()
    clone = client.post(f"{base}/{name}/clone")
    assert clone.status_code == 201
    data = clone.json()
    assert data["uid"]
    assert data["uid"] != original["uid"]
    assert data["id"] != original["id"]
