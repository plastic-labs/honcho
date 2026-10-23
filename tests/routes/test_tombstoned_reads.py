"""Tombstoned workspaces, peers and sessions are hidden from reads.

Nothing sets ``deleted_at`` yet, so these tests set it directly.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.crud.session import is_peer_in_session

Target = Literal["workspace", "peer", "session"]


@dataclass
class Graph:
    workspace: str
    author: str
    bystander: str
    session: str
    other_session: str


async def _seed(db: AsyncSession) -> Graph:
    graph = Graph(
        workspace=str(generate_nanoid()),
        author="author",
        bystander="bystander",
        session="session",
        other_session="other-session",
    )
    db.add(models.Workspace(name=graph.workspace))
    await db.flush()
    db.add_all(
        [
            models.Peer(name=name, workspace_name=graph.workspace)
            for name in ("author", "bystander")
        ]
        + [
            models.Session(name=name, workspace_name=graph.workspace)
            for name in (graph.session, graph.other_session)
        ]
    )
    await db.flush()
    db.add_all(
        [
            models.SessionPeer(
                workspace_name=graph.workspace, session_name=session, peer_name=peer
            )
            for session in (graph.session, graph.other_session)
            for peer in (graph.author, graph.bystander)
        ]
    )
    db.add_all(
        [
            models.Message(
                workspace_name=graph.workspace,
                session_name=session,
                peer_name=graph.author,
                content="hello",
                seq_in_session=1,
            )
            for session in (graph.session, graph.other_session)
        ]
    )
    await db.commit()
    return graph


async def _tombstone(db: AsyncSession, graph: Graph, target: Target) -> None:
    model, name = {
        "workspace": (models.Workspace, graph.workspace),
        "peer": (models.Peer, graph.author),
        "session": (models.Session, graph.session),
    }[target]
    stmt = update(model).where(model.name == name).values(deleted_at=datetime.now(UTC))
    if target != "workspace":
        stmt = stmt.where(model.workspace_name == graph.workspace)
    await db.execute(stmt)
    await db.commit()


def _ids(client: TestClient, method: str, path: str) -> set[str]:
    response = client.request(method, path, json={})
    assert response.status_code == 200, response.text
    return {item["id"] for item in response.json()["items"]}


async def test_tombstoned_workspace_hides_itself_and_its_children(
    client: TestClient, db_session: AsyncSession
):
    graph = await _seed(db_session)
    other = await _seed(db_session)
    await _tombstone(db_session, graph, "workspace")
    base = f"/v3/workspaces/{graph.workspace}"

    listed = _ids(client, "POST", "/v3/workspaces/list")
    assert graph.workspace not in listed
    assert other.workspace in listed
    assert _ids(client, "POST", f"{base}/peers/list") == set()
    assert _ids(client, "POST", f"{base}/sessions/list") == set()
    assert _ids(client, "GET", f"{base}/sessions/{graph.session}/peers") == set()
    assert _ids(client, "POST", f"{base}/peers/{graph.author}/sessions") == set()

    # The tombstone holds the name: get-or-create conflicts instead of
    # returning or recreating it.
    for response in (
        client.post("/v3/workspaces", json={"name": graph.workspace}),
        client.put(base, json={"metadata": {"k": "v"}}),
        client.post(f"{base}/peers", json={"name": "new-peer"}),
        client.post(f"{base}/sessions", json={"name": "new-session"}),
    ):
        assert response.status_code == 409, response.text

    other_base = f"/v3/workspaces/{other.workspace}"
    assert _ids(client, "POST", f"{other_base}/peers/list") == {"author", "bystander"}


async def test_tombstoned_peer_is_hidden_from_peer_and_membership_reads(
    client: TestClient, db_session: AsyncSession
):
    graph = await _seed(db_session)
    await _tombstone(db_session, graph, "peer")
    base = f"/v3/workspaces/{graph.workspace}"

    assert _ids(client, "POST", f"{base}/peers/list") == {graph.bystander}
    assert _ids(client, "GET", f"{base}/sessions/{graph.session}/peers") == {
        graph.bystander
    }
    assert _ids(client, "POST", f"{base}/peers/{graph.author}/sessions") == set()
    assert _ids(client, "POST", f"{base}/peers/{graph.bystander}/sessions") == {
        graph.session,
        graph.other_session,
    }
    # Sessions outlive a deleted member.
    assert _ids(client, "POST", f"{base}/sessions/list") == {
        graph.session,
        graph.other_session,
    }

    response = client.get(
        f"{base}/sessions/{graph.session}/peers/{graph.author}/config"
    )
    assert response.status_code == 404, response.text
    for response in (
        client.post(f"{base}/peers", json={"name": graph.author}),
        client.put(f"{base}/peers/{graph.author}", json={"metadata": {"k": "v"}}),
    ):
        assert response.status_code == 409, response.text


async def test_tombstoned_session_is_hidden_from_session_reads(
    client: TestClient, db_session: AsyncSession
):
    graph = await _seed(db_session)
    await _tombstone(db_session, graph, "session")
    base = f"/v3/workspaces/{graph.workspace}"

    assert _ids(client, "POST", f"{base}/sessions/list") == {graph.other_session}
    assert _ids(client, "POST", f"{base}/peers/{graph.author}/sessions") == {
        graph.other_session
    }
    assert _ids(client, "GET", f"{base}/sessions/{graph.session}/peers") == set()
    # Members outlive a deleted session.
    assert _ids(client, "POST", f"{base}/peers/list") == {
        graph.author,
        graph.bystander,
    }

    response = client.get(
        f"{base}/sessions/{graph.session}/peers/{graph.author}/config"
    )
    assert response.status_code == 404, response.text
    for response in (
        client.post(f"{base}/sessions", json={"name": graph.session}),
        client.put(f"{base}/sessions/{graph.session}", json={"metadata": {"k": "v"}}),
    ):
        assert response.status_code == 409, response.text


@pytest.mark.parametrize(
    ("target", "member"),
    [("workspace", False), ("peer", False), ("session", False), (None, True)],
)
async def test_membership_check_requires_live_session_and_peer(
    db_session: AsyncSession, target: Target | None, member: bool
):
    graph = await _seed(db_session)
    if target is not None:
        await _tombstone(db_session, graph, target)
    assert (
        await is_peer_in_session(
            db_session, graph.workspace, graph.session, graph.author
        )
        is member
    )


@pytest.mark.parametrize("target", [None, "peer", "session"])
async def test_workspace_stats_and_active_peers_skip_tombstones(
    db_session: AsyncSession, target: Target | None
):
    graph = await _seed(db_session)
    if target is not None:
        await _tombstone(db_session, graph, target)

    expected_messages = {None: 2, "peer": 0, "session": 1}[target]
    stats = await crud.get_workspace_stats(db_session, graph.workspace)
    assert stats.peer_count == (1 if target == "peer" else 2)
    assert stats.session_count == (1 if target == "session" else 2)
    assert stats.message_count == expected_messages

    scoped = await crud.get_workspace_stats(
        db_session, graph.workspace, session_names=[graph.session]
    )
    assert scoped.session_count == (0 if target == "session" else 1)
    assert scoped.message_count == (1 if target is None else 0)

    active = {
        peer.name: peer.message_count
        for peer in await crud.get_active_peers(db_session, graph.workspace)
    }
    if target == "peer":
        assert active == {graph.bystander: 0}
    else:
        assert active == {graph.author: expected_messages, graph.bystander: 0}
