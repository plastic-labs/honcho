"""Tests for removing many sessions from a scope in one request."""

import importlib
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.models import Peer, Workspace
from src.security import JWTParams, create_admin_jwt, create_jwt
from src.utils.scopes import scope_peer_name

# By module path: the `src.deriver` package exports an `enqueue` function that
# shadows the submodule's attribute name.
enqueue_mod = importlib.import_module("src.deriver.enqueue")


def _setup(client: TestClient, ws: str, members: int, others: int = 0):
    """Create a scope with `members` member sessions and `others` non-members."""
    scope_name = str(generate_nanoid())
    response = client.post(f"/v3/workspaces/{ws}/scopes", json={"id": scope_name})
    assert response.status_code == 201, response.text
    names = [str(generate_nanoid()) for _ in range(members + others)]
    for name in names:
        response = client.post(f"/v3/workspaces/{ws}/sessions", json={"id": name})
        assert response.status_code in [200, 201], response.text
    member_names, other_names = names[:members], names[members:]
    if member_names:
        response = client.post(
            f"/v3/workspaces/{ws}/scopes/{scope_name}/sessions",
            json={"session_ids": member_names},
        )
        assert response.status_code == 204, response.text
    return scope_name, member_names, other_names


def _remove(client: TestClient, ws: str, scope_name: str, session_ids: Any):
    """POST the bulk-remove route."""
    return client.post(
        f"/v3/workspaces/{ws}/scopes/{scope_name}/sessions/remove",
        json={"session_ids": session_ids},
    )


def _members(client: TestClient, ws: str, scope_name: str) -> set[str]:
    """The scope's member sessions as the member list reports them."""
    response = client.post(f"/v3/workspaces/{ws}/scopes/{scope_name}/sessions/list")
    assert response.status_code == 200, response.text
    return {item["id"] for item in response.json()["items"]}


async def _removal_tasks(db: AsyncSession, ws: str, scope_name: str) -> list[str]:
    """Session names of the scope_removal tasks queued for the scope, sorted."""
    result = await db.execute(
        select(models.QueueItem.payload)
        .where(models.QueueItem.workspace_name == ws)
        .where(models.QueueItem.task_type == "scope_removal")
    )
    return sorted(
        payload["session_name"]
        for (payload,) in result.all()
        if payload["scope_peer"] == scope_peer_name(scope_name)
    )


async def test_bulk_remove_ends_memberships_and_queues_reconciliation(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Members are removed and reconciled; non-members are ignored and get no task."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    scope_name, (kept, gone_1, gone_2), (outsider,) = _setup(client, ws, 3, others=1)

    response = _remove(client, ws, scope_name, [gone_2, gone_1, outsider, gone_1])
    assert response.status_code == 204, response.text

    assert _members(client, ws, scope_name) == {kept}
    # One task per membership actually ended: none for the outsider, and the
    # duplicate ID doesn't queue twice. A task for a non-member would still
    # clear the scope's dream notes.
    assert await _removal_tasks(db_session, ws, scope_name) == sorted([gone_1, gone_2])
    rows = await db_session.execute(
        select(models.SessionPeer.session_name, models.SessionPeer.left_at)
        .where(models.SessionPeer.workspace_name == ws)
        .where(models.SessionPeer.peer_name == scope_peer_name(scope_name))
    )
    left = {name for name, left_at in rows.all() if left_at is not None}
    assert left == {gone_1, gone_2}


async def test_bulk_remove_is_idempotent(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Removing sessions that already left is a no-op that queues nothing new."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    scope_name, members, _ = _setup(client, ws, 2)

    assert _remove(client, ws, scope_name, members).status_code == 204
    tasks = await _removal_tasks(db_session, ws, scope_name)
    # Drain the first tasks, so the queue's dedup of pending work can't hide a
    # second enqueue.
    await db_session.execute(
        update(models.QueueItem)
        .where(models.QueueItem.workspace_name == ws)
        .values(processed=True)
    )
    await db_session.commit()
    assert _remove(client, ws, scope_name, members).status_code == 204
    assert await _removal_tasks(db_session, ws, scope_name) == tasks
    assert _members(client, ws, scope_name) == set()


async def test_bulk_remove_unknown_session_removes_nothing(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """The batch is all-or-nothing: one unknown session is a 404 and no change."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    scope_name, members, _ = _setup(client, ws, 2)

    response = _remove(client, ws, scope_name, [*members, str(generate_nanoid())])
    assert response.status_code == 404, response.text
    assert _members(client, ws, scope_name) == set(members)
    assert await _removal_tasks(db_session, ws, scope_name) == []


def test_bulk_remove_validation(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Unknown scope is a 404; an empty or oversized batch is a 422."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    scope_name, members, _ = _setup(client, ws, 1)

    assert _remove(client, ws, str(generate_nanoid()), members).status_code == 404
    assert _remove(client, ws, scope_name, []).status_code == 422
    too_many = [str(generate_nanoid()) for _ in range(101)]
    assert _remove(client, ws, scope_name, too_many).status_code == 422
    assert _members(client, ws, scope_name) == set(members)


def test_bulk_remove_requires_workspace_level_key(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Peer-, session- and named-scope keys can't remove members; workspace keys can."""
    test_workspace, test_peer = sample_data
    ws = test_workspace.name
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    client.headers["Authorization"] = f"Bearer {create_admin_jwt()}"
    scope_name, members, _ = _setup(client, ws, 2)
    response = client.post(
        "/v3/keys", params={"workspace_id": ws, "scope_id": scope_name}
    )
    assert response.status_code == 200, response.text
    named_scope_key = response.json()["key"]

    for key in (
        create_jwt(JWTParams(w=ws, p=test_peer.name)),
        create_jwt(JWTParams(w=ws, s=members[0])),
        named_scope_key,
    ):
        client.headers["Authorization"] = f"Bearer {key}"
        assert _remove(client, ws, scope_name, members).status_code == 401

    client.headers["Authorization"] = f"Bearer {create_jwt(JWTParams(w=ws))}"
    assert _remove(client, ws, scope_name, members).status_code == 204
    assert _members(client, ws, scope_name) == set()


async def test_bulk_remove_stays_in_its_workspace(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Same scope and session names elsewhere: only this workspace changes.

    Both sessions are members in the other workspace; here only the first is.
    Removing both here must end just this workspace's membership, queue a task
    only for it, and leave the other workspace's memberships alone.
    """
    test_workspace, _ = sample_data
    ws = test_workspace.name
    other_ws = str(generate_nanoid())
    response = client.post("/v3/workspaces", json={"id": other_ws})
    assert response.status_code in [200, 201], response.text
    scope_name = str(generate_nanoid())
    shared = [str(generate_nanoid()) for _ in range(2)]
    for workspace_name in (ws, other_ws):
        response = client.post(
            f"/v3/workspaces/{workspace_name}/scopes", json={"id": scope_name}
        )
        assert response.status_code == 201, response.text
        for name in shared:
            response = client.post(
                f"/v3/workspaces/{workspace_name}/sessions", json={"id": name}
            )
            assert response.status_code in [200, 201], response.text
    response = client.post(
        f"/v3/workspaces/{other_ws}/scopes/{scope_name}/sessions",
        json={"session_ids": shared},
    )
    assert response.status_code == 204, response.text
    response = client.post(
        f"/v3/workspaces/{ws}/scopes/{scope_name}/sessions",
        json={"session_ids": shared[:1]},
    )
    assert response.status_code == 204, response.text

    assert _remove(client, ws, scope_name, shared).status_code == 204
    assert _members(client, ws, scope_name) == set()
    assert await _removal_tasks(db_session, ws, scope_name) == shared[:1]
    assert _members(client, other_ws, scope_name) == set(shared)


async def test_bulk_remove_is_atomic_with_its_reconciliation_tasks(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """If queueing a task fails, no membership ends, so a retry still works.

    Tasks are queued only for memberships the call ends, so a task lost after
    the memberships committed could never be queued by a retry.
    """
    test_workspace, _ = sample_data
    ws = test_workspace.name
    scope_name, members, _ = _setup(client, ws, 3)
    real_insert = enqueue_mod.insert_scope_task
    calls = 0

    async def fail_on_second(*args: Any, **kwargs: Any) -> bool:
        """Let the first task insert through, then fail the second."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("queue insert failed")
        return await real_insert(*args, **kwargs)

    monkeypatch.setattr(enqueue_mod, "insert_scope_task", fail_on_second)
    with pytest.raises(RuntimeError):
        await crud.remove_sessions_from_scope(db_session, ws, scope_name, members)
    await db_session.rollback()
    assert _members(client, ws, scope_name) == set(members)
    assert await _removal_tasks(db_session, ws, scope_name) == []

    monkeypatch.setattr(enqueue_mod, "insert_scope_task", real_insert)
    assert _remove(client, ws, scope_name, members).status_code == 204
    assert await _removal_tasks(db_session, ws, scope_name) == sorted(members)


async def test_bulk_remove_locks_in_the_order_bulk_add_does(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Memberships lock in Python sorted() order, as bulk add's upserts do.

    Under a linguistic database collation (en_US) "a…" sorts before "Z…", but
    sorted() puts "Z…" first; locking in the database's order instead could
    deadlock against a concurrent bulk add of the same sessions.
    """
    test_workspace, _ = sample_data
    ws = test_workspace.name
    scope_name = str(generate_nanoid())
    assert (
        client.post(f"/v3/workspaces/{ws}/scopes", json={"id": scope_name}).status_code
        == 201
    )
    suffix = str(generate_nanoid(size=6)).replace("-", "_")
    names = [f"a{suffix}", f"Z{suffix}", f"b{suffix}", f"B{suffix}", f"9{suffix}"]
    for name in names:
        response = client.post(f"/v3/workspaces/{ws}/sessions", json={"id": name})
        assert response.status_code in [200, 201], response.text
    response = client.post(
        f"/v3/workspaces/{ws}/scopes/{scope_name}/sessions",
        json={"session_ids": names},
    )
    assert response.status_code == 204, response.text

    removed = await crud.remove_sessions_from_scope(db_session, ws, scope_name, names)
    assert removed == sorted(names)


def test_bulk_remove_edges(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """100 IDs are accepted; another scope's memberships are untouched."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    scope_name, members, _ = _setup(client, ws, 2)
    other_scope, _, _ = _setup(client, ws, 0)
    response = client.post(
        f"/v3/workspaces/{ws}/scopes/{other_scope}/sessions",
        json={"session_ids": members},
    )
    assert response.status_code == 204, response.text

    # 98 IDs that aren't members, plus the two that are: exactly the limit
    padding = [str(generate_nanoid()) for _ in range(98)]
    for name in padding:
        response = client.post(f"/v3/workspaces/{ws}/sessions", json={"id": name})
        assert response.status_code in [200, 201], response.text
    assert _remove(client, ws, scope_name, [*members, *padding]).status_code == 204
    assert _members(client, ws, scope_name) == set()
    assert _members(client, ws, other_scope) == set(members)


async def test_bulk_remove_rejects_archived_sessions(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """An archived session counts as missing, as the single-session DELETE treats it."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    scope_name, members, _ = _setup(client, ws, 2)
    await db_session.execute(
        update(models.Session)
        .where(models.Session.workspace_name == ws)
        .where(models.Session.name == members[0])
        .values(is_active=False)
    )
    await db_session.commit()

    assert _remove(client, ws, scope_name, members).status_code == 404
    assert await _removal_tasks(db_session, ws, scope_name) == []
