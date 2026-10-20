"""Peer membership routes: structure, cycles, caps and auth."""

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from src import crud, models, schemas
from src.config import settings
from src.exceptions import ConflictException
from src.models import Peer, Workspace
from src.security import JWTParams, create_jwt


def _members_url(workspace: str, peer: str) -> str:
    return f"/v3/workspaces/{workspace}/peers/{peer}/members"


def _add(client: TestClient, workspace: str, parent: str, member: str, **body: Any):
    return client.post(
        _members_url(workspace, parent), json={"peer_id": member, **body}
    )


def _member_ids(client: TestClient, workspace: str, peer: str) -> list[str]:
    response = client.post(f"{_members_url(workspace, peer)}/list", json={})
    assert response.status_code == 200
    return [item["peer_id"] for item in response.json()["items"]]


def test_add_list_and_remove_members(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    workspace, _ = sample_data
    ws = workspace.name

    created = _add(client, ws, "team", "alice", metadata={"role": "lead"})
    assert created.status_code == 201
    body = created.json()
    assert body["parent_id"] == "team"
    assert body["peer_id"] == "alice"
    assert body["workspace_id"] == ws
    assert body["metadata"] == {"role": "lead"}

    assert _add(client, ws, "team", "bob").status_code == 201
    assert _member_ids(client, ws, "team") == ["alice", "bob"]

    parents = client.post(
        f"/v3/workspaces/{ws}/peers/alice/parents/list", json={}
    ).json()["items"]
    assert [p["parent_id"] for p in parents] == ["team"]

    # Re-adding is idempotent and replaces metadata when given.
    again = _add(client, ws, "team", "alice", metadata={"role": "ic"})
    assert again.status_code == 200
    assert again.json()["metadata"] == {"role": "ic"}
    assert _member_ids(client, ws, "team") == ["alice", "bob"]

    removed = client.delete(f"{_members_url(ws, 'team')}/alice")
    assert removed.status_code == 204
    assert _member_ids(client, ws, "team") == ["bob"]
    # Removing a non-member is a no-op.
    assert client.delete(f"{_members_url(ws, 'team')}/alice").status_code == 204


def test_members_list_filters_on_member_metadata(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    workspace, _ = sample_data
    ws = workspace.name
    client.post(
        f"/v3/workspaces/{ws}/peers",
        json={"name": "carol", "metadata": {"team": "infra"}},
    )
    _add(client, ws, "org", "carol")
    _add(client, ws, "org", "dave")

    response = client.post(
        f"{_members_url(ws, 'org')}/list",
        json={"filters": {"metadata": {"team": "infra"}}},
    )
    assert response.status_code == 200
    assert [i["peer_id"] for i in response.json()["items"]] == ["carol"]


def test_cycles_and_self_membership_are_refused(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    workspace, _ = sample_data
    ws = workspace.name
    assert _add(client, ws, "a", "b").status_code == 201
    assert _add(client, ws, "b", "c").status_code == 201

    cycle = _add(client, ws, "c", "a")
    assert cycle.status_code == 409
    assert "a -> b -> c -> a" in cycle.json()["detail"]

    # Direct two-node cycle.
    assert _add(client, ws, "b", "a").status_code == 409

    self_member = _add(client, ws, "a", "a")
    assert self_member.status_code == 422

    # A diamond is not a cycle.
    assert _add(client, ws, "a", "c").status_code == 201


def test_scope_peers_cannot_be_members_or_parents(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    workspace, _ = sample_data
    ws = workspace.name
    assert client.post(
        f"/v3/workspaces/{ws}/scopes", json={"id": "q3"}
    ).status_code in (200, 201)

    as_member = _add(client, ws, "team", "scope.q3")
    assert as_member.status_code == 422
    as_parent = _add(client, ws, "scope.q3", "alice")
    assert as_parent.status_code == 422


def test_member_cap(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, _ = sample_data
    ws = workspace.name
    monkeypatch.setattr(settings.PEER, "MAX_MEMBERS", 2)
    assert _add(client, ws, "team", "m1").status_code == 201
    assert _add(client, ws, "team", "m2").status_code == 201
    over = _add(client, ws, "team", "m3")
    assert over.status_code == 422
    assert "maximum of 2 members" in over.json()["detail"]
    # Re-adding an existing member is not a new member.
    assert _add(client, ws, "team", "m1").status_code == 200


@pytest.mark.asyncio
async def test_concurrent_inserts_cannot_close_a_cycle(
    db_engine: AsyncEngine, sample_data: tuple[Workspace, Peer]
):
    """x->y and y->x submitted together: the advisory lock lets exactly one in."""
    workspace, _ = sample_data
    ws = workspace.name
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async with session_factory() as setup:
        await crud.get_or_create_peers(
            setup,
            workspace_name=ws,
            peers=[
                schemas.PeerSpec(name="x"),
                schemas.PeerSpec(name="y"),
            ],
        )
        await setup.commit()

    async def attempt(parent: str, member: str) -> str:
        async with session_factory() as session:
            try:
                await crud.add_member(
                    session, ws, parent_name=parent, member_name=member
                )
                # Hold the lock long enough that the other insert must wait.
                await asyncio.sleep(0.2)
                await session.commit()
                return "ok"
            except ConflictException:
                await session.rollback()
                return "cycle"

    outcomes = await asyncio.gather(attempt("x", "y"), attempt("y", "x"))
    assert sorted(outcomes) == ["cycle", "ok"]

    async with session_factory() as check:
        count = await check.scalar(
            select(func.count())
            .select_from(models.PeerMembership)
            .where(models.PeerMembership.workspace_name == ws)
        )
    assert count == 1


@pytest.mark.asyncio
async def test_concurrent_first_adds_of_new_peers_cannot_close_a_cycle(
    db_engine: AsyncEngine, sample_data: tuple[Workspace, Peer]
):
    """Neither peer exists yet: both requests create them, then one wins."""
    workspace, _ = sample_data
    ws = workspace.name
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async def attempt(parent: str, member: str) -> str:
        async with session_factory() as session:
            try:
                await crud.add_member(
                    session, ws, parent_name=parent, member_name=member
                )
                await asyncio.sleep(0.2)
                await session.commit()
                return "ok"
            except ConflictException:
                await session.rollback()
                return "cycle"

    outcomes = await asyncio.gather(attempt("p", "q"), attempt("q", "p"))
    assert sorted(outcomes) == ["cycle", "ok"]


def test_cycle_check_is_linear_on_a_dense_ladder(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Every peer on a level belongs to both peers above it: 2^levels paths.

    Walking paths instead of peers would not finish; walking peers is a
    handful of queries.
    """
    workspace, _ = sample_data
    ws = workspace.name
    levels = 30
    for level in range(1, levels):
        for upper in ("a", "b"):
            for lower in ("a", "b"):
                assert (
                    _add(
                        client, ws, f"{upper}{level - 1}", f"{lower}{level}"
                    ).status_code
                    == 201
                )

    # a0 is above everything, so this closes a cycle at the far end.
    response = _add(client, ws, f"a{levels - 1}", "a0")
    assert response.status_code == 409
    # And an unrelated peer is still accepted.
    assert _add(client, ws, f"a{levels - 1}", "fresh").status_code == 201


def test_membership_mutation_requires_workspace_key(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, _ = sample_data
    ws = workspace.name
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")

    peer_key = create_jwt(JWTParams(w=ws, p="team"))
    workspace_key = create_jwt(JWTParams(w=ws))

    client.headers["Authorization"] = f"Bearer {peer_key}"
    assert _add(client, ws, "team", "alice").status_code == 401
    assert client.delete(f"{_members_url(ws, 'team')}/alice").status_code == 401

    client.headers["Authorization"] = f"Bearer {workspace_key}"
    assert _add(client, ws, "team", "alice").status_code == 201

    # A peer key may list its own members and parents.
    client.headers["Authorization"] = f"Bearer {peer_key}"
    assert _member_ids(client, ws, "team") == ["alice"]
    other = create_jwt(JWTParams(w=ws, p="mallory"))
    client.headers["Authorization"] = f"Bearer {other}"
    assert client.post(f"{_members_url(ws, 'team')}/list", json={}).status_code == 401


def test_members_routes_unique_name_per_workspace(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """The same names in another workspace never share memberships."""
    workspace, _ = sample_data
    other = str(generate_nanoid())
    client.post("/v3/workspaces", json={"name": other})
    _add(client, workspace.name, "team", "alice")
    assert _member_ids(client, other, "team") == []


@pytest.mark.parametrize("cursor_mode", [False, True])
def test_member_lists_paginate(
    client: TestClient, sample_data: tuple[Workspace, Peer], cursor_mode: bool
):
    workspace, _ = sample_data
    ws = workspace.name
    names = [f"m{i}" for i in range(5)]
    for name in names:
        _add(client, ws, "team", name)
        _add(client, ws, name, "leaf")

    def walk(url: str) -> list[dict[str, Any]]:
        seen: list[dict[str, str]] = []
        if not cursor_mode:
            page, pages = 1, 1
            while page <= pages:
                body = client.post(url, params={"page": page, "size": 2}, json={})
                assert body.status_code == 200, body.text
                seen += body.json()["items"]
                pages = body.json()["pages"]
                page += 1
            return seen
        cursor: str | None = ""
        while cursor is not None:
            body = client.post(url, params={"cursor": cursor, "size": 2}, json={})
            assert body.status_code == 200, body.text
            seen += body.json()["items"]
            cursor = body.json()["next_page"]
        return seen

    members = walk(f"{_members_url(ws, 'team')}/list")
    assert [m["peer_id"] for m in members] == names
    parents = walk(f"/v3/workspaces/{ws}/peers/leaf/parents/list")
    assert [p["parent_id"] for p in parents] == names
