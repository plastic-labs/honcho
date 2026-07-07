"""Tests for the scopes facade: scope-kind peers, guardrails, and CRUD routes.

A scope is a named grouping of sessions, implemented as a peer named
``scope__<name>`` with configuration ``{"kind": "scope", "observe_me": false}``
that observes its member sessions and never speaks. See src/utils/scopes.py.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.enqueue import enqueue
from src.models import Peer, QueueItem, Workspace
from src.security import JWTParams, create_jwt
from src.utils.scopes import (
    SCOPE_PEER_PREFIX,
    is_scope_peer_name,
    scope_name_from_peer,
    scope_peer_name,
)


def _create_scope(
    client: TestClient,
    workspace_name: str,
    scope_name: str,
    metadata: dict[str, Any] | None = None,
):
    body: dict[str, Any] = {"id": scope_name}
    if metadata is not None:
        body["metadata"] = metadata
    return client.post(f"/v3/workspaces/{workspace_name}/scopes", json=body)


def _create_session(
    client: TestClient,
    workspace_name: str,
    session_name: str | None = None,
    **extra: Any,
):
    session_name = session_name or str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions",
        json={"id": session_name, **extra},
    )
    assert response.status_code in [200, 201]
    return session_name


async def _get_session_peer(
    db_session: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_name: str,
) -> models.SessionPeer | None:
    return await db_session.scalar(
        select(models.SessionPeer)
        .where(models.SessionPeer.workspace_name == workspace_name)
        .where(models.SessionPeer.session_name == session_name)
        .where(models.SessionPeer.peer_name == peer_name)
    )


def test_scope_namespace_helpers():
    assert scope_peer_name("therapy") == "scope__therapy"
    assert is_scope_peer_name("scope__therapy")
    assert not is_scope_peer_name("therapy")
    assert scope_name_from_peer("scope__therapy") == "therapy"


async def test_create_scope_creates_flagged_peer(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Creating a scope creates a peer with the kind flag and observe_me=false."""
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())

    response = _create_scope(
        client, test_workspace.name, scope_name, metadata={"purpose": "testing"}
    )
    assert response.status_code == 201
    data = response.json()
    # The response id is the UNPREFIXED scope name
    assert data["id"] == scope_name
    assert data["metadata"] == {"purpose": "testing"}
    assert "created_at" in data

    peer = await db_session.scalar(
        select(models.Peer)
        .where(models.Peer.workspace_name == test_workspace.name)
        .where(models.Peer.name == scope_peer_name(scope_name))
    )
    assert peer is not None
    assert peer.configuration == {"kind": "scope", "observe_me": False}


def test_create_scope_idempotent(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())

    first = _create_scope(client, test_workspace.name, scope_name)
    assert first.status_code == 201

    second = _create_scope(client, test_workspace.name, scope_name)
    assert second.status_code == 200
    assert second.json()["id"] == scope_name


def test_create_scope_rejects_invalid_names(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    # Names must match the resource name pattern
    response = _create_scope(client, test_workspace.name, "bad name!")
    assert response.status_code == 422

    # Scope names are unprefixed: double-prefixing is rejected
    response = _create_scope(client, test_workspace.name, "scope__therapy")
    assert response.status_code == 422


async def test_create_scope_rejects_legacy_collision(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """A pre-existing plain peer occupying the reserved name is never adopted."""
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())

    legacy_peer = models.Peer(
        workspace_name=test_workspace.name,
        name=scope_peer_name(scope_name),
    )
    db_session.add(legacy_peer)
    await db_session.commit()

    response = _create_scope(client, test_workspace.name, scope_name)
    assert response.status_code == 409

    # And the collision peer is invisible to the scope read routes
    response = client.get(f"/v3/workspaces/{test_workspace.name}/scopes/{scope_name}")
    assert response.status_code == 404


def test_get_scope(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201

    response = client.get(f"/v3/workspaces/{test_workspace.name}/scopes/{scope_name}")
    assert response.status_code == 200
    assert response.json()["id"] == scope_name

    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/scopes/{generate_nanoid()}"
    )
    assert response.status_code == 404


def test_list_scopes(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """The scopes list contains only scopes, with unprefixed ids."""
    test_workspace, test_peer = sample_data
    scope_names = {str(generate_nanoid()), str(generate_nanoid())}
    for scope_name in scope_names:
        assert _create_scope(client, test_workspace.name, scope_name).status_code == 201

    response = client.post(f"/v3/workspaces/{test_workspace.name}/scopes/list")
    assert response.status_code == 200
    items = response.json()["items"]
    listed = {item["id"] for item in items}
    assert listed == scope_names
    assert test_peer.name not in listed


def test_scopes_routes_require_workspace_level_key(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Scopes are an app-level admin surface: workspace keys work, peer- and
    session-scoped keys are rejected."""
    test_workspace, test_peer = sample_data
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    scope_name = str(generate_nanoid())
    scopes_url = f"/v3/workspaces/{test_workspace.name}/scopes"

    # Workspace-scoped key: allowed
    client.headers["Authorization"] = (
        f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
    )
    assert client.post(scopes_url, json={"id": scope_name}).status_code == 201

    # Peer-scoped key: rejected
    client.headers["Authorization"] = (
        f"Bearer {create_jwt(JWTParams(w=test_workspace.name, p=test_peer.name))}"
    )
    assert client.post(scopes_url, json={"id": scope_name}).status_code == 401
    assert client.post(f"{scopes_url}/list").status_code == 401
    assert client.get(f"{scopes_url}/{scope_name}").status_code == 401

    # Session-scoped key: rejected
    client.headers["Authorization"] = (
        f"Bearer {create_jwt(JWTParams(w=test_workspace.name, s='some-session'))}"
    )
    assert client.get(f"{scopes_url}/{scope_name}/sessions").status_code == 401


def test_peer_create_rejects_reserved_prefix(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """User-created peers may not use the reserved scope prefix."""
    test_workspace, _ = sample_data

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": f"{SCOPE_PEER_PREFIX}{generate_nanoid()}"},
    )
    assert response.status_code == 422
    assert SCOPE_PEER_PREFIX in response.json()["detail"]


def test_peers_list_kind_filtering(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """peers.list excludes scope peers by default; kind switches the view."""
    test_workspace, test_peer = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    backing_peer_name = scope_peer_name(scope_name)

    # Default: scope peers excluded
    response = client.post(f"/v3/workspaces/{test_workspace.name}/peers/list")
    assert response.status_code == 200
    names = {item["id"] for item in response.json()["items"]}
    assert test_peer.name in names
    assert backing_peer_name not in names

    # kind=scope: only scope peers
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/list",
        json={"kind": "scope"},
    )
    assert response.status_code == 200
    names = {item["id"] for item in response.json()["items"]}
    assert names == {backing_peer_name}

    # kind=all: everything
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/list",
        json={"kind": "all"},
    )
    assert response.status_code == 200
    names = {item["id"] for item in response.json()["items"]}
    assert {test_peer.name, backing_peer_name} <= names


def test_scope_peer_cannot_author_messages(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    session_name = _create_session(client, test_workspace.name)

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/messages",
        json={
            "messages": [
                {
                    "peer_id": scope_peer_name(scope_name),
                    "content": "I should not speak",
                }
            ]
        },
    )
    assert response.status_code == 422
    assert SCOPE_PEER_PREFIX in response.json()["detail"]


def test_scope_peer_cannot_be_chat_target(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Chat validation happens before any LLM work, so this is safe to exercise."""
    test_workspace, test_peer = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        json={"query": "what do you know?", "target": scope_peer_name(scope_name)},
    )
    assert response.status_code == 422


def test_scope_peer_cannot_be_representation_target(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/{test_peer.name}/representation",
        json={"target": scope_peer_name(scope_name)},
    )
    assert response.status_code == 422


def test_generic_session_peer_routes_reject_scope_peers(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Scope membership is managed only via the scopes facade."""
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    backing_peer_name = scope_peer_name(scope_name)
    session_name = _create_session(client, test_workspace.name)

    base = f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}"

    response = client.post(f"{base}/peers", json={backing_peer_name: {}})
    assert response.status_code == 422
    assert "scopes" in response.json()["detail"]

    response = client.put(f"{base}/peers", json={backing_peer_name: {}})
    assert response.status_code == 422

    response = client.request("DELETE", f"{base}/peers", json=[backing_peer_name])
    assert response.status_code == 422

    # Session creation with a scope peer in the generic peers mapping is also
    # rejected; the `scopes` field is the supported path.
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": str(generate_nanoid()), "peers": {backing_peer_name: {}}},
    )
    assert response.status_code == 422


async def test_scope_sessions_add_list_remove(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    test_workspace, _ = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    assert _create_scope(client, workspace_name, scope_name).status_code == 201
    session_1 = _create_session(client, workspace_name)
    session_2 = _create_session(client, workspace_name)

    scope_base = f"/v3/workspaces/{workspace_name}/scopes/{scope_name}"

    # Add both sessions
    response = client.post(
        f"{scope_base}/sessions", json={"session_ids": [session_1, session_2]}
    )
    assert response.status_code == 200
    assert set(response.json()["session_ids"]) == {session_1, session_2}

    # Membership rows carry the observer shape: observe_others on, observe_me off
    session_peer = await _get_session_peer(
        db_session, workspace_name, session_1, scope_peer_name(scope_name)
    )
    assert session_peer is not None
    assert session_peer.left_at is None
    assert session_peer.configuration["observe_others"] is True
    assert session_peer.configuration["observe_me"] is False

    # List memberships
    response = client.get(f"{scope_base}/sessions")
    assert response.status_code == 200
    assert set(response.json()["session_ids"]) == {session_1, session_2}

    # Remove one membership (soft delete, like the generic remove-peer path)
    response = client.delete(f"{scope_base}/sessions/{session_1}")
    assert response.status_code == 204

    response = client.get(f"{scope_base}/sessions")
    assert response.status_code == 200
    assert response.json()["session_ids"] == [session_2]

    db_session.expire_all()
    session_peer = await _get_session_peer(
        db_session, workspace_name, session_1, scope_peer_name(scope_name)
    )
    assert session_peer is not None
    assert session_peer.left_at is not None


def test_scope_sessions_add_missing_session_404(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    existing_session = _create_session(client, test_workspace.name)

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/scopes/{scope_name}/sessions",
        json={"session_ids": [existing_session, str(generate_nanoid())]},
    )
    assert response.status_code == 404


def test_scope_sessions_routes_404_on_unknown_scope(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    session_name = _create_session(client, test_workspace.name)
    scope_base = f"/v3/workspaces/{test_workspace.name}/scopes/{generate_nanoid()}"

    response = client.post(
        f"{scope_base}/sessions", json={"session_ids": [session_name]}
    )
    assert response.status_code == 404

    assert client.get(f"{scope_base}/sessions").status_code == 404
    assert client.delete(f"{scope_base}/sessions/{session_name}").status_code == 404


async def test_session_create_with_scopes(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """`scopes` on session creation creates the scope peers and memberships."""
    test_workspace, _ = sample_data
    scope_a = str(generate_nanoid())
    scope_b = str(generate_nanoid())
    session_name = str(generate_nanoid())

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_name, "scopes": [scope_a, scope_b]},
    )
    assert response.status_code == 201

    for scope_name in (scope_a, scope_b):
        peer = await db_session.scalar(
            select(models.Peer)
            .where(models.Peer.workspace_name == test_workspace.name)
            .where(models.Peer.name == scope_peer_name(scope_name))
        )
        assert peer is not None
        assert peer.configuration == {"kind": "scope", "observe_me": False}

        session_peer = await _get_session_peer(
            db_session, test_workspace.name, session_name, scope_peer_name(scope_name)
        )
        assert session_peer is not None
        assert session_peer.configuration["observe_others"] is True
        assert session_peer.configuration["observe_me"] is False

    # And the memberships show up through the facade
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/scopes/{scope_a}/sessions"
    )
    assert response.status_code == 200
    assert response.json()["session_ids"] == [session_name]


def test_session_create_rejects_prefixed_scope_names(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": str(generate_nanoid()), "scopes": ["scope__x"]},
    )
    assert response.status_code == 422

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": str(generate_nanoid()), "scopes": ["bad name!"]},
    )
    assert response.status_code == 422


async def test_scope_membership_equals_hand_built_observer(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Facade-less equivalence: a scope membership row is exactly what a
    hand-built observer peer would have (name and kind flag aside)."""
    test_workspace, _ = sample_data

    # Hand-built observer peer added through the generic session-peer route
    observer_name = str(generate_nanoid())
    observer_session = _create_session(client, test_workspace.name)
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{observer_session}/peers",
        json={observer_name: {"observe_others": True, "observe_me": False}},
    )
    assert response.status_code == 200

    # Scope membership added through the facade
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    scope_session = _create_session(client, test_workspace.name)
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/scopes/{scope_name}/sessions",
        json={"session_ids": [scope_session]},
    )
    assert response.status_code == 200

    hand_built = await _get_session_peer(
        db_session, test_workspace.name, observer_session, observer_name
    )
    via_facade = await _get_session_peer(
        db_session, test_workspace.name, scope_session, scope_peer_name(scope_name)
    )
    assert hand_built is not None and via_facade is not None
    assert hand_built.configuration == via_facade.configuration
    assert hand_built.left_at is None and via_facade.left_at is None


async def test_scope_peer_observes_ingested_messages(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """End-to-end litmus: after adding a session to a scope, a message from a
    real peer fans out a representation task with the scope peer as observer."""
    test_workspace, test_peer = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    session_name = _create_session(client, test_workspace.name)

    # Add the speaking peer and the scope membership
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
        json={test_peer.name: {}},
    )
    assert response.status_code == 200
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/scopes/{scope_name}/sessions",
        json={"session_ids": [session_name]},
    )
    assert response.status_code == 200

    # Ingest a message from the real peer and run the deriver enqueue fan-out
    message = models.Message(
        workspace_name=test_workspace.name,
        session_name=session_name,
        peer_name=test_peer.name,
        content="I love hiking in the mountains",
        public_id=generate_nanoid(),
        seq_in_session=1,
        token_count=10,
    )
    db_session.add(message)
    await db_session.commit()

    await enqueue(
        [
            {
                "workspace_name": test_workspace.name,
                "session_name": session_name,
                "message_id": message.id,
                "content": message.content,
                "peer_name": test_peer.name,
                "created_at": message.created_at,
                "message_public_id": message.public_id,
                "seq_in_session": message.seq_in_session,
            }
        ]
    )

    result = await db_session.execute(
        select(QueueItem)
        .where(QueueItem.task_type == "representation")
        .where(QueueItem.message_id == message.id)
    )
    representation_items = list(result.scalars().all())
    assert len(representation_items) == 1
    payload = representation_items[0].payload
    assert payload.get("observed") == test_peer.name
    observers = payload.get("observers")
    assert observers is not None
    assert test_peer.name in observers  # self-observation
    assert scope_peer_name(scope_name) in observers  # the scope observes
