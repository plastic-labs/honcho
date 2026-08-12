"""Tests for the scopes facade: scope-kind peers, guardrails, and CRUD routes.

A scope is a named grouping of sessions, implemented as a peer named
``scope.<name>`` carrying ``{"kind": "scope"}`` in ``internal_metadata`` and
``{"observe_me": false}`` in ``configuration``, that observes its member sessions
and never speaks. See src/utils/scopes.py.
"""

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.deriver.enqueue import enqueue
from src.exceptions import ValidationException
from src.models import Peer, QueueItem, Workspace
from src.schemas.api import RESOURCE_NAME_PATTERN
from src.security import JWTParams, create_jwt
from src.utils.scopes import (
    SCOPE_PEER_PREFIX,
    is_scope_peer,
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


def _add_sessions(
    client: TestClient,
    workspace_name: str,
    scope_name: str,
    session_names: list[str],
) -> int:
    """Add sessions to a scope via the facade; returns the status code (204 on success)."""
    return client.post(
        f"/v3/workspaces/{workspace_name}/scopes/{scope_name}/sessions",
        json={"session_ids": session_names},
    ).status_code


def _scope_sessions(
    client: TestClient, workspace_name: str, scope_name: str
) -> list[str]:
    """Names of a scope's member sessions, oldest membership first."""
    response = client.post(
        f"/v3/workspaces/{workspace_name}/scopes/{scope_name}/sessions/list"
    )
    assert response.status_code == 200, response.text
    return [item["id"] for item in response.json()["items"]]


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
    assert scope_peer_name("therapy") == "scope.therapy"
    assert is_scope_peer_name("scope.therapy")
    assert not is_scope_peer_name("therapy")
    assert scope_name_from_peer("scope.therapy") == "therapy"
    # The prefix must stay outside the peer-name charset, or an existing peer
    # could occupy the scope namespace.
    assert not re.fullmatch(RESOURCE_NAME_PATTERN, SCOPE_PEER_PREFIX)


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
    # The kind flag lives in internal_metadata (not user-writable); only
    # observe_me is in the user-visible configuration.
    assert peer.internal_metadata == {"kind": "scope"}
    assert peer.configuration == {"observe_me": False}


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
    response = _create_scope(client, test_workspace.name, "scope.therapy")
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
    assert client.post(f"{scopes_url}/{scope_name}/sessions/list").status_code == 401


def test_session_create_scopes_requires_workspace_level_key(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """`scopes` on session create is the scopes routes by another door.

    It creates scope peers and attaches memberships, so it carries the same
    workspace-level requirement. Session create is otherwise a self-authorizing
    route that accepts peer- and session-scoped keys, which is exactly why this
    needs its own check rather than the route's auth dependency.
    """
    test_workspace, test_peer = sample_data
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    sessions_url = f"/v3/workspaces/{test_workspace.name}/sessions"
    scopes_url = f"/v3/workspaces/{test_workspace.name}/scopes"
    scope_name = str(generate_nanoid())

    def create_with_scope(session_name: str) -> int:
        return client.post(
            sessions_url, json={"id": session_name, "scopes": [scope_name]}
        ).status_code

    def listed_scopes() -> set[str]:
        response = client.post(f"{scopes_url}/list")
        assert response.status_code == 200
        return {item["id"] for item in response.json()["items"]}

    # Peer-scoped key: rejected
    client.headers["Authorization"] = (
        f"Bearer {create_jwt(JWTParams(w=test_workspace.name, p=test_peer.name))}"
    )
    assert create_with_scope(str(generate_nanoid())) == 401

    # Session-scoped key: rejected. The session name must match the token's `s`
    # claim, or the handler's own session check would 401 first and this would
    # pass without ever reaching the scopes gate.
    own_session = str(generate_nanoid())
    client.headers["Authorization"] = (
        f"Bearer {create_jwt(JWTParams(w=test_workspace.name, s=own_session))}"
    )
    assert create_with_scope(own_session) == 401

    # Workspace-scoped key: allowed. The scope is absent until this call lands,
    # which proves the rejected requests did not mint it on their way out.
    client.headers["Authorization"] = (
        f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
    )
    assert scope_name not in listed_scopes()
    assert create_with_scope(str(generate_nanoid())) == 201
    assert scope_name in listed_scopes()


def test_peer_create_rejects_reserved_prefix(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """User-created peers may not use the reserved scope prefix.

    The prefix sits outside RESOURCE_NAME_PATTERN, so PeerCreate's own charset
    validation rejects it at the schema boundary — one layer earlier than the
    route's validate_no_scope_peer_names guard. Either way the caller gets 422.
    """
    test_workspace, _ = sample_data

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": f"{SCOPE_PEER_PREFIX}{generate_nanoid()}"},
    )
    assert response.status_code == 422
    assert RESOURCE_NAME_PATTERN in str(response.json()["detail"])


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
    assert (
        _add_sessions(client, workspace_name, scope_name, [session_1, session_2]) == 204
    )
    assert set(_scope_sessions(client, workspace_name, scope_name)) == {
        session_1,
        session_2,
    }

    # Membership rows carry the observer shape: observe_others on, observe_me off
    session_peer = await _get_session_peer(
        db_session, workspace_name, session_1, scope_peer_name(scope_name)
    )
    assert session_peer is not None
    assert session_peer.left_at is None
    assert session_peer.configuration["observe_others"] is True
    assert session_peer.configuration["observe_me"] is False

    # Remove one membership (soft delete, like the generic remove-peer path)
    response = client.delete(f"{scope_base}/sessions/{session_1}")
    assert response.status_code == 204

    assert _scope_sessions(client, workspace_name, scope_name) == [session_2]

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

    assert (
        _add_sessions(
            client,
            test_workspace.name,
            scope_name,
            [existing_session, str(generate_nanoid())],
        )
        == 404
    )


def test_scope_sessions_routes_404_on_unknown_scope(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    session_name = _create_session(client, test_workspace.name)
    unknown_scope = str(generate_nanoid())
    scope_base = f"/v3/workspaces/{test_workspace.name}/scopes/{unknown_scope}"

    assert (
        _add_sessions(client, test_workspace.name, unknown_scope, [session_name]) == 404
    )
    assert client.post(f"{scope_base}/sessions/list").status_code == 404
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
        assert peer.internal_metadata == {"kind": "scope"}
        assert peer.configuration == {"observe_me": False}

        session_peer = await _get_session_peer(
            db_session, test_workspace.name, session_name, scope_peer_name(scope_name)
        )
        assert session_peer is not None
        assert session_peer.configuration["observe_others"] is True
        assert session_peer.configuration["observe_me"] is False

    # And the memberships show up through the facade
    assert _scope_sessions(client, test_workspace.name, scope_a) == [session_name]


def test_session_create_rejects_prefixed_scope_names(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": str(generate_nanoid()), "scopes": ["scope.x"]},
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
    assert (
        _add_sessions(client, test_workspace.name, scope_name, [scope_session]) == 204
    )

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
    assert _add_sessions(client, test_workspace.name, scope_name, [session_name]) == 204

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


# ---------------------------------------------------------------------------
# Dotted / legacy peer names must not 500 (regression for the PeerCreate-as-DTO
# chokepoint). Names were length-only validated before migration
# d429de0e5338, so pre-existing peers can contain '.' — and re-validating them
# against PeerCreate's charset pattern raised a raw pydantic error, i.e. a 500.
# ---------------------------------------------------------------------------


async def test_legacy_dotted_peer_name_is_fully_usable(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """A pre-existing dotted peer name must work end to end, not 500."""
    test_workspace, _ = sample_data
    legacy_name = f"alice.smith.{generate_nanoid()}"

    db_session.add(models.Peer(workspace_name=test_workspace.name, name=legacy_name))
    await db_session.commit()

    session_name = _create_session(client, test_workspace.name)

    # Can author messages
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/messages",
        json={"messages": [{"peer_id": legacy_name, "content": "hello"}]},
    )
    assert response.status_code in [200, 201], response.text

    # Can be added to a session through the generic route
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
        json={legacy_name: {}},
    )
    assert response.status_code == 200, response.text

    # Can be updated through the generic peer route
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/peers/{legacy_name}",
        json={"metadata": {"k": "v"}},
    )
    assert response.status_code == 200, response.text


async def test_legacy_prefixed_peer_without_flag_is_not_a_scope(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """A peer merely occupying the reserved namespace keeps working.

    Only the name half of the invariant is present, so it is not a scope: it can
    still author messages and shows up in the default peers list, while the
    scopes facade refuses to treat it as a scope.
    """
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    squatter = scope_peer_name(scope_name)

    db_session.add(models.Peer(workspace_name=test_workspace.name, name=squatter))
    await db_session.commit()

    session_name = _create_session(client, test_workspace.name)

    # Not a scope, so the message-author guard must not fire
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/messages",
        json={"messages": [{"peer_id": squatter, "content": "hello"}]},
    )
    assert response.status_code in [200, 201], response.text

    # Visible in the default (non-scope) peers list
    response = client.post(f"/v3/workspaces/{test_workspace.name}/peers/list")
    assert squatter in [p["id"] for p in response.json()["items"]]

    # But the facade does not recognise it
    response = client.get(f"/v3/workspaces/{test_workspace.name}/scopes/{scope_name}")
    assert response.status_code == 404

    response = client.post(f"/v3/workspaces/{test_workspace.name}/scopes/list")
    assert scope_name not in [s["id"] for s in response.json()["items"]]


async def test_forged_configuration_kind_does_not_make_a_scope(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
):
    """`configuration` is user-writable, so it must not be load-bearing.

    A peer that forges `{"kind": "scope"}` in configuration has neither the
    reserved name nor the internal flag, so it stays an ordinary peer.
    """
    test_workspace, _ = sample_data
    peer_name = str(generate_nanoid())

    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/peers/{peer_name}",
        json={"configuration": {"kind": "scope"}},
    )
    assert response.status_code == 200, response.text

    # Still an ordinary peer: present in the default list, absent from scopes
    response = client.post(f"/v3/workspaces/{test_workspace.name}/peers/list")
    assert peer_name in [p["id"] for p in response.json()["items"]]

    response = client.post(f"/v3/workspaces/{test_workspace.name}/scopes/list")
    assert peer_name not in [s["id"] for s in response.json()["items"]]

    # And can still author messages
    session_name = _create_session(client, test_workspace.name)
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/messages",
        json={"messages": [{"peer_id": peer_name, "content": "hello"}]},
    )
    assert response.status_code in [200, 201], response.text


def test_internal_metadata_never_in_peer_response(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """The scope flag must not leak into any peer response body."""
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/list", json={"kind": "all"}
    )
    assert response.status_code == 200
    for peer in response.json()["items"]:
        assert "internal_metadata" not in peer
        assert "kind" not in peer.get("configuration", {})


async def test_crud_get_peer_resolves_scope_and_dotted_names(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """crud.get_peer must accept names outside RESOURCE_NAME_PATTERN.

    This is the Dreamer's preflight path: DreamScheduler passes
    ``collection.observer`` straight through, and scope peers have
    ``observe_others=true``, so ``(scope.x, peer)`` collections exist and get
    dreamt. While get_peer took a PeerCreate, every such dream died at preflight
    on a raw pydantic ValidationError.
    """
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201

    resolved = await crud.get_peer(
        db_session, test_workspace.name, scope_peer_name(scope_name)
    )
    assert resolved.name == scope_peer_name(scope_name)
    assert is_scope_peer(resolved.name, resolved.internal_metadata)

    dotted = f"legacy.dotted.{generate_nanoid()}"
    db_session.add(models.Peer(workspace_name=test_workspace.name, name=dotted))
    await db_session.commit()

    resolved = await crud.get_peer(db_session, test_workspace.name, dotted)
    assert resolved.name == dotted
    # Has neither half of the invariant
    assert not is_scope_peer(resolved.name, resolved.internal_metadata)


# ---------------------------------------------------------------------------
# Name validation on the create path. PeerSpec carries no charset pattern so
# existing names can be *looked up*, but anything crud is about to INSERT is a
# new peer and must obey the public contract — otherwise request-controlled
# names (message authors, session peer maps) mint squatters.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    ["not a valid name!@#", "has spaces", "emoji-\U0001f600"],
)
def test_message_author_cannot_create_invalid_peer_name(
    client: TestClient, sample_data: tuple[Workspace, Peer], bad_name: str
):
    """A message author must not be able to create a non-conforming peer."""
    test_workspace, _ = sample_data
    session_name = _create_session(client, test_workspace.name)

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/messages",
        json={"messages": [{"peer_id": bad_name, "content": "hello"}]},
    )
    assert response.status_code == 422, response.text

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/list", json={"kind": "all"}
    )
    assert bad_name not in [p["id"] for p in response.json()["items"]]


async def test_resolved_scope_peer_rejected_at_membership_upsert(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """The last-line guard runs on resolved rows, closing the check-then-upsert race.

    Calls crud directly, bypassing the route-level name check, so the only thing
    standing between the caller and a scope membership is the guard on the
    resolved peer row — the guard the racing caller would hit. The unflagged →
    flagged transition itself is not simulated here.
    """
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    backing = scope_peer_name(scope_name)
    session_name = _create_session(client, test_workspace.name)

    # crud-level: the resolved row is a scope, so membership must be refused even
    # though the peer already exists (no create-path validation fires).
    with pytest.raises(ValidationException):
        await crud.get_or_create_session(
            db_session,
            session=schemas.SessionCreate(
                name=session_name, peers={backing: schemas.SessionPeerConfig()}
            ),
            workspace_name=test_workspace.name,
        )


async def test_set_peer_config_cannot_disable_a_scope(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """A scope's membership config belongs to the facade, not the caller.

    `observe_others=false` would silently stop all fan-out into the scope, and
    `observe_me=true` would make Honcho form a representation *of* a scope.

    Verified against the row rather than the config route, because that read is
    refused for a scope too — see the policy table.
    """
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    backing = scope_peer_name(scope_name)
    session_name = _create_session(client, test_workspace.name)
    assert _add_sessions(client, test_workspace.name, scope_name, [session_name]) == 204

    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers/{backing}/config",
        json={"observe_others": False, "observe_me": True},
    )
    assert response.status_code == 422, response.text

    # Observer semantics intact
    membership = await _get_session_peer(
        db_session, test_workspace.name, session_name, backing
    )
    assert membership is not None
    assert membership.configuration == {"observe_me": False, "observe_others": True}


def test_session_peers_listing_excludes_scopes(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """The generic session-peer surface must not expose the facade's observer.

    Without the filter this listing returns a peer literally named
    `scope.<name>` carrying `observe_others=true` — the exact mechanic the
    facade exists to hide. The membership-config read is refused for the same
    reason; ordinary members are unaffected by both.
    """
    test_workspace, test_peer = sample_data
    base = f"/v3/workspaces/{test_workspace.name}"
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    backing = scope_peer_name(scope_name)
    session_name = _create_session(client, test_workspace.name)

    assert (
        client.post(
            f"{base}/sessions/{session_name}/peers", json={test_peer.name: {}}
        ).status_code
        == 200
    )
    assert _add_sessions(client, test_workspace.name, scope_name, [session_name]) == 204

    listed = client.get(f"{base}/sessions/{session_name}/peers")
    assert listed.status_code == 200
    names = {item["id"] for item in listed.json()["items"]}
    assert test_peer.name in names
    assert backing not in names

    assert (
        client.get(f"{base}/sessions/{session_name}/peers/{backing}/config").status_code
        == 422
    )
    assert (
        client.get(
            f"{base}/sessions/{session_name}/peers/{test_peer.name}/config"
        ).status_code
        == 200
    )


@pytest.mark.parametrize("bad_name", ["", "a" * 513])
def test_degenerate_peer_names_are_422_not_500(
    client: TestClient, sample_data: tuple[Workspace, Peer], bad_name: str
):
    """Empty and over-long author names must not reach PeerSpec and 500.

    Request-bound peer names carry no length bound of their own, so these used to
    raise a raw pydantic ValidationError inside crud, which the catch-all handler
    turned into a 500.
    """
    test_workspace, _ = sample_data
    session_name = _create_session(client, test_workspace.name)

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/messages",
        json={"messages": [{"peer_id": bad_name, "content": "hello"}]},
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# Detach-by-omission. The replacement routes never name the scope, so there is
# no peer position for the route-policy table to classify — the caller detaches
# by leaving it out. Preservation has to be flag-based, not request-based.
# ---------------------------------------------------------------------------


def test_generic_replacement_preserves_scope_membership(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """`PUT /sessions/{id}/peers` replaces ordinary peers, never scopes.

    The guard cannot key off the request body: a caller detaches a scope by simply
    *omitting* it from an otherwise valid replacement map, never naming it.
    """
    test_workspace, test_peer = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    session_name = _create_session(client, test_workspace.name)
    assert _add_sessions(client, test_workspace.name, scope_name, [session_name]) == 204

    # Replacement naming only an ordinary peer must succeed...
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
        json={test_peer.name: {}},
    )
    assert response.status_code == 200, response.text

    # ...while leaving the scope's membership intact.
    assert _scope_sessions(client, test_workspace.name, scope_name) == [session_name]


async def test_empty_replacement_preserves_scope_membership(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """An empty replacement map clears ordinary peers but not scopes."""
    test_workspace, test_peer = sample_data
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    session_name = _create_session(client, test_workspace.name)
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
        json={test_peer.name: {}},
    )
    _add_sessions(client, test_workspace.name, scope_name, [session_name])

    assert (
        client.put(
            f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
            json={},
        ).status_code
        == 200
    )

    assert _scope_sessions(client, test_workspace.name, scope_name) == [session_name]

    # The other half of the docstring: without this the test passes even if the
    # empty replacement became a no-op for ordinary peers too.
    session_peer = await _get_session_peer(
        db_session, test_workspace.name, session_name, test_peer.name
    )
    assert session_peer is not None
    assert session_peer.left_at is not None


async def test_replacement_still_removes_unflagged_squatter(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """The preservation is flag-based: a squatter keeps ordinary semantics."""
    test_workspace, test_peer = sample_data
    squatter = scope_peer_name(str(generate_nanoid()))
    db_session.add(models.Peer(workspace_name=test_workspace.name, name=squatter))
    await db_session.commit()

    session_name = _create_session(client, test_workspace.name)
    assert (
        client.post(
            f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
            json={squatter: {}},
        ).status_code
        == 200
    )

    assert (
        client.put(
            f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
            json={test_peer.name: {}},
        ).status_code
        == 200
    )
    session_peer = await _get_session_peer(
        db_session, test_workspace.name, session_name, squatter
    )
    assert session_peer is not None
    assert session_peer.left_at is not None, "squatter should be replaced normally"


# ---------------------------------------------------------------------------
# Observed-position guards below the HTTP surface. A scope may be an observer
# but must never be observed — and "observed" includes a reserved name that does
# not yet exist, since nothing on these paths creates the peer and the state
# would retroactively describe the scope once created. The route-level half of
# this is enumerated in test_scope_route_policy.py; these cover the crud entry
# points that no route table reaches, plus the side effects a status code alone
# would not catch.
# ---------------------------------------------------------------------------


async def test_set_peer_card_guard_covers_internal_callers(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """The guard is in crud, so Dreamer and agent-tool paths are covered too."""
    test_workspace, test_peer = sample_data
    with pytest.raises(ValidationException):
        await crud.set_peer_card(
            db_session,
            test_workspace.name,
            peer_card=["x"],
            observer=test_peer.name,
            observed=scope_peer_name(str(generate_nanoid())),
        )


async def test_dream_cannot_be_queued_for_a_future_scope(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """A refused dream leaves no queue row behind.

    The 422 itself is enumerated in test_scope_route_policy.py; what that cannot
    see is the side effect. The route's own precheck cannot catch this — the peer
    is not flagged yet — so the check has to sit in the transaction that inserts
    the queue item, and a check placed after the insert would still 422.
    """
    test_workspace, test_peer = sample_data
    future = scope_peer_name(str(generate_nanoid()))

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/schedule_dream",
        json={"observer": test_peer.name, "observed": future, "dream_type": "omni"},
    )
    assert response.status_code == 422, response.text

    # No queue row was inserted for it
    items = (
        await db_session.execute(
            select(QueueItem).where(QueueItem.work_unit_key.contains(future))
        )
    ).all()
    assert not items


async def test_enqueue_dream_guard_covers_internal_callers(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Direct enqueue_dream calls enforce the same invariant as the route."""
    from src.deriver.enqueue import enqueue_dream
    from src.schemas.configuration import DreamType

    test_workspace, test_peer = sample_data
    scope_name = str(generate_nanoid())
    await db_session.commit()

    with pytest.raises(ValidationException):
        await enqueue_dream(
            test_workspace.name,
            observer=test_peer.name,
            observed=scope_peer_name(scope_name),
            dream_type=DreamType.OMNI,
        )


def test_prefixed_nul_name_is_422_not_500(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """A reserved-prefix name containing NUL must not reach a text comparison.

    It passes the request schemas and PeerSpec, so without pre-SQL rejection it
    reaches psycopg inside the scope lookup and raises DataError — a 500.
    """
    test_workspace, _ = sample_data
    session_name = _create_session(client, test_workspace.name)

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/messages",
        json={"messages": [{"peer_id": "scope.future\x00name", "content": "hi"}]},
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# Observer limit. Scope memberships carry observe_others=True but must not
# consume SESSION_OBSERVERS_LIMIT: that would cap scopes-per-session at the
# limit and report it as an observer-shaped 400 through a facade whose whole
# job is hiding observers.
# ---------------------------------------------------------------------------


def test_scopes_do_not_count_toward_observer_limit(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """A session can join more scopes than SESSION_OBSERVERS_LIMIT allows observers."""
    test_workspace, _ = sample_data
    session_name = _create_session(client, test_workspace.name)
    scope_names = [
        str(generate_nanoid()) for _ in range(settings.SESSION_OBSERVERS_LIMIT + 2)
    ]

    for scope_name in scope_names:
        assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
        assert (
            _add_sessions(client, test_workspace.name, scope_name, [session_name])
            == 204
        )

    # Every membership is live, and the session-create path agrees.
    for scope_name in scope_names:
        assert _scope_sessions(client, test_workspace.name, scope_name) == [
            session_name
        ]

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": str(generate_nanoid()), "scopes": scope_names},
    )
    assert response.status_code == 201, response.text


def test_observer_limit_still_applies_to_real_peers(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """The exclusion is scope-only: real observers are still capped.

    Without this the scope carve-out could quietly disable the limit entirely.
    """
    test_workspace, _ = sample_data
    session_name = _create_session(client, test_workspace.name)
    scope_name = str(generate_nanoid())
    assert _create_scope(client, test_workspace.name, scope_name).status_code == 201
    assert _add_sessions(client, test_workspace.name, scope_name, [session_name]) == 204

    observers = {
        str(generate_nanoid()): {"observe_others": True}
        for _ in range(settings.SESSION_OBSERVERS_LIMIT + 1)
    }
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
        json=observers,
    )
    assert response.status_code == 400, response.text


def test_session_create_scopes_list_is_capped(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """`scopes` is bounded like `session_ids` on the add-sessions route.

    Nothing downstream bounds it now that scopes are outside the observer limit,
    so an unbounded list would mint a scope peer per element in one request.
    """
    test_workspace, _ = sample_data

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": str(generate_nanoid()),
            "scopes": [str(generate_nanoid()) for _ in range(101)],
        },
    )
    assert response.status_code == 422, response.text
