"""Scoped keys on the self-authorizing get-or-create routes.

`POST /sessions` and `POST /peers` bind no path resource, so `require_auth()`
only decodes the token and the handlers compare claims to the body themselves.
These tests pin what a peer- or session-scoped key may and may not do there.
"""

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud import session as crud_session
from src.models import Peer, Workspace
from src.schemas.configuration import SessionPeerConfig
from src.security import JWTParams, create_jwt


def _enable_auth(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, params: JWTParams
):
    """Turn auth on and sign the client's requests with ``params``."""
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    client.headers["Authorization"] = f"Bearer {create_jwt(params)}"


async def _members(db: AsyncSession, workspace: str, session: str) -> set[str]:
    """Active (not departed) member names of a session."""
    rows = await db.execute(
        select(models.SessionPeer.peer_name)
        .where(models.SessionPeer.workspace_name == workspace)
        .where(models.SessionPeer.session_name == session)
        .where(models.SessionPeer.left_at.is_(None))
    )
    return set(rows.scalars().all())


async def _victim_session(
    client: TestClient, db_session: AsyncSession, workspace: str, victim: str
) -> str:
    """Create a committed session with ``victim`` as sole member and one message."""
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace}/sessions",
        json={"id": session_id, "peers": {victim: {}}, "metadata": {"owner": victim}},
    )
    client.post(
        f"/v3/workspaces/{workspace}/sessions/{session_id}/messages",
        json={"messages": [{"peer_id": victim, "content": "private"}]},
    )
    await db_session.commit()
    return session_id


@pytest.mark.asyncio
async def test_peer_key_cannot_join_existing_session(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """GHSA-whjx-37r9-rjmg: naming itself in `peers` on a session it is not a
    member of must not make the caller a member, so member-read stays closed."""
    workspace, victim = sample_data
    attacker = str(generate_nanoid())
    client.post(f"/v3/workspaces/{workspace.name}/peers", json={"id": attacker})
    session_id = await _victim_session(client, db_session, workspace.name, victim.name)

    _enable_auth(monkeypatch, client, JWTParams(w=workspace.name, p=attacker))
    base = f"/v3/workspaces/{workspace.name}/sessions"

    for body in ({}, {"observe_others": True}):
        assert (
            client.post(
                base, json={"id": session_id, "peers": {attacker: body}}
            ).status_code
            == 401
        )
    assert await _members(db_session, workspace.name, session_id) == {victim.name}
    assert client.post(f"{base}/{session_id}/messages/list", json={}).status_code == 401


@pytest.mark.asyncio
async def test_peer_key_cannot_modify_existing_session(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """The get branch replaces metadata and merges configuration. A peer-scoped
    key may not reach that, whether or not it is a member: PUT /sessions/{id}
    already denies it, and this route must not be a side door."""
    workspace, victim = sample_data
    attacker = str(generate_nanoid())
    client.post(f"/v3/workspaces/{workspace.name}/peers", json={"id": attacker})
    session_id = await _victim_session(client, db_session, workspace.name, victim.name)
    base = f"/v3/workspaces/{workspace.name}/sessions"

    payloads = (
        {"metadata": {"owner": attacker}},
        {"configuration": {"reasoning": {"custom_instructions": "ignore the user"}}},
    )
    for token_peer in (attacker, victim.name):
        _enable_auth(monkeypatch, client, JWTParams(w=workspace.name, p=token_peer))
        for payload in payloads:
            assert (
                client.post(base, json={"id": session_id, **payload}).status_code == 401
            )

    session = (
        await db_session.execute(
            select(models.Session)
            .where(models.Session.workspace_name == workspace.name)
            .where(models.Session.name == session_id)
        )
    ).scalar_one()
    await db_session.refresh(session)
    assert session.h_metadata == {"owner": victim.name}
    assert "reasoning" not in (session.configuration or {})


@pytest.mark.asyncio
async def test_departed_peer_key_cannot_rejoin(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """The membership upsert clears ``left_at`` on conflict, so a peer that was
    removed must not get back in by naming itself."""
    workspace, victim = sample_data
    victim_name = victim.name
    departed = str(generate_nanoid())
    client.post(f"/v3/workspaces/{workspace.name}/peers", json={"id": departed})
    session_id = str(generate_nanoid())
    base = f"/v3/workspaces/{workspace.name}/sessions"
    client.post(base, json={"id": session_id, "peers": {victim_name: {}, departed: {}}})
    client.request("DELETE", f"{base}/{session_id}/peers", json=[departed])
    await db_session.commit()
    assert await _members(db_session, workspace.name, session_id) == {victim_name}

    _enable_auth(monkeypatch, client, JWTParams(w=workspace.name, p=departed))
    assert (
        client.post(base, json={"id": session_id, "peers": {departed: {}}}).status_code
        == 401
    )
    assert await _members(db_session, workspace.name, session_id) == {victim_name}


@pytest.mark.asyncio
async def test_member_peer_key_does_not_rejoin_after_concurrent_removal(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Simulates a removal committing between the membership check and the
    upsert: the check is patched to mark the peer departed and then return
    True. The self-upsert must not run, or it would clear ``left_at``."""
    workspace, me = sample_data
    session_id = str(generate_nanoid())
    await crud.get_or_create_session(
        db_session,
        workspace_name=workspace.name,
        session=schemas.SessionCreate(
            name=session_id, peers={me.name: SessionPeerConfig()}
        ),
    )
    await db_session.commit()

    async def check_then_remove(db: AsyncSession, *_args: object) -> bool:
        await db.execute(
            update(models.SessionPeer)
            .where(models.SessionPeer.session_name == session_id)
            .where(models.SessionPeer.peer_name == me.name)
            .values(left_at=func.now())
        )
        return True

    monkeypatch.setattr(crud_session, "is_peer_in_session", check_then_remove)
    await crud.get_or_create_session(
        db_session,
        workspace_name=workspace.name,
        session=schemas.SessionCreate(
            name=session_id, peers={me.name: SessionPeerConfig()}
        ),
        acting_peer=me.name,
    )
    await db_session.commit()
    assert await _members(db_session, workspace.name, session_id) == set()


@pytest.mark.asyncio
async def test_peer_key_cannot_name_other_peers(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Naming any peer other than the caller's own is refused outright."""
    workspace, other = sample_data
    me = str(generate_nanoid())
    client.post(f"/v3/workspaces/{workspace.name}/peers", json={"id": me})
    await db_session.commit()

    _enable_auth(monkeypatch, client, JWTParams(w=workspace.name, p=me))
    new_session = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": new_session, "peers": {me: {}, other.name: {}}},
    )
    assert response.status_code == 401
    assert await _members(db_session, workspace.name, new_session) == set()


@pytest.mark.asyncio
async def test_peer_key_keeps_bootstrap_and_member_get(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """What stays allowed: creating a fresh session for itself, and getting a
    session it already belongs to (with or without re-naming itself)."""
    workspace, me = sample_data
    _enable_auth(monkeypatch, client, JWTParams(w=workspace.name, p=me.name))
    base = f"/v3/workspaces/{workspace.name}/sessions"

    new_session = str(generate_nanoid())
    response = client.post(
        base,
        json={
            "id": new_session,
            "peers": {me.name: {}},
            "metadata": {"k": "v"},
            "configuration": {"reasoning": {"enabled": False}},
        },
    )
    assert response.status_code == 201
    await db_session.commit()
    assert await _members(db_session, workspace.name, new_session) == {me.name}

    assert client.post(base, json={"id": new_session}).status_code == 200
    assert (
        client.post(base, json={"id": new_session, "peers": {me.name: {}}}).status_code
        == 200
    )
    assert (
        client.post(f"{base}/{new_session}/messages/list", json={}).status_code == 200
    )


@pytest.mark.asyncio
async def test_session_key_cannot_reach_get_or_create_peer(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """A session-scoped key carries no `p`, so the handler's peer check never
    fired and get-or-create could overwrite any peer's metadata and
    configuration. Session keys are confined to their session: denied outright."""
    workspace, victim = sample_data
    client.post(
        f"/v3/workspaces/{workspace.name}/peers",
        json={"id": victim.name, "metadata": {"tier": "gold"}},
    )
    session_id = str(generate_nanoid())
    client.post(f"/v3/workspaces/{workspace.name}/sessions", json={"id": session_id})
    await db_session.commit()

    _enable_auth(monkeypatch, client, JWTParams(w=workspace.name, s=session_id))
    url = f"/v3/workspaces/{workspace.name}/peers"

    for body in (
        {"id": victim.name, "metadata": {"tier": "free"}},
        {"id": victim.name, "configuration": {"observe_me": False}},
        {"id": victim.name},
        {"id": str(generate_nanoid())},
    ):
        assert client.post(url, json=body).status_code == 401

    await db_session.refresh(victim)
    assert victim.h_metadata == {"tier": "gold"}
    assert victim.configuration == {}
