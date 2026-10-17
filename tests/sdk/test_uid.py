"""Tests for the server-assigned ``uid`` on workspaces, peers and sessions."""

from typing import cast

import pytest

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_workspace_uid(client_fixture: tuple[Honcho, str]) -> None:
    """The client learns its workspace uid from its first API call."""
    honcho_client, client_type = client_fixture

    assert honcho_client.uid is None
    if client_type == "async":
        await honcho_client.aio.get_metadata()
    else:
        honcho_client.get_metadata()
    uid = honcho_client.uid
    assert uid
    assert uid != honcho_client.workspace_id


@pytest.mark.asyncio
async def test_peer_uid(client_fixture: tuple[Honcho, str]) -> None:
    """Peers carry the same uid from get-or-create, list and refresh."""
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-uid-peer")
        listed = cast(list[Peer], (await honcho_client.aio.peers()).items)
        unfetched = Peer("test-uid-peer", honcho_client)
        assert unfetched.uid is None
        await unfetched.aio.refresh()
    else:
        peer = honcho_client.peer(id="test-uid-peer")
        listed = cast(list[Peer], list(honcho_client.peers()))
        unfetched = Peer("test-uid-peer", honcho_client)
        assert unfetched.uid is None
        unfetched.refresh()

    assert peer.uid
    assert peer.uid != peer.id
    assert {p.id: p.uid for p in listed}[peer.id] == peer.uid
    assert unfetched.uid == peer.uid
    assert honcho_client.uid  # peer calls ensure the workspace first


@pytest.mark.asyncio
async def test_session_uid(client_fixture: tuple[Honcho, str]) -> None:
    """Sessions carry their uid; session peers carry theirs; clones get a new one."""
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-uid-session-peer")
        session = await honcho_client.aio.session(id="test-uid-session")
        await session.aio.add_peers(peer)
        listed = cast(list[Session], (await honcho_client.aio.sessions()).items)
        session_peers = await session.aio.peers()
        clone = await session.aio.clone()
    else:
        peer = honcho_client.peer(id="test-uid-session-peer")
        session = honcho_client.session(id="test-uid-session")
        session.add_peers(peer)
        listed = cast(list[Session], list(honcho_client.sessions()))
        session_peers = session.peers()
        clone = session.clone()

    assert session.uid
    assert session.uid != session.id
    assert {s.id: s.uid for s in listed}[session.id] == session.uid
    assert [p.uid for p in session_peers] == [peer.uid]
    assert clone.uid
    assert clone.uid != session.uid
