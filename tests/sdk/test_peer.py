import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.peer import Peer


@pytest.mark.asyncio
async def test_peer_metadata(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests creation and metadata operations for peers.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-meta")
        assert isinstance(peer, AsyncPeer)

        metadata = await peer.get_metadata()
        assert metadata == {}

        await peer.set_metadata({"foo": "bar"})
        metadata = await peer.get_metadata()
        assert metadata == {"foo": "bar"}
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-meta")
        assert isinstance(peer, Peer)

        metadata = peer.get_metadata()
        assert metadata == {}

        peer.set_metadata({"foo": "bar"})
        metadata = peer.get_metadata()
        assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_peer_get_sessions(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests retrieving the sessions a peer is a member of.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-sessions")
        session1 = await honcho_client.session(id="s1")
        session2 = await honcho_client.session(id="s2")

        await session1.add_peers(peer)
        await session2.add_peers(peer)

        sessions_page = await peer.get_sessions()
        sessions = sessions_page.items
        assert len(sessions) == 2
        session_ids = {s.id for s in sessions}
        assert "s1" in session_ids
        assert "s2" in session_ids
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-sessions")
        session1 = honcho_client.session(id="s1")
        session2 = honcho_client.session(id="s2")

        session1.add_peers(peer)
        session2.add_peers(peer)

        sessions_page = peer.get_sessions()
        sessions = list(sessions_page)
        assert len(sessions) == 2
        session_ids = {s.id for s in sessions}
        assert "s1" in session_ids
        assert "s2" in session_ids
