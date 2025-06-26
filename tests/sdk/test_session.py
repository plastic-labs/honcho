import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.async_client.session import (
    AsyncSession,
)
from sdks.python.src.honcho.async_client.session import (
    SessionPeerConfig as AsyncSessionPeerConfig,
)
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session, SessionPeerConfig


@pytest.mark.asyncio
async def test_session_metadata(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests creation and metadata operations for sessions.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-session-meta")
        assert isinstance(session, AsyncSession)

        metadata = await session.get_metadata()
        assert metadata == {}

        await session.set_metadata({"foo": "bar"})
        metadata = await session.get_metadata()
        assert metadata == {"foo": "bar"}
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-session-meta")
        assert isinstance(session, Session)

        metadata = session.get_metadata()
        assert metadata == {}

        session.set_metadata({"foo": "bar"})
        metadata = session.get_metadata()
        assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_session_peer_management(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests adding, setting, getting, and removing peers from a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-session-peers")
        assert isinstance(session, AsyncSession)
        peer1 = await honcho_client.peer(id="p1")
        assert isinstance(peer1, AsyncPeer)
        peer2 = await honcho_client.peer(id="p2")
        assert isinstance(peer2, AsyncPeer)
        peer3 = await honcho_client.peer(id="p3")
        assert isinstance(peer3, AsyncPeer)

        await session.add_peers([peer1, peer2])
        peers = await session.get_peers()
        assert len(peers) == 2
        peer_ids = {p.id for p in peers}
        assert "p1" in peer_ids and "p2" in peer_ids

        await session.set_peers([peer2, peer3])
        peers = await session.get_peers()
        assert len(peers) == 2
        peer_ids = {p.id for p in peers}
        assert "p2" in peer_ids and "p3" in peer_ids

        await session.remove_peers([peer2])
        peers = await session.get_peers()
        assert len(peers) == 1
        assert peers[0].id == "p3"
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-session-peers")
        assert isinstance(session, Session)
        peer1 = honcho_client.peer(id="p1")
        assert isinstance(peer1, Peer)
        peer2 = honcho_client.peer(id="p2")
        assert isinstance(peer2, Peer)
        peer3 = honcho_client.peer(id="p3")
        assert isinstance(peer3, Peer)

        session.add_peers([peer1, peer2])
        peers = session.get_peers()
        assert len(peers) == 2
        peer_ids = {p.id for p in peers}
        assert "p1" in peer_ids and "p2" in peer_ids

        session.set_peers([peer2, peer3])
        peers = session.get_peers()
        assert len(peers) == 2
        peer_ids = {p.id for p in peers}
        assert "p2" in peer_ids and "p3" in peer_ids

        session.remove_peers([peer2])
        peers = session.get_peers()
        assert len(peers) == 1
        assert peers[0].id == "p3"


@pytest.mark.asyncio
async def test_session_peer_config(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting and setting peer configurations in a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        config = AsyncSessionPeerConfig(observe_others=False, observe_me=False)
        session = await honcho_client.session(id="test-session-config")
        assert isinstance(session, AsyncSession)
        peer = await honcho_client.peer(id="p-config")
        assert isinstance(peer, AsyncPeer)
        await session.add_peers([(peer, config)])

        retrieved_config = await session.get_peer_config(peer)
        assert retrieved_config.observe_me is False
        assert retrieved_config.observe_others is False

        await session.set_peer_config(
            peer, AsyncSessionPeerConfig(observe_others=True, observe_me=True)
        )
        retrieved_config = await session.get_peer_config(peer)
        assert retrieved_config.observe_me is True
        assert retrieved_config.observe_others is True
    else:
        assert isinstance(honcho_client, Honcho)
        config = SessionPeerConfig(observe_others=False, observe_me=False)
        session = honcho_client.session(id="test-session-config")
        assert isinstance(session, Session)
        peer = honcho_client.peer(id="p-config")
        assert isinstance(peer, Peer)
        session.add_peers([(peer, config)])

        retrieved_config = session.get_peer_config(peer)
        assert retrieved_config.observe_me is False
        assert retrieved_config.observe_others is False

        session.set_peer_config(
            peer, SessionPeerConfig(observe_others=True, observe_me=True)
        )
        retrieved_config = session.get_peer_config(peer)
        assert retrieved_config.observe_me
        assert retrieved_config.observe_others


@pytest.mark.asyncio
async def test_session_messages(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests adding and getting messages from a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-session-msg")
        assert isinstance(session, AsyncSession)
        user = await honcho_client.peer(id="user-msg")
        assert isinstance(user, AsyncPeer)
        assistant = await honcho_client.peer(id="assistant-msg")
        assert isinstance(assistant, AsyncPeer)

        await session.add_messages(
            [
                user.message("Hello assistant"),
                assistant.message("Hello user"),
            ]
        )
        messages_page = await session.get_messages()
        messages = messages_page.items
        assert len(messages) == 2

        messages_page = await session.get_messages(filters={"peer_id": user.id})
        messages = messages_page.items
        assert len(messages) == 1
        assert messages[0].content == "Hello assistant"
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-session-msg")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="user-msg")
        assert isinstance(user, Peer)
        assistant = honcho_client.peer(id="assistant-msg")
        assert isinstance(assistant, Peer)

        session.add_messages(
            [
                user.message("Hello assistant"),
                assistant.message("Hello user"),
            ]
        )
        messages_page = session.get_messages()
        messages = list(messages_page)
        assert len(messages) == 2

        messages_page = session.get_messages(filters={"peer_id": user.id})
        messages = list(messages_page)
        assert len(messages) == 1
        assert messages[0].content == "Hello assistant"


@pytest.mark.asyncio
async def test_session_get_context(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting the context of a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-session-ctx")
        assert isinstance(session, AsyncSession)
        user = await honcho_client.peer(id="user-ctx")
        assert isinstance(user, AsyncPeer)
        await session.add_messages([user.message("This is a context test.")])
        context = await session.get_context()
        assert len(context.messages) == 1
        assert "context test" in context.messages[0].content
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-session-ctx")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="user-ctx")
        assert isinstance(user, Peer)
        session.add_messages([user.message("This is a context test.")])
        context = session.get_context()
        assert len(context.messages) == 1
        assert "context test" in context.messages[0].content


@pytest.mark.asyncio
async def test_session_search(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests searching for messages in a session.
    """
    honcho_client, client_type = client_fixture
    search_query = "a unique message for session search"

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="search-session-s")
        assert isinstance(session, AsyncSession)
        user = await honcho_client.peer(id="search-user-s")
        assert isinstance(user, AsyncPeer)
        await session.add_messages([user.message(search_query)])

        search_results = await session.search(search_query)
        results = search_results.items
        assert len(results) >= 1
        assert search_query in results[0].content
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="search-session-s")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="search-user-s")
        assert isinstance(user, Peer)
        session.add_messages([user.message(search_query)])

        search_results = session.search(search_query)
        results = list(search_results)
        assert len(results) >= 1
        assert search_query in results[0].content


@pytest.mark.asyncio
async def test_session_working_rep(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting the working representation of a peer in a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-session-wr")
        assert isinstance(session, AsyncSession)
        peer = await honcho_client.peer(id="peer-wr")
        assert isinstance(peer, AsyncPeer)
        await session.add_messages([peer.message("test message for working rep")])
        _working_rep = await session.working_rep(peer)
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-session-wr")
        assert isinstance(session, Session)
        peer = honcho_client.peer(id="peer-wr")
        assert isinstance(peer, Peer)
        session.add_messages([peer.message("test message for working rep")])
        _working_rep = session.working_rep(peer)
