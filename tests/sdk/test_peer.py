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
async def test_peer_add_and_get_messages(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests adding and getting messages from a peer's global representation.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-gms")
        assert isinstance(peer, AsyncPeer)

        await peer.add_messages("a simple string message")
        message_obj = peer.message("a message object")
        await peer.add_messages(message_obj)
        await peer.add_messages([peer.message("a message in a list")])

        messages_page = await peer.get_messages()
        messages = messages_page.items
        assert len(messages) == 3
        contents = {m.content for m in messages}
        assert "a simple string message" in contents
        assert "a message object" in contents
        assert "a message in a list" in contents
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-gms")
        assert isinstance(peer, Peer)

        peer.add_messages("a simple string message")
        message_obj = peer.message("a message object")
        peer.add_messages(message_obj)
        peer.add_messages([peer.message("a message in a list")])

        messages_page = peer.get_messages()
        messages = list(messages_page)
        assert len(messages) == 3
        contents = {m.content for m in messages}
        assert "a simple string message" in contents
        assert "a message object" in contents
        assert "a message in a list" in contents


@pytest.mark.asyncio
async def test_peer_chat(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests the chat functionality of a peer, including target and session scoping.
    """
    honcho_client, client_type = client_fixture
    question = "What is my name?"
    answer = "Your name is test-peer-chat"

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-chat")
        assert isinstance(peer, AsyncPeer)
        await peer.add_messages(answer)

        _response = await peer.chat(question)

        # Test target
        target_peer = await honcho_client.peer(id="target-peer-chat")
        _response = await peer.chat(
            "Does the assistant know my name?", target=target_peer
        )

        # Test session_id
        session = await honcho_client.session(id="chat-session-scope")
        await session.add_messages(peer.message(answer))
        _response = await peer.chat(question, session_id=session.id)

    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-chat")
        assert isinstance(peer, Peer)
        peer.add_messages(answer)

        _response = peer.chat(question)

        # Test target
        target_peer = honcho_client.peer(id="target-peer-chat")
        _response = peer.chat("Does the assistant know my name?", target=target_peer)

        # Test session_id
        session = honcho_client.session(id="chat-session-scope")
        session.add_messages(peer.message(answer))
        _response = peer.chat(question, session_id=session.id)


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


@pytest.mark.asyncio
async def test_peer_search(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests searching for messages in a peer's global representation.
    """
    honcho_client, client_type = client_fixture
    search_query = "a unique message for peer search"

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="search-peer")
        assert isinstance(peer, AsyncPeer)
        await peer.add_messages(search_query)

        search_results = await peer.search(search_query)
        results = search_results.items
        assert len(results) >= 1
        assert search_query in results[0].content
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="search-peer")
        assert isinstance(peer, Peer)
        peer.add_messages(search_query)

        search_results = peer.search(search_query)
        results = list(search_results)
        assert len(results) >= 1
        assert search_query in results[0].content
