from unittest.mock import AsyncMock, patch

import pytest

from sdks.python.src.honcho.api_types import QueueStatusResponse
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.message import Message
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session, SessionPeerConfig


@pytest.mark.asyncio
async def test_session_metadata(client_fixture: tuple[Honcho, str]):
    """
    Tests creation and metadata operations for sessions.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-meta")
        assert isinstance(session, Session)

        metadata = await session.aio.get_metadata()
        assert metadata == {}

        await session.aio.set_metadata({"foo": "bar"})
        metadata = await session.aio.get_metadata()
        assert metadata == {"foo": "bar"}
    else:
        session = honcho_client.session(id="test-session-meta")
        assert isinstance(session, Session)

        metadata = session.get_metadata()
        assert metadata == {}

        session.set_metadata({"foo": "bar"})
        metadata = session.get_metadata()
        assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_session_peer_management(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests adding, setting, getting, and removing peers from a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-peers")
        assert isinstance(session, Session)
        peer1 = await honcho_client.aio.peer(id="p1")
        assert isinstance(peer1, Peer)
        peer2 = await honcho_client.aio.peer(id="p2")
        assert isinstance(peer2, Peer)
        peer3 = await honcho_client.aio.peer(id="p3")
        assert isinstance(peer3, Peer)

        await session.aio.add_peers([peer1, peer2])
        peers = await session.aio.peers()
        assert len(peers) == 2
        peer_ids = {p.id for p in peers}
        assert "p1" in peer_ids and "p2" in peer_ids

        await session.aio.set_peers([peer2, peer3])
        peers = await session.aio.peers()
        assert len(peers) == 2
        peer_ids = {p.id for p in peers}
        assert "p2" in peer_ids and "p3" in peer_ids

        await session.aio.remove_peers([peer2])
        peers = await session.aio.peers()
        assert len(peers) == 1
        assert peers[0].id == "p3"
    else:
        session = honcho_client.session(id="test-session-peers")
        assert isinstance(session, Session)
        peer1 = honcho_client.peer(id="p1")
        assert isinstance(peer1, Peer)
        peer2 = honcho_client.peer(id="p2")
        assert isinstance(peer2, Peer)
        peer3 = honcho_client.peer(id="p3")
        assert isinstance(peer3, Peer)

        session.add_peers([peer1, peer2])
        peers = session.peers()
        assert len(peers) == 2
        peer_ids = {p.id for p in peers}
        assert "p1" in peer_ids and "p2" in peer_ids

        session.set_peers([peer2, peer3])
        peers = session.peers()
        assert len(peers) == 2
        peer_ids = {p.id for p in peers}
        assert "p2" in peer_ids and "p3" in peer_ids

        session.remove_peers([peer2])
        peers = session.peers()
        assert len(peers) == 1
        assert peers[0].id == "p3"


@pytest.mark.asyncio
async def test_session_peer_config(client_fixture: tuple[Honcho, str]):
    """
    Tests getting and setting peer configurations in a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        config = SessionPeerConfig(observe_others=False, observe_me=False)
        session = await honcho_client.aio.session(id="test-session-config")
        assert isinstance(session, Session)
        peer = await honcho_client.aio.peer(id="p-config")
        assert isinstance(peer, Peer)
        await session.aio.add_peers([(peer, config)])

        retrieved_config = await session.aio.peer_config(peer)
        assert retrieved_config.observe_me is False
        assert retrieved_config.observe_others is False

        await session.aio.set_peer_config(
            peer, SessionPeerConfig(observe_others=True, observe_me=True)
        )
        retrieved_config = await session.aio.peer_config(peer)
        assert retrieved_config.observe_me is True
        assert retrieved_config.observe_others is True
    else:
        config = SessionPeerConfig(observe_others=False, observe_me=False)
        session = honcho_client.session(id="test-session-config")
        assert isinstance(session, Session)
        peer = honcho_client.peer(id="p-config")
        assert isinstance(peer, Peer)
        session.add_peers([(peer, config)])

        retrieved_config = session.peer_config(peer)
        assert retrieved_config.observe_me is False
        assert retrieved_config.observe_others is False

        session.set_peer_config(
            peer, SessionPeerConfig(observe_others=True, observe_me=True)
        )
        retrieved_config = session.peer_config(peer)
        assert retrieved_config.observe_me
        assert retrieved_config.observe_others


@pytest.mark.asyncio
async def test_session_messages(client_fixture: tuple[Honcho, str]):
    """
    Tests adding and getting messages from a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-msg")
        assert isinstance(session, Session)
        user = await honcho_client.aio.peer(id="user-msg")
        assert isinstance(user, Peer)
        assistant = await honcho_client.aio.peer(id="assistant-msg")
        assert isinstance(assistant, Peer)

        await session.aio.add_messages(
            [
                user.message("Hello assistant"),
                assistant.message("Hello user"),
            ]
        )
        messages_page = await session.aio.messages()
        messages = messages_page.items
        assert len(messages) == 2

        messages_page = await session.aio.messages(filters={"peer_id": user.id})
        messages = messages_page.items
        assert len(messages) == 1
        assert messages[0].content == "Hello assistant"
    else:
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
        messages_page = session.messages()
        messages = list(messages_page)
        assert len(messages) == 2

        messages_page = session.messages(filters={"peer_id": user.id})
        messages = list(messages_page)
        assert len(messages) == 1
        assert messages[0].content == "Hello assistant"


@pytest.mark.asyncio
async def test_session_context(client_fixture: tuple[Honcho, str]):
    """
    Tests getting the context of a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-ctx")
        assert isinstance(session, Session)
        user = await honcho_client.aio.peer(id="user-ctx")
        assert isinstance(user, Peer)
        await session.aio.add_messages([user.message("This is a context test.")])
        context = await session.aio.context()
        assert len(context.messages) == 1
        assert "context test" in context.messages[0].content
    else:
        session = honcho_client.session(id="test-session-ctx")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="user-ctx")
        assert isinstance(user, Peer)
        session.add_messages([user.message("This is a context test.")])
        context = session.context()
        assert len(context.messages) == 1
        assert "context test" in context.messages[0].content


@pytest.mark.asyncio
async def test_session_search(client_fixture: tuple[Honcho, str]):
    """
    Tests searching for messages in a session.
    """
    honcho_client, client_type = client_fixture
    search_query = "a unique message for session search"

    if client_type == "async":
        session = await honcho_client.aio.session(id="search-session-s")
        assert isinstance(session, Session)
        user = await honcho_client.aio.peer(id="search-user-s")
        assert isinstance(user, Peer)
        await session.aio.add_messages([user.message(search_query)])

        search_results = await session.aio.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1
        assert search_query in search_results[0].content
    else:
        session = honcho_client.session(id="search-session-s")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="search-user-s")
        assert isinstance(user, Peer)
        session.add_messages([user.message(search_query)])

        search_results = session.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1
        assert search_query in search_results[0].content


@pytest.mark.asyncio
async def test_session_add_messages_return_value(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests that add_messages returns a list of Message objects.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-add-msg-return")
        assert isinstance(session, Session)
        user = await honcho_client.aio.peer(id="user-add-msg-return")
        assert isinstance(user, Peer)
        assistant = await honcho_client.aio.peer(id="assistant-add-msg-return")
        assert isinstance(assistant, Peer)

        # Test single message return value

        result = await session.aio.add_messages(user.message("Hello assistant"))
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Message)
        assert result[0].content == "Hello assistant"
        assert result[0].peer_id == user.id

        # Test multiple messages return value
        result = await session.aio.add_messages(
            [
                user.message("How are you?"),
                assistant.message("I'm doing well, thank you!"),
            ]
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(msg, Message) for msg in result)
        assert result[0].content == "How are you?"
        assert result[0].peer_id == user.id
        assert result[1].content == "I'm doing well, thank you!"
        assert result[1].peer_id == assistant.id
    else:
        session = honcho_client.session(id="test-session-add-msg-return")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="user-add-msg-return")
        assert isinstance(user, Peer)
        assistant = honcho_client.peer(id="assistant-add-msg-return")
        assert isinstance(assistant, Peer)

        # Test single message return value

        result = session.add_messages(user.message("Hello assistant"))
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Message)
        assert result[0].content == "Hello assistant"
        assert result[0].peer_id == user.id

        # Test multiple messages return value
        result = session.add_messages(
            [
                user.message("How are you?"),
                assistant.message("I'm doing well, thank you!"),
            ]
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(msg, Message) for msg in result)
        assert result[0].content == "How are you?"
        assert result[0].peer_id == user.id
        assert result[1].content == "I'm doing well, thank you!"
        assert result[1].peer_id == assistant.id


@pytest.mark.asyncio
async def test_session_representation(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests getting the working representation of a peer in a session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-wr")
        assert isinstance(session, Session)
        peer = await honcho_client.aio.peer(id="peer-wr")
        assert isinstance(peer, Peer)
        await session.aio.add_messages([peer.message("test message for working rep")])
        await session.aio.representation(peer)
    else:
        session = honcho_client.session(id="test-session-wr")
        assert isinstance(session, Session)
        peer = honcho_client.peer(id="peer-wr")
        assert isinstance(peer, Peer)
        session.add_messages([peer.message("test message for working rep")])
        session.representation(peer)


@pytest.mark.asyncio
async def test_session_delete(client_fixture: tuple[Honcho, str]) -> None:
    """
    Tests deleting a session and verifying all associated data is removed.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-delete")
        assert isinstance(session, Session)

        # Add a peer and messages to make the session have data
        user = await honcho_client.aio.peer(id="user-delete")
        await session.aio.add_peers([user])
        await session.aio.add_messages(
            [user.message("Test message that should be deleted")]
        )

        # Verify messages exist before deletion
        messages_page = await session.aio.messages()
        messages = messages_page.items
        assert len(messages) == 1

        # Delete should not raise an exception
        await session.aio.delete()

        # Verify session is removed from active sessions list
        all_sessions_page = await honcho_client.aio.sessions({"is_active": True})
        all_sessions = all_sessions_page.items
        all_session_ids = [s.id for s in all_sessions]
        assert "test-session-delete" not in all_session_ids

        # Verify session is also removed from all sessions (hard delete, not soft)
        all_sessions_page = await honcho_client.aio.sessions()
        all_sessions = all_sessions_page.items
        all_session_ids = [s.id for s in all_sessions]
        assert "test-session-delete" not in all_session_ids
    else:
        session = honcho_client.session(id="test-session-delete")
        assert isinstance(session, Session)

        # Add a peer and messages to make the session have data
        user = honcho_client.peer(id="user-delete")
        session.add_peers([user])
        session.add_messages([user.message("Test message that should be deleted")])

        # Verify messages exist before deletion
        messages_page = session.messages()
        messages = list(messages_page)
        assert len(messages) == 1

        # Delete should not raise an exception
        session.delete()

        # Verify session is removed from active sessions list
        all_sessions_page = honcho_client.sessions({"is_active": True})
        all_sessions = list(all_sessions_page)
        all_session_ids = [s.id for s in all_sessions]
        assert "test-session-delete" not in all_session_ids

        # Verify session is also removed from all sessions (hard delete, not soft)
        all_sessions_page = honcho_client.sessions()
        all_sessions = list(all_sessions_page)
        all_session_ids = [s.id for s in all_sessions]
        assert "test-session-delete" not in all_session_ids


@pytest.mark.asyncio
async def test_session_queue_status(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests getting deriver status with various parameter combinations for sessions.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-deriver-status")
        assert isinstance(session, Session)

        status = await session.aio.queue_status()
        assert isinstance(status, QueueStatusResponse)
        assert hasattr(status, "total_work_units")
        assert hasattr(status, "completed_work_units")
        assert hasattr(status, "in_progress_work_units")
        assert hasattr(status, "pending_work_units")
        assert status.sessions is None

        # Test with observer only
        peer = await honcho_client.aio.peer(id="test-peer-session-deriver")
        await peer.aio.get_metadata()  # Create the peer
        status = await session.aio.queue_status(observer=peer.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with sender only
        status = await session.aio.queue_status(sender=peer.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with both observer and sender
        status = await session.aio.queue_status(observer=peer.id, sender=peer.id)
        assert isinstance(status, QueueStatusResponse)
    else:
        session = honcho_client.session(id="test-session-deriver-status")
        assert isinstance(session, Session)

        # Test with no parameters
        status = session.queue_status()
        assert isinstance(status, QueueStatusResponse)
        assert hasattr(status, "total_work_units")
        assert hasattr(status, "completed_work_units")
        assert hasattr(status, "in_progress_work_units")
        assert hasattr(status, "pending_work_units")
        assert status.sessions is None

        # Test with observer only
        peer = honcho_client.peer(id="test-peer-session-deriver")
        peer.get_metadata()  # Create the peer
        status = session.queue_status(observer=peer.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with sender only
        status = session.queue_status(sender=peer.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with both observer and sender
        status = session.queue_status(observer=peer.id, sender=peer.id)
        assert isinstance(status, QueueStatusResponse)


@pytest.mark.asyncio
async def test_session_poll_queue_status(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests polling deriver status until completion for sessions.
    """
    honcho_client, client_type = client_fixture

    # Mock the queue_status method to return a "completed" status
    # to avoid infinite polling in tests
    completed_status = QueueStatusResponse(
        total_work_units=0,
        completed_work_units=0,
        in_progress_work_units=0,
        pending_work_units=0,
    )

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-poll-queue")
        assert isinstance(session, Session)

        with patch.object(
            session.aio.__class__,
            "queue_status",
            new=AsyncMock(return_value=completed_status),
        ):
            status = await session.aio.poll_queue_status()
            assert isinstance(status, QueueStatusResponse)
            assert status.pending_work_units == 0
            assert status.in_progress_work_units == 0

        # Test with parameters
        peer = await honcho_client.aio.peer(id="test-peer-session-poll")
        with patch.object(
            session.aio.__class__,
            "queue_status",
            new=AsyncMock(return_value=completed_status),
        ):
            status = await session.aio.poll_queue_status(
                observer=peer.id, sender=peer.id
            )
            assert isinstance(status, QueueStatusResponse)
    else:
        session = honcho_client.session(id="test-session-poll-queue")
        assert isinstance(session, Session)

        with patch.object(
            session.__class__, "queue_status", return_value=completed_status
        ):
            status = session.poll_queue_status()
            assert isinstance(status, QueueStatusResponse)
            assert status.pending_work_units == 0
            assert status.in_progress_work_units == 0

        # Test with parameters
        peer = honcho_client.peer(id="test-peer-session-poll")
        with patch.object(
            session.__class__, "queue_status", return_value=completed_status
        ):
            status = session.poll_queue_status(observer=peer.id, sender=peer.id)
            assert isinstance(status, QueueStatusResponse)


@pytest.mark.asyncio
async def test_session_clone(client_fixture: tuple[Honcho, str]):
    """
    Tests cloning a session and verifying the cloned session has copied messages.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-clone-async")
        assert isinstance(session, Session)
        user = await honcho_client.aio.peer(id="user-clone-async")
        assert isinstance(user, Peer)

        # Add messages to the session (implicitly creates session and adds peer)
        await session.aio.add_messages(
            [
                user.message("First message"),
                user.message("Second message"),
            ]
        )

        # Clone the entire session
        cloned = await session.aio.clone()
        assert isinstance(cloned, Session)
        assert cloned.id != session.id  # Should have a different ID

        # Verify cloned session has the same messages
        cloned_messages_page = await cloned.aio.messages()
        cloned_messages = cloned_messages_page.items
        assert len(cloned_messages) == 2

        # Verify original session still has messages
        original_messages_page = await session.aio.messages()
        original_messages = original_messages_page.items
        assert len(original_messages) == 2
    else:
        session = honcho_client.session(id="test-session-clone-sync")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="user-clone-sync")
        assert isinstance(user, Peer)

        # Add messages to the session (implicitly creates session and adds peer)
        session.add_messages(
            [
                user.message("First message"),
                user.message("Second message"),
            ]
        )

        # Clone the entire session
        cloned = session.clone()
        assert isinstance(cloned, Session)
        assert cloned.id != session.id  # Should have a different ID

        # Verify cloned session has the same messages
        cloned_messages_page = cloned.messages()
        cloned_messages = list(cloned_messages_page)
        assert len(cloned_messages) == 2

        # Verify original session still has messages
        original_messages_page = session.messages()
        original_messages = list(original_messages_page)
        assert len(original_messages) == 2


@pytest.mark.asyncio
async def test_session_clone_with_cutoff(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests cloning a session up to a specific message.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-clone-cutoff-async")
        assert isinstance(session, Session)
        user = await honcho_client.aio.peer(id="user-clone-cutoff-async")
        assert isinstance(user, Peer)

        # Add messages to the session (implicitly creates session and adds peer)
        messages = await session.aio.add_messages(
            [
                user.message("First message"),
                user.message("Second message"),
                user.message("Third message"),
            ]
        )

        # Clone up to the first message
        first_message_id = messages[0].id
        cloned = await session.aio.clone(message_id=first_message_id)
        assert isinstance(cloned, Session)
        assert cloned.id != session.id

        # Verify cloned session only has 1 message
        cloned_messages_page = await cloned.aio.messages()
        cloned_messages = cloned_messages_page.items
        assert len(cloned_messages) == 1
        assert cloned_messages[0].content == "First message"

        # Verify original session still has all 3 messages
        original_messages_page = await session.aio.messages()
        original_messages = original_messages_page.items
        assert len(original_messages) == 3
    else:
        session = honcho_client.session(id="test-session-clone-cutoff-sync")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="user-clone-cutoff-sync")
        assert isinstance(user, Peer)

        # Add messages to the session (implicitly creates session and adds peer)
        messages = session.add_messages(
            [
                user.message("First message"),
                user.message("Second message"),
                user.message("Third message"),
            ]
        )

        # Clone up to the first message
        first_message_id = messages[0].id
        cloned = session.clone(message_id=first_message_id)
        assert isinstance(cloned, Session)
        assert cloned.id != session.id

        # Verify cloned session only has 1 message
        cloned_messages_page = cloned.messages()
        cloned_messages = list(cloned_messages_page)
        assert len(cloned_messages) == 1
        assert cloned_messages[0].content == "First message"

        # Verify original session still has all 3 messages
        original_messages_page = session.messages()
        original_messages = list(original_messages_page)
        assert len(original_messages) == 3
