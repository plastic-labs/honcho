from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from honcho_core.types.workspaces import QueueStatusResponse
from honcho_core.types.workspaces.sessions.message import Message

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.pagination import AsyncPage
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.async_client.session import AsyncSession
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.pagination import SyncPage
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_client_init(
    client_fixture: tuple[Honcho | AsyncHoncho, str], client: TestClient
):
    """
    Tests that the Honcho SDK clients can be initialized and that they create a workspace.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        assert honcho_client.workspace_id == "sdk-test-workspace-async"
    else:
        assert isinstance(honcho_client, Honcho)
        assert honcho_client.workspace_id == "sdk-test-workspace-sync"

    # Check all pages to find the workspace
    found_workspace = False
    page = 1

    while not found_workspace:
        res = client.post("/v2/workspaces/list", json={}, params={"page": page})
        assert res.status_code == 200

        data = res.json()
        workspaces = data["items"]
        workspace_ids = [w["id"] for w in workspaces]

        if honcho_client.workspace_id in workspace_ids:
            found_workspace = True
            break

        # Check if there are more pages
        if page >= data.get("pages", 1) or len(workspaces) == 0:
            break

        page += 1

    assert (
        found_workspace
    ), f"Workspace {honcho_client.workspace_id} not found in any page of results"


@pytest.mark.asyncio
async def test_workspace_metadata(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting and setting metadata on a workspace.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        metadata = await honcho_client.get_metadata()
        assert metadata == {}
        await honcho_client.set_metadata({"foo": "bar"})
        metadata = await honcho_client.get_metadata()
        assert metadata == {"foo": "bar"}
    else:
        assert isinstance(honcho_client, Honcho)
        metadata = honcho_client.get_metadata()
        assert metadata == {}
        honcho_client.set_metadata({"foo": "bar"})
        metadata = honcho_client.get_metadata()
        assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_get_workspaces(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests listing available workspaces.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        workspaces = await honcho_client.get_workspaces()
    else:
        assert isinstance(honcho_client, Honcho)
        workspaces = honcho_client.get_workspaces()

    assert isinstance(workspaces, list)
    assert honcho_client.workspace_id in workspaces


@pytest.mark.asyncio
async def test_client_list_peers_and_sessions(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests listing peers and sessions at the client level.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peers_page = await honcho_client.get_peers()
        assert isinstance(peers_page, AsyncPage)
        assert len(peers_page.items) == 0

        sessions_page = await honcho_client.get_sessions()
        assert isinstance(sessions_page, AsyncPage)
        assert len(sessions_page.items) == 0

        peer = await honcho_client.peer(id="test-peer-client")
        assert isinstance(peer, AsyncPeer)
        await peer.get_metadata()  # Creates the peer

        peers_page = await honcho_client.get_peers()
        assert len(peers_page.items) == 1

        session = await honcho_client.session(id="test-session-client")
        assert isinstance(session, AsyncSession)
        await session.get_metadata()  # Creates the session

        sessions_page = await honcho_client.get_sessions()
        assert len(sessions_page.items) == 1
    else:
        assert isinstance(honcho_client, Honcho)
        peers_page = honcho_client.get_peers()
        assert isinstance(peers_page, SyncPage)
        assert len(list(peers_page)) == 0

        sessions_page = honcho_client.get_sessions()
        assert isinstance(sessions_page, SyncPage)
        assert len(list(sessions_page)) == 0

        peer = honcho_client.peer(id="test-peer-client")
        assert isinstance(peer, Peer)
        peer.get_metadata()

        peers_page = honcho_client.get_peers()
        assert len(list(peers_page)) == 1

        session = honcho_client.session(id="test-session-client")
        assert isinstance(session, Session)
        session.get_metadata()

        sessions_page = honcho_client.get_sessions()
        assert len(list(sessions_page)) == 1


@pytest.mark.asyncio
async def test_workspace_search(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests searching for messages within a workspace.
    """
    honcho_client, client_type = client_fixture
    search_query = "a unique message for workspace search"

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="search-session-ws")
        assert isinstance(session, AsyncSession)
        user = await honcho_client.peer(id="search-user-ws")
        assert isinstance(user, AsyncPeer)
        await session.add_messages([user.message(search_query)])

        search_results = await honcho_client.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1
        assert search_query in search_results[0].content
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="search-session-ws")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="search-user-ws")
        assert isinstance(user, Peer)
        session.add_messages([user.message(search_query)])

        search_results = honcho_client.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1
        assert search_query in search_results[0].content


@pytest.mark.asyncio
async def test_get_deriver_status(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting deriver status with various parameter combinations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        # Test with no parameters - this should work in the SDK even though API requires at least one
        status = await honcho_client.get_queue_status()
        assert isinstance(status, QueueStatusResponse)
        assert hasattr(status, "total_work_units")
        assert hasattr(status, "completed_work_units")
        assert hasattr(status, "in_progress_work_units")
        assert hasattr(status, "pending_work_units")

        # Test with peer_id only
        peer = await honcho_client.peer(id="test-peer-deriver-status")
        await peer.get_metadata()  # Create the peer
        status = await honcho_client.get_queue_status(observer=peer.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with session_id only
        session = await honcho_client.session(id="test-session-deriver-status")
        await session.get_metadata()  # Create the session
        status = await honcho_client.get_queue_status(session=session.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with both peer and session
        status = await honcho_client.get_queue_status(
            observer=peer.id, session=session.id
        )
        assert isinstance(status, QueueStatusResponse)

        # Test with sender
        status = await honcho_client.get_queue_status(observer=peer.id, sender=peer.id)
        assert isinstance(status, QueueStatusResponse)
    else:
        assert isinstance(honcho_client, Honcho)
        # Test with no parameters
        status = honcho_client.get_queue_status()
        assert isinstance(status, QueueStatusResponse)
        assert hasattr(status, "total_work_units")
        assert hasattr(status, "completed_work_units")
        assert hasattr(status, "in_progress_work_units")
        assert hasattr(status, "pending_work_units")

        # Test with peer_id only
        peer = honcho_client.peer(id="test-peer-queue-status")
        peer.get_metadata()  # Create the peer
        status = honcho_client.get_queue_status(observer=peer.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with session_id only
        session = honcho_client.session(id="test-session-queue-status")
        session.get_metadata()  # Create the session
        status = honcho_client.get_queue_status(session=session.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with both peer and session
        status = honcho_client.get_queue_status(observer=peer.id, session=session.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with sender
        status = honcho_client.get_queue_status(observer=peer.id, sender=peer.id)
        assert isinstance(status, QueueStatusResponse)


@pytest.mark.asyncio
async def test_poll_queue_status(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests polling queue status until completion.
    """
    honcho_client, client_type = client_fixture

    # Mock the get_queue_status method to return a "completed" status
    # to avoid infinite polling in tests
    completed_status = QueueStatusResponse(
        total_work_units=0,
        completed_work_units=0,
        in_progress_work_units=0,
        pending_work_units=0,
    )

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        with patch.object(
            honcho_client, "get_queue_status", return_value=completed_status
        ):
            status = await honcho_client.poll_queue_status()
            assert isinstance(status, QueueStatusResponse)
            assert status.pending_work_units == 0
            assert status.in_progress_work_units == 0

        # Test with parameters
        peer = await honcho_client.peer(id="test-peer-poll-status")
        with patch.object(
            honcho_client, "get_queue_status", return_value=completed_status
        ):
            status = await honcho_client.poll_queue_status(
                observer=peer.id, sender=peer.id
            )
            assert isinstance(status, QueueStatusResponse)
    else:
        assert isinstance(honcho_client, Honcho)
        with patch.object(
            honcho_client, "get_queue_status", return_value=completed_status
        ):
            status = honcho_client.poll_queue_status()
            assert isinstance(status, QueueStatusResponse)
            assert status.pending_work_units == 0
            assert status.in_progress_work_units == 0

        # Test with parameters
        peer = honcho_client.peer(id="test-peer-poll-status")
        with patch.object(
            honcho_client, "get_queue_status", return_value=completed_status
        ):
            status = honcho_client.poll_queue_status(observer=peer.id, sender=peer.id)
            assert isinstance(status, QueueStatusResponse)


@pytest.mark.asyncio
async def test_update_message_with_message_object(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests updating message metadata using a Message object.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-update-msg-session")
        peer = await honcho_client.peer(id="test-update-msg-peer")

        # Create a message
        await session.add_messages([peer.message("test message")])
        messages = await session.get_messages()
        assert len(messages) >= 1
        message = messages[0]
        assert isinstance(message, Message)

        # Update using Message object
        updated = await honcho_client.update_message(message, {"key": "value"})
        assert isinstance(updated, Message)
        assert updated.metadata == {"key": "value"}
        assert updated.id == message.id
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-update-msg-session")
        peer = honcho_client.peer(id="test-update-msg-peer")

        # Create a message
        session.add_messages([peer.message("test message")])
        messages = session.get_messages()
        assert len(messages) >= 1
        message = messages[0]
        assert isinstance(message, Message)

        # Update using Message object
        updated = honcho_client.update_message(message, {"key": "value"})
        assert isinstance(updated, Message)
        assert updated.metadata == {"key": "value"}
        assert updated.id == message.id


@pytest.mark.asyncio
async def test_update_message_with_message_id(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests updating message metadata using message_id string.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-update-msg-id-session")
        peer = await honcho_client.peer(id="test-update-msg-id-peer")

        # Create a message
        await session.add_messages([peer.message("test message")])

        messages = await session.get_messages()
        assert len(messages) >= 1
        message = messages[0]
        assert message.metadata == {}

        # Update using message_id string
        updated = await honcho_client.update_message(
            message.id, {"updated": True}, session=session.id
        )
        assert isinstance(updated, Message)
        assert updated.metadata == {"updated": True}
        assert updated.id == message.id
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-update-msg-id-session")
        peer = honcho_client.peer(id="test-update-msg-id-peer")

        # Create a message
        messages = session.add_messages([peer.message("test message")])

        messages = session.get_messages()
        assert len(messages) >= 1
        message = messages[0]
        assert message.metadata == {}

        # Update using message_id string
        updated = honcho_client.update_message(
            message.id, {"updated": True}, session=session.id
        )
        assert isinstance(updated, Message)
        assert updated.metadata == {"updated": True}
        assert updated.id == message.id


@pytest.mark.asyncio
async def test_update_message_validation(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that update_message raises ValueError when message ID is provided without session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        with pytest.raises(
            ValueError, match="session is required when message is a string ID"
        ):
            await honcho_client.update_message("msg_123", {"key": "value"})
    else:
        assert isinstance(honcho_client, Honcho)
        with pytest.raises(
            ValueError, match="session is required when message is a string ID"
        ):
            honcho_client.update_message("msg_123", {"key": "value"})
