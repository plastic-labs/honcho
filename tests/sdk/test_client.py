import pytest
from fastapi.testclient import TestClient

from sdks.python.src.honcho.api_types import QueueStatusResponse
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.message import Message
from sdks.python.src.honcho.pagination import AsyncPage, SyncPage
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_client_init(client_fixture: tuple[Honcho, str], client: TestClient):
    """
    Tests that the Honcho SDK clients can be initialized and that a workspace
    is created on first use.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert honcho_client.workspace_id == "sdk-test-workspace-async"
        # Use the sync client to avoid mixing ASGI transports in this test.
        honcho_client.get_metadata()
    else:
        assert honcho_client.workspace_id == "sdk-test-workspace-sync"
        honcho_client.get_metadata()

    # Check all pages to find the workspace
    found_workspace = False
    page = 1

    while not found_workspace:
        res = client.post("/v3/workspaces/list", json={}, params={"page": page})
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
async def test_workspace_metadata(client_fixture: tuple[Honcho, str]):
    """
    Tests getting and setting metadata on a workspace.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        metadata = await honcho_client.aio.get_metadata()
        assert metadata == {}
        await honcho_client.aio.set_metadata({"foo": "bar"})
        metadata = await honcho_client.aio.get_metadata()
        assert metadata == {"foo": "bar"}
    else:
        metadata = honcho_client.get_metadata()
        assert metadata == {}
        honcho_client.set_metadata({"foo": "bar"})
        metadata = honcho_client.get_metadata()
        assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_workspaces(client_fixture: tuple[Honcho, str]):
    """
    Tests listing available workspaces.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        workspaces = await honcho_client.aio.workspaces()
    else:
        workspaces = honcho_client.workspaces()

    # workspaces returns a paginated Page of workspace ID strings
    assert hasattr(workspaces, "items")
    assert isinstance(workspaces.items, list)
    # Each item should be a string (workspace ID)
    for ws_id in workspaces.items:
        assert isinstance(ws_id, str)


@pytest.mark.asyncio
async def test_client_list_peers_and_sessions(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests listing peers and sessions at the client level.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peers_page = await honcho_client.aio.peers()
        assert isinstance(peers_page, AsyncPage)
        assert len(peers_page.items) == 0

        sessions_page = await honcho_client.aio.sessions()
        assert isinstance(sessions_page, AsyncPage)
        assert len(sessions_page.items) == 0

        peer = await honcho_client.aio.peer(id="test-peer-client")
        assert isinstance(peer, Peer)
        await peer.aio.get_metadata()  # Creates the peer

        peers_page = await honcho_client.aio.peers()
        assert len(peers_page.items) == 1

        session = await honcho_client.aio.session(id="test-session-client")
        assert isinstance(session, Session)
        await session.aio.get_metadata()  # Creates the session

        sessions_page = await honcho_client.aio.sessions()
        assert len(sessions_page.items) == 1
    else:
        peers_page = honcho_client.peers()
        assert isinstance(peers_page, SyncPage)
        assert len(list(peers_page)) == 0

        sessions_page = honcho_client.sessions()
        assert isinstance(sessions_page, SyncPage)
        assert len(list(sessions_page)) == 0

        peer = honcho_client.peer(id="test-peer-client")
        assert isinstance(peer, Peer)
        peer.get_metadata()

        peers_page = honcho_client.peers()
        assert len(list(peers_page)) == 1

        session = honcho_client.session(id="test-session-client")
        assert isinstance(session, Session)
        session.get_metadata()

        sessions_page = honcho_client.sessions()
        assert len(list(sessions_page)) == 1


@pytest.mark.asyncio
async def test_workspace_search(client_fixture: tuple[Honcho, str]):
    """
    Tests searching for messages within a workspace.
    """
    honcho_client, client_type = client_fixture
    search_query = "a unique message for workspace search"

    if client_type == "async":
        session = await honcho_client.aio.session(id="search-session-ws")
        assert isinstance(session, Session)
        user = await honcho_client.aio.peer(id="search-user-ws")
        assert isinstance(user, Peer)
        await session.aio.add_messages([user.message(search_query)])

        search_results = await honcho_client.aio.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1
        assert search_query in search_results[0].content
    else:
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
async def test_get_queue_status(client_fixture: tuple[Honcho, str]):
    """
    Tests getting queue status with various parameter combinations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        # Test with no parameters - this should work in the SDK even though API requires at least one
        status = await honcho_client.aio.queue_status()
        assert isinstance(status, QueueStatusResponse)
        assert hasattr(status, "total_work_units")
        assert hasattr(status, "completed_work_units")
        assert hasattr(status, "in_progress_work_units")
        assert hasattr(status, "pending_work_units")

        # Test with peer_id only
        peer = await honcho_client.aio.peer(id="test-peer-queue-status")
        await peer.aio.get_metadata()  # Create the peer
        status = await honcho_client.aio.queue_status(observer=peer.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with session_id only
        session = await honcho_client.aio.session(id="test-session-queue-status")
        await session.aio.get_metadata()  # Create the session
        status = await honcho_client.aio.queue_status(session=session.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with both peer and session
        status = await honcho_client.aio.queue_status(
            observer=peer.id, session=session.id
        )
        assert isinstance(status, QueueStatusResponse)

        # Test with sender
        status = await honcho_client.aio.queue_status(observer=peer.id, sender=peer.id)
        assert isinstance(status, QueueStatusResponse)
    else:
        # Test with no parameters
        status = honcho_client.queue_status()
        assert isinstance(status, QueueStatusResponse)
        assert hasattr(status, "total_work_units")
        assert hasattr(status, "completed_work_units")
        assert hasattr(status, "in_progress_work_units")
        assert hasattr(status, "pending_work_units")

        # Test with peer_id only
        peer = honcho_client.peer(id="test-peer-queue-status")
        peer.get_metadata()  # Create the peer
        status = honcho_client.queue_status(observer=peer.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with session_id only
        session = honcho_client.session(id="test-session-queue-status")
        session.get_metadata()  # Create the session
        status = honcho_client.queue_status(session=session.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with both peer and session
        status = honcho_client.queue_status(observer=peer.id, session=session.id)
        assert isinstance(status, QueueStatusResponse)

        # Test with sender
        status = honcho_client.queue_status(observer=peer.id, sender=peer.id)
        assert isinstance(status, QueueStatusResponse)


@pytest.mark.asyncio
async def test_update_message_with_message_object(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests updating message metadata using a Message object.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-update-msg-session")
        peer = await honcho_client.aio.peer(id="test-update-msg-peer")

        # Create a message
        await session.aio.add_messages([peer.message("test message")])
        messages = await session.aio.messages()
        assert len(messages) >= 1
        message = messages[0]
        assert isinstance(message, Message)

        # Update using Message object
        updated = await session.aio.update_message(message, {"key": "value"})
        assert isinstance(updated, Message)
        assert updated.metadata == {"key": "value"}
        assert updated.id == message.id
    else:
        session = honcho_client.session(id="test-update-msg-session")
        peer = honcho_client.peer(id="test-update-msg-peer")

        # Create a message
        session.add_messages([peer.message("test message")])
        messages = session.messages()
        assert len(messages) >= 1
        message = messages[0]
        assert isinstance(message, Message)

        # Update using Message object
        updated = session.update_message(message, {"key": "value"})
        assert isinstance(updated, Message)
        assert updated.metadata == {"key": "value"}
        assert updated.id == message.id


@pytest.mark.asyncio
async def test_update_message_with_message_id(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests updating message metadata using message_id string.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-update-msg-id-session")
        peer = await honcho_client.aio.peer(id="test-update-msg-id-peer")

        # Create a message
        await session.aio.add_messages([peer.message("test message")])

        messages = await session.aio.messages()
        assert len(messages) >= 1
        message = messages[0]
        assert message.metadata == {}

        # Update using message_id string
        updated = await session.aio.update_message(message.id, {"updated": True})
        assert isinstance(updated, Message)
        assert updated.metadata == {"updated": True}
        assert updated.id == message.id
    else:
        session = honcho_client.session(id="test-update-msg-id-session")
        peer = honcho_client.peer(id="test-update-msg-id-peer")

        # Create a message
        messages = session.add_messages([peer.message("test message")])

        messages = session.messages()
        assert len(messages) >= 1
        message = messages[0]
        assert message.metadata == {}

        # Update using message_id string
        updated = session.update_message(message.id, {"updated": True})
        assert isinstance(updated, Message)
        assert updated.metadata == {"updated": True}
        assert updated.id == message.id
