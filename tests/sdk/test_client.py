from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from honcho_core.types import DeriverStatus

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
        results = search_results.items
        assert len(results) >= 1
        assert search_query in results[0].content
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="search-session-ws")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="search-user-ws")
        assert isinstance(user, Peer)
        session.add_messages([user.message(search_query)])

        search_results = honcho_client.search(search_query)
        results = list(search_results)
        assert len(results) >= 1
        assert search_query in results[0].content


@pytest.mark.asyncio
async def test_get_deriver_status(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting deriver status with various parameter combinations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        # Test with no parameters - this should work in the SDK even though API requires at least one
        status = await honcho_client.get_deriver_status()
        assert isinstance(status, DeriverStatus)
        assert hasattr(status, "total_work_units")
        assert hasattr(status, "completed_work_units")
        assert hasattr(status, "in_progress_work_units")
        assert hasattr(status, "pending_work_units")

        # Test with peer_id only
        peer = await honcho_client.peer(id="test-peer-deriver-status")
        await peer.get_metadata()  # Create the peer
        status = await honcho_client.get_deriver_status(observer_id=peer.id)
        assert isinstance(status, DeriverStatus)

        # Test with session_id only
        session = await honcho_client.session(id="test-session-deriver-status")
        await session.get_metadata()  # Create the session
        status = await honcho_client.get_deriver_status(session_id=session.id)
        assert isinstance(status, DeriverStatus)

        # Test with both peer_id and session_id
        status = await honcho_client.get_deriver_status(
            observer_id=peer.id, session_id=session.id
        )
        assert isinstance(status, DeriverStatus)

        # Test with include_sender=True
        status = await honcho_client.get_deriver_status(
            observer_id=peer.id, sender_id=peer.id
        )
        assert isinstance(status, DeriverStatus)
    else:
        assert isinstance(honcho_client, Honcho)
        # Test with no parameters
        status = honcho_client.get_deriver_status()
        assert isinstance(status, DeriverStatus)
        assert hasattr(status, "total_work_units")
        assert hasattr(status, "completed_work_units")
        assert hasattr(status, "in_progress_work_units")
        assert hasattr(status, "pending_work_units")

        # Test with peer_id only
        peer = honcho_client.peer(id="test-peer-deriver-status")
        peer.get_metadata()  # Create the peer
        status = honcho_client.get_deriver_status(observer_id=peer.id)
        assert isinstance(status, DeriverStatus)

        # Test with session_id only
        session = honcho_client.session(id="test-session-deriver-status")
        session.get_metadata()  # Create the session
        status = honcho_client.get_deriver_status(session_id=session.id)
        assert isinstance(status, DeriverStatus)

        # Test with both peer_id and session_id
        status = honcho_client.get_deriver_status(
            observer_id=peer.id, session_id=session.id
        )
        assert isinstance(status, DeriverStatus)

        # Test with include_sender=True
        status = honcho_client.get_deriver_status(
            observer_id=peer.id, sender_id=peer.id
        )
        assert isinstance(status, DeriverStatus)


@pytest.mark.asyncio
async def test_poll_deriver_status(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests polling deriver status until completion.
    """
    honcho_client, client_type = client_fixture

    # Mock the get_deriver_status method to return a "completed" status
    # to avoid infinite polling in tests
    completed_status = DeriverStatus(
        total_work_units=0,
        completed_work_units=0,
        in_progress_work_units=0,
        pending_work_units=0,
    )

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        with patch.object(
            honcho_client, "get_deriver_status", return_value=completed_status
        ):
            status = await honcho_client.poll_deriver_status()
            assert isinstance(status, DeriverStatus)
            assert status.pending_work_units == 0
            assert status.in_progress_work_units == 0

        # Test with parameters
        peer = await honcho_client.peer(id="test-peer-poll-status")
        with patch.object(
            honcho_client, "get_deriver_status", return_value=completed_status
        ):
            status = await honcho_client.poll_deriver_status(
                observer_id=peer.id, sender_id=peer.id
            )
            assert isinstance(status, DeriverStatus)
    else:
        assert isinstance(honcho_client, Honcho)
        with patch.object(
            honcho_client, "get_deriver_status", return_value=completed_status
        ):
            status = honcho_client.poll_deriver_status()
            assert isinstance(status, DeriverStatus)
            assert status.pending_work_units == 0
            assert status.in_progress_work_units == 0

        # Test with parameters
        peer = honcho_client.peer(id="test-peer-poll-status")
        with patch.object(
            honcho_client, "get_deriver_status", return_value=completed_status
        ):
            status = honcho_client.poll_deriver_status(
                observer_id=peer.id, sender_id=peer.id
            )
            assert isinstance(status, DeriverStatus)
