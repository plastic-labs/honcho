from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.models import Peer, Workspace


def test_get_or_create_peer(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, _ = sample_data
    name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": {"peer_key": "peer_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["metadata"] == {"peer_key": "peer_value"}
    assert "id" in data


def test_get_or_create_peer_with_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer creation with configuration parameter"""
    test_workspace, _ = sample_data
    name = str(generate_nanoid())
    configuration = {"experimental": True, "beta": False}

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": name, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["configuration"] == configuration


def test_get_or_create_peer_with_all_optional_params(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer creation with all optional parameters"""
    test_workspace, _ = sample_data
    name = str(generate_nanoid())
    metadata = {"key": "value", "number": 42}
    configuration = {"feature1": True, "feature2": False}

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_get_or_create_existing_peer(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    name = str(generate_nanoid())

    # Create the peer
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": {"peer_key": "peer_value"}},
    )
    assert response.status_code == 200
    peer1 = response.json()

    # Try to create the same peer again - should return existing peer
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": {"peer_key": "peer_value"}},
    )
    assert response.status_code == 200
    peer2 = response.json()

    # Both should be the same peer
    assert peer1["id"] == peer2["id"]
    assert peer1["metadata"] == peer2["metadata"]


def test_get_peers(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, _ = sample_data

    # Create a few peers with metadata
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": str(generate_nanoid()), "metadata": {"peer_key": "peer_value"}},
    )
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": str(generate_nanoid()), "metadata": {"peer_key": "peer_value"}},
    )
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": str(generate_nanoid()), "metadata": {"peer_key": "peer_value2"}},
    )

    # Get all peers
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0

    # Get peers with simple filter (backward compatibility)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filters": {"metadata": {"peer_key": "peer_value"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    assert data["items"][0]["metadata"]["peer_key"] == "peer_value"

    # Test new filter with NOT operator
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filters": {"NOT": [{"metadata": {"peer_key": "peer_value2"}}]}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    # Should find peers that don't have peer_key = "peer_value2"
    # This includes the 2 peers with "peer_value" + the sample peer with empty metadata
    assert len(data["items"]) == 3


def test_get_peers_with_empty_filter(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer listing with empty filter object"""
    test_workspace, _ = sample_data

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list", json={"filters": {}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_get_peers_with_null_filter(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer listing with null filter"""
    test_workspace, _ = sample_data

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list", json={"filters": None}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_update_peer(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


def test_update_peer_with_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer update with configuration parameter"""
    test_workspace, test_peer = sample_data
    configuration = {"new_feature": True, "legacy_feature": False}

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["configuration"] == configuration


def test_update_peer_with_all_optional_params(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer update with both metadata and configuration"""
    test_workspace, test_peer = sample_data
    metadata = {"updated_key": "updated_value", "count": 100}
    configuration = {"experimental": True, "beta": True}

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_update_peer_with_null_metadata(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer update with null metadata"""
    test_workspace, test_peer = sample_data

    # First set some metadata
    client.put(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": {"temp": "value"}},
    )

    # Then clear it with null
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data


def test_update_peer_with_null_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer update with null configuration"""
    test_workspace, test_peer = sample_data

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"configuration": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert "configuration" in data


def test_get_sessions_for_peer_no_sessions(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data

    # Get sessions for the peer
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data


def test_get_sessions_for_peer(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data

    # Create session for the peer
    session_name = str(generate_nanoid())
    create_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_name, "peer_names": {test_peer.name: {}}},
    )
    assert create_response.status_code == 200
    created_session = create_response.json()
    assert created_session["id"] == session_name

    # Now get sessions for the peer and validate the session is returned
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    # Check that the created session is in the returned items
    session_ids = [item["id"] for item in data["items"]]
    assert session_name in session_ids
    assert len(data["items"]) == 1


def test_get_sessions_for_peer_with_empty_filter(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test getting sessions for peer with empty filter object"""
    test_workspace, test_peer = sample_data

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
        json={"filters": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@patch("src.routers.peers.tracked_db")
def test_chat(
    mock_tracked_db: AsyncMock,
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
):
    mock_tracked_db.return_value.__aenter__.return_value = db_session

    test_workspace, test_peer = sample_data
    target_peer = str(generate_nanoid())

    # Test chat endpoint
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        json={
            "query": "Hello, how are you?",
            "stream": False,
            "target": target_peer,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data


@patch("src.routers.peers.tracked_db")
def test_chat_with_optional_params(
    mock_tracked_db: AsyncMock,
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
):
    """Test chat endpoint with optional parameters"""
    mock_tracked_db.return_value.__aenter__.return_value = db_session

    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create a session first
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Test chat without optional parameters
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        json={
            "query": "Hello, how are you?",
            "stream": False,
            "session_id": session_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data


def test_get_peer_representation_with_session(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer representation with session_id parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create a session first
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Test representation scoped to session
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/representation",
        json={
            "session_id": session_id,
            "queries": "Hello, how are you?",
        },
    )
    assert response.status_code == 200


def test_search_peer(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test the peer search functionality"""
    test_workspace, test_peer = sample_data

    # Add some messages to search through
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions/test_session/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another searchable message", "peer_id": test_peer.name},
            ]
        },
    )

    # Search with a query
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/search",
        json={"query": "search query", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_peer_empty_query(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer search with empty query"""
    test_workspace, test_peer = sample_data

    # Search with empty query
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/search",
        json={"query": "", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_peer_nonexistent(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test searching a peer that doesn't exist"""
    test_workspace, _ = sample_data
    nonexistent_peer_id = str(generate_nanoid())

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{nonexistent_peer_id}/search",
        json={"query": "test query", "limit": 10},
    )
    assert response.status_code == 200
    data: list[dict[str, Any]] = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_search_peer_with_messages(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer search with actual messages"""
    test_workspace, test_peer = sample_data

    # Add some messages to search through
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions/test_session/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another searchable message", "peer_id": test_peer.name},
            ]
        },
    )

    # Search for content
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/search",
        json={"query": "search", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_peer_with_limit(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer search with custom limit"""
    test_workspace, test_peer = sample_data

    # Add some messages to search through
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions/test_session/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another searchable message", "peer_id": test_peer.name},
                {"content": "More searchable content", "peer_id": test_peer.name},
            ]
        },
    )

    # Search with custom limit
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/search",
        json={"query": "search", "limit": 2},
    )

    assert response.status_code == 200
    data: list[dict[str, Any]] = response.json()
    assert isinstance(data, list)
    # Should not exceed the limit
    assert len(data) <= 2


def test_get_peers_with_complex_filter(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer listing with complex filters"""
    test_workspace, _ = sample_data

    # Create peers with different metadata
    for i in range(3):
        client.post(
            f"/v2/workspaces/{test_workspace.name}/peers",
            json={
                "name": str(generate_nanoid()),
                "metadata": {"index": i, "type": "test"},
            },
        )

    # Test complex filter combination
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filters": {
                "AND": [
                    {"metadata": {"type": "test"}},
                    {"metadata": {"index": {"gte": 1}}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    # for each peer returned, assert that the metadata["index"] fits our filter
    for peer in data["items"]:
        assert (
            peer["metadata"].get("index", 0) >= 1
            and peer["metadata"].get("type", "") == "test"
        )
    assert "items" in data


def test_update_peer_all_fields(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test peer update with all possible fields"""
    test_workspace, test_peer = sample_data

    # Test updating both metadata and configuration
    metadata = {"updated": True, "version": 2}
    configuration = {"features": {"new_feature": True}}

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_get_peer_card(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test the peer cards endpoint"""
    test_workspace, observer_peer = sample_data

    # Create a second peer (the target/observed peer)
    target_peer_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": target_peer_name},
    )
    assert response.status_code == 200

    # Test getting observer's own card (should return null initially)
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/peers/{observer_peer.name}/card"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["peer_card"] is None

    # Test getting card for target peer (should return null initially)
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/peers/{observer_peer.name}/card",
        params={"target": target_peer_name},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["peer_card"] is None


@pytest.mark.asyncio
async def test_get_peer_card_with_data(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test the peer cards endpoint with actual peer card data"""
    test_workspace, observer_peer = sample_data

    # Create a second peer (the target/observed peer)
    target_peer_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": target_peer_name},
    )
    assert response.status_code == 200

    # Set up peer cards using the database directly
    # Set a self-card for the observer peer
    self_card_content = ["I am a helpful AI assistant", "I enjoy learning about users"]
    await crud.set_peer_card(
        db_session,
        test_workspace.name,
        observer_peer.name,
        observer_peer.name,
        self_card_content,
    )

    # Set a card for the observer describing the target peer
    target_card_content = ["This peer seems friendly", "They ask good questions"]
    await crud.set_peer_card(
        db_session,
        test_workspace.name,
        target_peer_name,
        observer_peer.name,
        target_card_content,
    )

    # Test getting observer's own card
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/peers/{observer_peer.name}/card"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["peer_card"] == self_card_content

    # Test getting card for target peer
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/peers/{observer_peer.name}/card",
        params={"target": target_peer_name},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["peer_card"] == target_card_content
