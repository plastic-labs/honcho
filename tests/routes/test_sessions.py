from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src.models import Peer, Workspace


def test_get_or_create_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    # Test get or create session
    test_workspace, test_peer = sample_data

    # Test creating a new session with no parameters
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": str(generate_nanoid())},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["id"], str)
    assert data["metadata"] == {}
    assert data["workspace_id"] == test_workspace.name

    # Test creating a session with a specific id and peer_names (should get or create)
    session_id = str(generate_nanoid())
    response2 = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["id"] == session_id
    assert data2["metadata"] == {}
    assert data2["workspace_id"] == test_workspace.name

    # Test getting the same session again (should return the same session)
    response3 = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert response3.status_code == 200
    data3 = response3.json()
    assert data3["id"] == session_id
    assert data3["metadata"] == {}
    assert data3["workspace_id"] == test_workspace.name


def test_create_session_with_metadata(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"session_key": "session_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"session_key": "session_value"}
    assert "id" in data
    assert data["id"] == session_id
    assert data["workspace_id"] == test_workspace.name


def test_create_session_with_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session creation with configuration parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())
    configuration = {"experimental_feature": True, "beta_mode": False}

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "configuration": configuration,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["configuration"] == configuration
    assert data["id"] == session_id
    assert data["workspace_id"] == test_workspace.name


def test_create_session_with_all_optional_params(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session creation with all optional parameters"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())
    metadata = {"key": "value", "number": 42}
    configuration = {"feature1": True, "feature2": False}

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": metadata,
            "configuration": configuration,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration
    assert data["id"] == session_id
    assert data["workspace_id"] == test_workspace.name


def test_create_session_with_too_many_peers(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    caplog: pytest.LogCaptureFixture,
):
    test_workspace, test_peer = sample_data
    # create 10 peers
    peer_names = [test_peer.name]
    for _ in range(10):
        peer_name = str(generate_nanoid())
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/peers",
            json={"name": peer_name, "metadata": {}},
        )
        assert response.status_code == 200
        peer_names.append(peer_name)

    # create session with 11 peers
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": str(generate_nanoid()),
            "peer_names": {peer_name: {} for peer_name in peer_names},
        },
    )
    assert response.status_code == 422
    assert "Failed to get or create session" in caplog.text

    session_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"name": "test_session"}},
    )
    assert session_response.status_code == 200
    assert len(session_response.json()["items"]) == 0

    # Remove one peer from our list
    peer_names.pop()
    # Attempt to create session with same name
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": "test_session",
            "peer_names": {peer_name: {} for peer_name in peer_names},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test_session"
    assert data["workspace_id"] == test_workspace.name


def test_get_sessions(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"test_key": "test_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test_key": "test_value"}
    assert "id" in data
    assert data["workspace_id"] == test_workspace.name
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"test_key": "test_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["metadata"] == {"test_key": "test_value"}
    assert data["items"][0]["workspace_id"] == test_workspace.name


def test_get_sessions_with_is_active_filter(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session listing with is_active parameter"""
    test_workspace, test_peer = sample_data

    # Create and then delete a session to have inactive session
    session_id = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    client.delete(f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}")

    # Test getting inactive sessions
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list", json={"is_active": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    # Should find at least our deleted session
    inactive_sessions = [s for s in data["items"] if not s["is_active"]]
    assert len(inactive_sessions) > 0


def test_get_sessions_with_empty_filter(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session listing with empty filter object"""
    test_workspace, _ = sample_data

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list", json={"filter": {}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_update_delete_metadata(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"default": "value"},
        },
    )
    assert response.status_code == 200

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {}


def test_update_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
        },
    )
    assert response.status_code == 200

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


def test_update_session_with_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session update with configuration parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Update with configuration
    configuration = {"new_feature": True, "legacy_feature": False}
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": {}, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["configuration"] == configuration


def test_update_session_with_all_optional_params(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session update with both metadata and configuration"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Update with all params
    metadata = {"updated_key": "updated_value"}
    configuration = {"experimental": True, "beta": True}
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_update_session_with_null_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session update with null configuration"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session with configuration
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "configuration": {"temp": "value"},
        },
    )

    # Update with null configuration
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": {}, "configuration": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert "configuration" in data


def test_delete_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
        },
    )
    assert response.status_code == 200

    response = client.delete(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}"
    )
    assert response.status_code == 200

    # Check that session is marked as inactive
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"is_active": False},
    )
    data = response.json()
    # Find our session in the inactive sessions
    inactive_session = next((s for s in data["items"] if s["id"] == session_id), None)
    assert inactive_session is not None
    assert inactive_session["is_active"] is False


def test_clone_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"test": "key"},
        },
    )
    assert response.status_code == 200

    # Create some messages in the session
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Test message",
                    "peer_id": test_peer.name,
                    "metadata": {"key": "value"},
                },
                {
                    "content": "Test message 2",
                    "peer_id": test_peer.name,
                    "metadata": {"key": "value2"},
                },
            ]
        },
    )
    assert response.status_code == 200

    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/clone",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test": "key"}

    # Check messages were cloned
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{data['id']}/messages/list",
        json={},
    )

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) >= 2

    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["metadata"] == {"key": "value"}

    assert data["items"][1]["content"] == "Test message 2"
    assert data["items"][1]["metadata"] == {"key": "value2"}


def test_clone_session_with_cutoff(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session cloning with message cutoff parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Create messages
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Message 1", "peer_id": test_peer.name},
                {"content": "Message 2", "peer_id": test_peer.name},
            ]
        },
    )
    assert response.status_code == 200
    messages_data = response.json()
    # The response is a list of messages, not a paginated response
    first_message_id = messages_data[0]["id"]

    # Clone with cutoff at first message
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/clone?message_id={first_message_id}",
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data


def test_add_peers_to_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code == 200

    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
        },
    )
    assert response.status_code == 200

    # Add another peer to the session
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={peer2_name: {}},
    )
    assert response.status_code == 200


def test_get_session_peers(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code == 200

    # Create a test session with multiple peers
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}, peer2_name: {}},
        },
    )
    assert response.status_code == 200

    # Get peers from the session
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    peer_names = [peer["id"] for peer in data["items"]]
    assert test_peer.name in peer_names
    assert peer2_name in peer_names


def test_set_session_peers(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code == 200

    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
        },
    )
    assert response.status_code == 200

    # Set peers for the session (should replace existing peers)
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={peer2_name: {}},
    )
    assert response.status_code == 200

    # Check that only the new peer is in the session
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer2_name


def test_set_session_peers_with_limit(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    caplog: pytest.LogCaptureFixture,
):
    test_workspace, test_peer = sample_data

    # Create a test session with multiple peers
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
        },
    )
    assert response.status_code == 200

    # create 10 peers
    peer_names = [test_peer.name]
    for _ in range(10):
        peer_name = str(generate_nanoid())
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/peers",
            json={"name": peer_name, "metadata": {}},
        )
        assert response.status_code == 200
        peer_names.append(peer_name)

    # set peers with 11 peers (as a dict of peer_name: {})
    peers_dict: dict[str, dict[Any, Any]] = {peer_name: {} for peer_name in peer_names}
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=peers_dict,
    )
    assert response.status_code == 404
    assert "Failed to set peers for session" in caplog.text


def test_remove_peers_from_session(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code == 200

    # Create a test session with multiple peers
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}, peer2_name: {}},
        },
    )
    assert response.status_code == 200

    # Remove one peer from the session
    response = client.request(
        "DELETE",
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=[test_peer.name],
    )
    assert response.status_code == 200

    # Check that only the remaining peer is in the session
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer2_name


def test_get_session_context(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test the session context endpoint"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Add some messages to have context
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Test message 1", "peer_id": test_peer.name},
                {"content": "Test message 2", "peer_id": test_peer.name},
            ]
        },
    )

    # Get context
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/context",
    )
    assert response.status_code == 200
    data = response.json()
    # SessionContext schema serializes "name" as "id"
    assert "id" in data
    assert "messages" in data
    assert "summary" in data
    assert data["id"] == session_id
    assert isinstance(data["messages"], list)
    assert data["summary"] == ""  # Default is empty when summary=False


def test_get_session_context_with_summary(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with summary parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session with messages
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with summary
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/context?summary=true",
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data


def test_get_session_context_with_tokens(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with token limit parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with token limit
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/context?tokens=100",
    )
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert isinstance(data["messages"], list)


def test_get_session_context_with_all_params(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with both summary and tokens parameters"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with all parameters
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/context?tokens=100&summary=true",
    )
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert "summary" in data


def test_search_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test the session search functionality"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Add messages to search through
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another message", "peer_id": test_peer.name},
            ]
        },
    )

    # Search with a query
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/search",
        json={"query": "search query", "use_semantic_search": False},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should have pagination structure
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert isinstance(data["items"], list)


def test_search_session_empty_query(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session search with empty query"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Search with empty query
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/search",
        json={"query": "", "use_semantic_search": False},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should still have proper pagination structure
    assert "items" in data
    assert isinstance(data["items"], list)


def test_search_session_nonexistent(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test searching a session that doesn't exist"""
    test_workspace, _ = sample_data
    nonexistent_session_id = str(generate_nanoid())

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{nonexistent_session_id}/search",
        json={"query": "test query", "use_semantic_search": False},
    )
    # This should probably return 404 or handle gracefully
    # The exact behavior depends on the crud.search implementation
    assert response.status_code in [200, 404, 422]


def test_search_session_with_semantic_search_false(client, sample_data):
    """Test session search with use_semantic_search=false"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Add messages to search through
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another message to find", "peer_id": test_peer.name},
            ]
        },
    )

    # Search with use_semantic_search=false
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/search",
        json={"query": "search", "use_semantic_search": False},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should have pagination structure
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert isinstance(data["items"], list)


def test_search_session_with_semantic_search_true_disabled(client, sample_data):
    """Test session search with use_semantic_search=true when EMBED_MESSAGES is disabled"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Add messages to search through
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another message to find", "peer_id": test_peer.name},
            ]
        },
    )

    # Search with use_semantic_search=true (should fail if EMBED_MESSAGES is disabled)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/search",
        json={"query": "search", "use_semantic_search": True},
    )

    assert response.status_code == 405

    data = response.json()
    assert "Semantic search requires EMBED_MESSAGES flag to be enabled" in data.get(
        "detail", ""
    )
