from typing import Any

from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src.models import Peer, Workspace


def test_get_or_create_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    # Test get or create session
    test_workspace, test_peer = sample_data

    # Test creating a new session with no parameters
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": str(generate_nanoid())},
    )
    assert response.status_code in [200, 201]
    data = response.json()
    assert isinstance(data["id"], str)
    assert data["metadata"] == {}
    assert data["workspace_id"] == test_workspace.name

    # Test creating a session with a specific id and peer_names (should get or create)
    session_id = str(generate_nanoid())
    response2 = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert response2.status_code in [200, 201]
    data2 = response2.json()
    assert data2["id"] == session_id
    assert data2["metadata"] == {}
    assert data2["workspace_id"] == test_workspace.name

    # Test getting the same session again (should return the same session)
    response3 = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert response3.status_code in [200, 201]
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"session_key": "session_value"},
        },
    )
    assert response.status_code in [200, 201]
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "configuration": configuration,
        },
    )
    assert response.status_code in [200, 201]
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": metadata,
            "configuration": configuration,
        },
    )
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration
    assert data["id"] == session_id
    assert data["workspace_id"] == test_workspace.name


def test_create_session_with_too_many_peers(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
):
    """Test that creating a session with too many observers fails"""
    test_workspace, test_peer = sample_data
    # create 10 peers
    peer_names = [test_peer.name]
    for _ in range(10):
        peer_name = str(generate_nanoid())
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/peers",
            json={"name": peer_name, "metadata": {}},
        )
        assert response.status_code in [200, 201]
        peer_names.append(peer_name)

    # Test 1: Create session with 11 non-observers should succeed
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": str(generate_nanoid()),
            "peer_names": {peer_name: {} for peer_name in peer_names},
        },
    )
    assert response.status_code in [200, 201]  # Should succeed since no observers

    # Test 2: Try to create session with 11 observers (exceeds limit)
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": str(generate_nanoid()),
            "peer_names": {
                peer_name: {"observe_others": True} for peer_name in peer_names
            },
        },
    )
    assert response.status_code == 400
    assert "11 observers" in response.json()["detail"]
    assert "Maximum allowed is 10 observers" in response.json()["detail"]

    session_response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/list",
        json={"filters": {"id": "test_session"}},
    )
    assert session_response.status_code == 200
    assert len(session_response.json()["items"]) == 0

    # Remove one peer from our list
    peer_names.pop()
    # Attempt to create session with same name
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": "test_session",
            "peer_names": {peer_name: {} for peer_name in peer_names},
        },
    )
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["id"] == "test_session"
    assert data["workspace_id"] == test_workspace.name


def test_get_sessions(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"test_key": "test_value"},
        },
    )
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["metadata"] == {"test_key": "test_value"}
    assert "id" in data
    assert data["workspace_id"] == test_workspace.name
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/list",
        json={"filters": {"metadata": {"test_key": "test_value"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["metadata"] == {"test_key": "test_value"}
    assert data["items"][0]["workspace_id"] == test_workspace.name


def test_get_sessions_with_empty_filter(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session listing with empty filter object"""
    test_workspace, _ = sample_data

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/list", json={"filters": {}}
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"default": "value"},
        },
    )
    assert response.status_code in [200, 201]

    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
        },
    )
    assert response.status_code in [200, 201]

    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


def test_delete_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test deleting a session"""
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"test_key": "test_value"},
        },
    )
    assert response.status_code in [200, 201]

    # Delete the session
    response = client.delete(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}",
    )
    assert response.status_code == 202
    data = response.json()
    assert data["message"] == "Session deleted successfully"

    # Verify the session is deleted by trying to list it
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/list",
        json={"filters": {"id": session_id}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 0


def test_update_session_with_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session update with configuration parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Update with configuration
    configuration = {"new_feature": True, "legacy_feature": False}
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Update with all params
    metadata = {"updated_key": "updated_value"}
    configuration = {"experimental": True, "beta": True}
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "configuration": {"temp": "value"},
        },
    )

    # Update with null configuration
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": {}, "configuration": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert "configuration" in data


def test_clone_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"test": "key"},
        },
    )
    assert response.status_code in [200, 201]

    # Create some messages in the session
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
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
    assert response.status_code == 201

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/clone",
    )
    assert response.status_code == 201
    data = response.json()
    assert data["metadata"] == {"test": "key"}

    # Check messages were cloned
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{data['id']}/messages/list",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Create messages
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Message 1", "peer_id": test_peer.name},
                {"content": "Message 2", "peer_id": test_peer.name},
            ]
        },
    )
    assert response.status_code == 201
    messages_data = response.json()
    # The response is a list of messages, not a paginated response
    first_message_id = messages_data[0]["id"]

    # Clone with cutoff at first message
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/clone?message_id={first_message_id}",
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data


def test_add_peers_to_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code in [200, 201]

    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
        },
    )
    assert response.status_code in [200, 201]

    # Add another peer to the session
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={peer2_name: {}},
    )
    assert response.status_code == 200


def test_get_session_peers(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code in [200, 201]

    # Create a test session with multiple peers
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}, peer2_name: {}},
        },
    )
    assert response.status_code in [200, 201]

    # Get peers from the session
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
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
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code in [200, 201]

    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
        },
    )
    assert response.status_code in [200, 201]

    # Set peers for the session (should replace existing peers)
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json={peer2_name: {}},
    )
    assert response.status_code == 200

    # Check that only the new peer is in the session
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer2_name


def test_set_session_peers_with_observer_limit(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
):
    """Test that session observer limit is enforced based on observe_others setting"""
    test_workspace, test_peer = sample_data

    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
        },
    )
    assert response.status_code in [200, 201]

    # Create 15 peers (more than the limit of 10 observers)
    peer_names = [test_peer.name]
    for _ in range(14):
        peer_name = str(generate_nanoid())
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/peers",
            json={"name": peer_name, "metadata": {}},
        )
        assert response.status_code in [200, 201]
        peer_names.append(peer_name)

    # Test 1: Adding 15 peers with observe_others=False should succeed
    peers_dict_no_observers: dict[str, dict[str, Any]] = {
        peer_name: {"observe_others": False} for peer_name in peer_names
    }
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=peers_dict_no_observers,
    )
    assert response.status_code == 200  # Should succeed since no observers

    # Test 2: Try to set 11 peers with observe_others=True (exceeds limit of 10)
    peers_dict_all_observers: dict[str, dict[str, Any]] = {
        peer_name: {"observe_others": True} for peer_name in peer_names[:11]
    }
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=peers_dict_all_observers,
    )
    assert response.status_code == 400  # ObserverException
    assert "11 observers" in response.json()["detail"]
    assert "Maximum allowed is 10 observers" in response.json()["detail"]

    # Test 3: Set exactly 10 observers should succeed
    peers_dict_ten_observers: dict[str, dict[str, Any]] = {
        peer_name: {"observe_others": True} for peer_name in peer_names[:10]
    }
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=peers_dict_ten_observers,
    )
    assert response.status_code == 200  # Should succeed with exactly 10 observers


def test_update_peer_config_observer_limit(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
):
    """Test that updating peer config respects observer limits"""
    test_workspace, test_peer = sample_data

    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
        },
    )
    assert response.status_code in [200, 201]

    # Create exactly 10 peers and add them as observers
    peer_names: list[str] = []
    for i in range(10):
        peer_name = f"observer_{i}"
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/peers",
            json={"name": peer_name, "metadata": {}},
        )
        assert response.status_code in [200, 201]
        peer_names.append(peer_name)

    # Add all 10 peers as observers
    peers_dict_observers: dict[str, dict[str, Any]] = {
        peer_name: {"observe_others": True} for peer_name in peer_names
    }
    # Also add the test_peer as non-observer
    peers_dict_observers[test_peer.name] = {"observe_others": False}

    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=peers_dict_observers,
    )
    assert response.status_code == 200  # Should succeed with exactly 10 observers

    # Now try to update test_peer to become an observer (would exceed limit)
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers/{test_peer.name}/config",
        json={"observe_others": True},
    )
    assert response.status_code == 400  # ObserverException
    assert "11 observers" in response.json()["detail"]
    assert "Maximum allowed is 10 observers" in response.json()["detail"]

    # Verify that updating a peer that's already an observer still works
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers/{peer_names[0]}/config",
        json={"observe_others": True, "observe_me": False},  # Still an observer
    )
    assert response.status_code == 204  # Should succeed since count doesn't change

    # Change one observer to non-observer
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers/{peer_names[0]}/config",
        json={"observe_others": False},
    )
    assert response.status_code == 204

    # Now test_peer can become an observer (9 + 1 = 10)
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers/{test_peer.name}/config",
        json={"observe_others": True},
    )
    assert response.status_code == 204  # Should succeed now


def test_remove_peers_from_session(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code in [200, 201]

    # Create a test session with multiple peers
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}, peer2_name: {}},
        },
    )
    assert response.status_code in [200, 201]

    # Remove one peer from the session
    response = client.request(
        "DELETE",
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=[test_peer.name],
    )
    assert response.status_code == 200

    # Check that only the remaining peer is in the session
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Add some messages to have context
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Test message 1", "peer_id": test_peer.name},
                {"content": "Test message 2", "peer_id": test_peer.name},
            ]
        },
    )

    # Get context
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context",
    )
    assert response.status_code == 200
    data = response.json()
    # SessionContext schema serializes "name" as "id"
    assert "id" in data
    assert "messages" in data
    assert "summary" in data
    assert data["id"] == session_id
    assert isinstance(data["messages"], list)
    assert data["summary"] is None  # No summary available


def test_get_session_context_with_summary(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with summary parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session with messages
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with summary
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context?summary=true",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with token limit
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context?tokens=100",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with all parameters
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context?tokens=100&summary=true",
    )
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert "summary" in data


def test_get_session_summaries(
    client: TestClient, sample_data: tuple[Workspace, Peer]
) -> None:
    """Test getting summaries for a valid session"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get summaries
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/summaries",
    )
    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "id" in data
    assert data["id"] == session_id
    assert "short_summary" in data
    assert "long_summary" in data
    # Summaries will be None since they're created asynchronously
    assert data["short_summary"] is None
    assert data["long_summary"] is None


def test_get_session_summaries_nonexistent_session(
    client: TestClient, sample_data: tuple[Workspace, Peer]
) -> None:
    """Test getting summaries for a non-existent session"""
    test_workspace, _ = sample_data
    nonexistent_session_id = str(generate_nanoid())

    # Try to get summaries for non-existent session
    # Should still return 200 with null summaries
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{nonexistent_session_id}/summaries",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == nonexistent_session_id
    assert data["short_summary"] is None
    assert data["long_summary"] is None


def test_search_session(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test the session search functionality"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Add messages to search through
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another message", "peer_id": test_peer.name},
            ]
        },
    )

    # Search with a query
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/search",
        json={"query": "search query", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_session_empty_query(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session search with empty query"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Search with empty query
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/search",
        json={"query": "", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_session_nonexistent(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test searching a session that doesn't exist"""
    test_workspace, _ = sample_data
    nonexistent_session_id = str(generate_nanoid())

    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{nonexistent_session_id}/search",
        json={"query": "test query", "limit": 10},
    )
    assert response.status_code == 200
    data: list[dict[str, Any]] = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_search_session_with_messages(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session search with actual messages"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Add messages to search through
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another message to find", "peer_id": test_peer.name},
            ]
        },
    )

    # Search for content
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/search",
        json={"query": "search", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_session_with_limit(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session search with custom limit"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Add messages to search through
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another message to find", "peer_id": test_peer.name},
                {"content": "More searchable content", "peer_id": test_peer.name},
            ]
        },
    )

    # Search with custom limit
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/search",
        json={"query": "search", "limit": 2},
    )

    assert response.status_code == 200
    data: list[dict[str, Any]] = response.json()
    assert isinstance(data, list)
    # Should not exceed the limit
    assert len(data) <= 2


def test_get_session_context_with_peer_target(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with peer_target parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Add some messages
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Test message 1", "peer_id": test_peer.name},
            ]
        },
    )

    # Get context with peer_target
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context?peer_target={test_peer.name}",
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "messages" in data
    assert "peer_representation" in data
    assert "peer_card" in data
    # Representation should be present
    assert data["peer_representation"] is not None
    assert isinstance(data["peer_representation"], str)


def test_get_session_context_with_peer_perspective(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with both peer_target and peer_perspective"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code in [200, 201]

    # Create session with both peers
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}, peer2_name: {}}},
    )

    # Get context with peer_perspective
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context?peer_target={test_peer.name}&peer_perspective={peer2_name}",
    )
    assert response.status_code == 200
    data = response.json()
    assert "peer_representation" in data
    assert "peer_card" in data


def test_get_session_context_peer_perspective_without_target_fails(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test that peer_perspective without peer_target raises validation error"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Try to get context with peer_perspective but no peer_target (should fail)
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context?peer_perspective={test_peer.name}",
    )
    # FastAPI returns 422 for validation errors, or 400 if it's a custom ValidationException
    assert response.status_code in [400, 422]
    error_detail = response.json()["detail"]
    assert "peer_target" in error_detail.lower()


def test_get_session_context_with_last_message(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with last_message parameter for semantic search"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with last_message and peer_target
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context",
        params={
            "peer_target": test_peer.name,
            "last_message": "What is my favorite color?",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "peer_representation" in data


def test_get_session_context_with_limit_to_session(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with limit_to_session parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with limit_to_session=true
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context",
        params={
            "peer_target": test_peer.name,
            "last_message": "Test query",
            "limit_to_session": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "peer_representation" in data


def test_get_session_context_with_search_parameters(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with search_top_k and search_max_distance parameters"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with search parameters
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context",
        params={
            "peer_target": test_peer.name,
            "last_message": "Test query",
            "search_top_k": 5,
            "search_max_distance": 0.8,  # float value (semantic distance 0.0-1.0)
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "peer_representation" in data


def test_get_session_context_with_include_most_frequent(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with include_most_frequent parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with include_most_frequent
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context",
        params={
            "peer_target": test_peer.name,
            "last_message": "Test query",
            "include_most_frequent": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "peer_representation" in data


def test_get_session_context_with_max_observations(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with max_observations parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Get context with max_observations
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context",
        params={
            "peer_target": test_peer.name,
            "last_message": "Test query",
            "max_observations": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "peer_representation" in data


def test_get_session_context_with_all_representation_params(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test session context with all representation-related parameters"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code in [200, 201]

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}, peer2_name: {}}},
    )

    # Get context with all representation parameters
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context",
        params={
            "tokens": 500,
            "peer_target": test_peer.name,
            "peer_perspective": peer2_name,
            "last_message": "What do you know about me?",
            "limit_to_session": True,
            "search_top_k": 10,
            "search_max_distance": 0.9,  # float value (semantic distance 0.0-1.0)
            "include_most_frequent": True,
            "max_observations": 15,
            "summary": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["id"] == session_id
    assert "messages" in data
    assert isinstance(data["messages"], list)
    assert "summary" in data
    assert "peer_representation" in data
    assert "peer_card" in data
    # Validate representation structure
    assert isinstance(data["peer_representation"], str)


def test_get_session_context_response_structure(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test that session context response has correct structure"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create session
    client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peers": {test_peer.name: {}}},
    )

    # Add messages
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": "Message 1", "peer_id": test_peer.name},
                {"content": "Message 2", "peer_id": test_peer.name},
            ]
        },
    )
    assert response.status_code == 201

    # Get context and validate response structure
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/context",
    )
    assert response.status_code == 200
    data = response.json()

    # Validate SessionContext schema
    assert "id" in data
    assert data["id"] == session_id
    assert "messages" in data
    assert isinstance(data["messages"], list)
    assert len(data["messages"]) >= 2

    # Validate Message schema
    for message in data["messages"]:
        assert "id" in message
        assert "content" in message
        assert "peer_id" in message
        assert "session_id" in message
        assert "workspace_id" in message
        assert "created_at" in message
        assert "token_count" in message

    # When no peer_target, these should not be present or be None
    assert data.get("peer_representation") is None
    assert data.get("peer_card") is None
