from nanoid import generate as generate_nanoid

def test_get_or_create_session(client, sample_data):
    # Test get or create session
    test_workspace, test_peer = sample_data

    # Test creating a new session with no parameters
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={"id": str(generate_nanoid())},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["id"], str)
    assert data["metadata"] == {}

    # Test creating a session with a specific id and peer_names (should get or create)
    session_id = str(generate_nanoid())
    response2 = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name]
        },
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["id"] == session_id
    assert data2["metadata"] == {}

    # Test getting the same session again (should return the same session)
    response3 = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name]
        },
    )
    assert response3.status_code == 200
    data3 = response3.json()
    assert data3["id"] == session_id
    assert data3["metadata"] == {}


def test_create_session_with_metadata(client, sample_data):
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
            "metadata": {"session_key": "session_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"session_key": "session_value"}
    assert "id" in data
    assert data["id"] == session_id


def test_get_sessions(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
            "metadata": {"test_key": "test_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test_key": "test_value"}
    assert "id" in data

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"test_key": "test_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["metadata"] == {"test_key": "test_value"}


def test_empty_update_session(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
        },
    )
    assert response.status_code == 200

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={},
    )
    assert response.status_code == 422


def test_update_delete_metadata(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
            "metadata": {"default": "value"},
        },
    )
    assert response.status_code == 200

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {}


def test_update_session(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
        },
    )
    assert response.status_code == 200

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


def test_delete_session(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
        },
    )
    assert response.status_code == 200
    
    response = client.delete(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}"
    )
    assert response.status_code == 200
    
    # Check that session is marked as inactive
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions/list",
        json={"is_active": False},
    )
    data = response.json()
    # Find our session in the inactive sessions
    inactive_session = next((s for s in data["items"] if s["id"] == session_id), None)
    assert inactive_session is not None
    assert inactive_session["is_active"] is False


def test_clone_session(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
            "metadata": {"test": "key"},
        },
    )
    assert response.status_code == 200

    # Create some messages in the session
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Test message",
                    "peer_id": test_peer.name,
                    "metadata": {"key": "value"}
                },
                {
                    "content": "Test message 2", 
                    "peer_id": test_peer.name,
                    "metadata": {"key": "value2"}
                }
            ]
        },
    )
    assert response.status_code == 200

    response = client.get(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}/clone",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test": "key"}

    # Check messages were cloned
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions/{data['id']}/messages/list",
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


def test_add_peers_to_session(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code == 200

    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
        },
    )
    assert response.status_code == 200

    # Add another peer to the session
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=[peer2_name],
    )
    assert response.status_code == 200


def test_get_session_peers(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code == 200

    # Create a test session with multiple peers
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name, peer2_name],
        },
    )
    assert response.status_code == 200

    # Get peers from the session
    response = client.get(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    peer_names = [peer["id"] for peer in data["items"]]
    assert test_peer.name in peer_names
    assert peer2_name in peer_names


def test_set_session_peers(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code == 200

    # Create a test session
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name],
        },
    )
    assert response.status_code == 200

    # Set peers for the session (should replace existing peers)
    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=[peer2_name],
    )
    assert response.status_code == 200

    # Check that only the new peer is in the session
    response = client.get(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer2_name


def test_remove_peers_from_session(client, sample_data):
    test_workspace, test_peer = sample_data
    # Create another peer
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {}},
    )
    assert response.status_code == 200

    # Create a test session with multiple peers
    session_id = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": [test_peer.name, peer2_name],
        },
    )
    assert response.status_code == 200

    # Remove one peer from the session
    response = client.request(
        "DELETE",
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
        json=[test_peer.name],
    )
    assert response.status_code == 200

    # Check that only the remaining peer is in the session
    response = client.get(
        f"/v1/workspaces/{test_workspace.name}/sessions/{session_id}/peers",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer2_name
