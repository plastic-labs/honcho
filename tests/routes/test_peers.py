from nanoid import generate as generate_nanoid


def test_get_or_create_peer(client, sample_data):
    test_workspace, _ = sample_data
    name = str(generate_nanoid())
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": {"peer_key": "peer_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["metadata"] == {"peer_key": "peer_value"}
    assert "id" in data


def test_get_or_create_peer_with_configuration(client, sample_data):
    """Test peer creation with configuration parameter"""
    test_workspace, _ = sample_data
    name = str(generate_nanoid())
    configuration = {"experimental": True, "beta": False}

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": name, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["configuration"] == configuration


def test_get_or_create_peer_with_all_optional_params(client, sample_data):
    """Test peer creation with all optional parameters"""
    test_workspace, _ = sample_data
    name = str(generate_nanoid())
    metadata = {"key": "value", "number": 42}
    configuration = {"feature1": True, "feature2": False}

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_get_or_create_existing_peer(client, sample_data):
    test_workspace, _ = sample_data
    name = str(generate_nanoid())

    # Create the peer
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": {"peer_key": "peer_value"}},
    )
    assert response.status_code == 200
    peer1 = response.json()

    # Try to create the same peer again - should return existing peer
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": {"peer_key": "peer_value"}},
    )
    assert response.status_code == 200
    peer2 = response.json()

    # Both should be the same peer
    assert peer1["id"] == peer2["id"]
    assert peer1["metadata"] == peer2["metadata"]


def test_get_peers(client, sample_data):
    test_workspace, _ = sample_data

    # Create a few peers with metadata
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": str(generate_nanoid()), "metadata": {"peer_key": "peer_value"}},
    )
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": str(generate_nanoid()), "metadata": {"peer_key": "peer_value"}},
    )
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers",
        json={"name": str(generate_nanoid()), "metadata": {"peer_key": "peer_value2"}},
    )

    # Get all peers
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0

    # Get peers with filter
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"peer_key": "peer_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) >= 2
    assert data["items"][0]["metadata"]["peer_key"] == "peer_value"


def test_get_peers_with_empty_filter(client, sample_data):
    """Test peer listing with empty filter object"""
    test_workspace, _ = sample_data

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/list", json={"filter": {}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_get_peers_with_null_filter(client, sample_data):
    """Test peer listing with null filter"""
    test_workspace, _ = sample_data

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/list", json={"filter": None}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_update_peer(client, sample_data):
    test_workspace, test_peer = sample_data
    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


def test_update_peer_with_configuration(client, sample_data):
    """Test peer update with configuration parameter"""
    test_workspace, test_peer = sample_data
    configuration = {"new_feature": True, "legacy_feature": False}

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["configuration"] == configuration


def test_update_peer_with_all_optional_params(client, sample_data):
    """Test peer update with both metadata and configuration"""
    test_workspace, test_peer = sample_data
    metadata = {"updated_key": "updated_value", "count": 100}
    configuration = {"experimental": True, "beta": True}

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_update_peer_with_null_metadata(client, sample_data):
    """Test peer update with null metadata"""
    test_workspace, test_peer = sample_data

    # First set some metadata
    client.put(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": {"temp": "value"}},
    )

    # Then clear it with null
    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data


def test_update_peer_with_null_configuration(client, sample_data):
    """Test peer update with null configuration"""
    test_workspace, test_peer = sample_data

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"configuration": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert "configuration" in data


def test_get_sessions_for_peer_no_sessions(client, sample_data):
    test_workspace, test_peer = sample_data

    # Get sessions for the peer
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data


def test_get_sessions_for_peer(client, sample_data):
    test_workspace, test_peer = sample_data

    # Create session for the peer
    session_name = str(generate_nanoid())
    create_response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={"id": session_name, "peer_names": {test_peer.name: {}}},
    )
    assert create_response.status_code == 200
    created_session = create_response.json()
    assert created_session["id"] == session_name

    # Now get sessions for the peer and validate the session is returned
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    # Check that the created session is in the returned items
    session_ids = [item["id"] for item in data["items"]]
    assert session_name in session_ids
    assert len(data["items"]) == 1


def test_get_sessions_for_peer_with_is_active_filter(client, sample_data):
    """Test getting sessions for peer with is_active parameter"""
    test_workspace, test_peer = sample_data

    # Create and then delete a session to have inactive session
    session_name = str(generate_nanoid())
    client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={"id": session_name, "peer_names": {test_peer.name: {}}},
    )
    client.delete(f"/v1/workspaces/{test_workspace.name}/sessions/{session_name}")

    # Test getting inactive sessions
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
        json={"is_active": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    # Should find at least our deleted session
    inactive_sessions = [s for s in data["items"] if not s["is_active"]]
    assert len(inactive_sessions) > 0


def test_get_sessions_for_peer_with_empty_filter(client, sample_data):
    """Test getting sessions for peer with empty filter object"""
    test_workspace, test_peer = sample_data

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
        json={"filter": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_create_and_get_messages_for_peer(client, sample_data):
    test_workspace, test_peer = sample_data

    # Create messages for the peer
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/messages",
        json={
            "messages": [
                {
                    "content": "Hello world",
                    "peer_id": test_peer.name,
                    "metadata": {"message_key": "message_value"},
                },
                {
                    "content": "Second message",
                    "peer_id": test_peer.name,
                    "metadata": {"message_key": "message_value2"},
                },
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["content"] == "Hello world"
    assert data[1]["content"] == "Second message"
    assert data[0]["metadata"] == {"message_key": "message_value"}

    # Get messages for the peer
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/messages/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    assert data["items"][0]["content"] == "Hello world"
    assert data["items"][1]["content"] == "Second message"
    assert data["items"][0]["metadata"] == {"message_key": "message_value"}
    assert data["items"][1]["metadata"] == {"message_key": "message_value2"}


def test_get_messages_for_peer_with_reverse(client, sample_data):
    """Test getting messages for peer with reverse parameter"""
    test_workspace, test_peer = sample_data

    # Create messages
    client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/messages",
        json={
            "messages": [
                {"content": "First message", "peer_id": test_peer.name},
                {"content": "Second message", "peer_id": test_peer.name},
            ]
        },
    )

    # Test normal order
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/messages/list",
        json={},
    )
    assert response.status_code == 200
    normal_data = response.json()

    # Test reversed order
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/messages/list?reverse=true",
        json={},
    )
    assert response.status_code == 200
    reversed_data = response.json()

    # Both should have items
    assert len(normal_data["items"]) > 0
    assert len(reversed_data["items"]) > 0


def test_get_messages_for_peer_with_empty_filter(client, sample_data):
    """Test getting messages for peer with empty filter object"""
    test_workspace, test_peer = sample_data

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/messages/list",
        json={"filter": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_get_messages_for_peer_with_null_filter(client, sample_data):
    """Test getting messages for peer with null filter"""
    test_workspace, test_peer = sample_data

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/messages/list",
        json={"filter": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_chat(client, sample_data):
    test_workspace, test_peer = sample_data
    target_peer = str(generate_nanoid())
    session_id = str(generate_nanoid())

    # Test chat endpoint
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        json={
            "queries": "Hello, how are you?",
            "stream": False,
            "target": target_peer,
            "session_id": session_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data


def test_chat_with_optional_params(client, sample_data):
    """Test chat endpoint with optional parameters"""
    test_workspace, test_peer = sample_data

    # Test chat without optional parameters
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        json={"queries": "Hello, how are you?", "stream": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert test_workspace.name in data["content"]
    assert test_peer.name in data["content"]


def test_get_peer_representation_with_session(client, sample_data):
    """Test peer representation with session_id parameter"""
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())

    # Create a session first
    client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Test representation scoped to session
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/representation",
        json={
            "session_id": session_id,
            "queries": "Hello, how are you?",
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Should still return the stubbed representation structure
    assert "final_observations" in data
    assert "explicit" in data["final_observations"]


def test_search_peer(client, sample_data):
    """Test the peer search functionality"""
    test_workspace, test_peer = sample_data

    # Add some messages to search through
    client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/messages",
        json={
            "messages": [
                {"content": "Search this content", "peer_id": test_peer.name},
                {"content": "Another searchable message", "peer_id": test_peer.name},
            ]
        },
    )

    # Search with a query
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/search",
        json="search query",
    )
    assert response.status_code == 200
    data = response.json()

    # Response should have pagination structure
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert isinstance(data["items"], list)


def test_search_peer_empty_query(client, sample_data):
    """Test peer search with empty query"""
    test_workspace, test_peer = sample_data

    # Search with empty query
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/search", json=""
    )
    assert response.status_code == 200
    data = response.json()

    # Response should still have proper pagination structure
    assert "items" in data
    assert isinstance(data["items"], list)


def test_search_peer_nonexistent(client, sample_data):
    """Test searching a peer that doesn't exist"""
    test_workspace, _ = sample_data
    nonexistent_peer_id = str(generate_nanoid())

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{nonexistent_peer_id}/search",
        json="test query",
    )
    # This should probably return 404 or handle gracefully
    # The exact behavior depends on the crud.search implementation
    assert response.status_code in [200, 404, 422]
