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


def test_update_peer(client, sample_data):
    test_workspace, test_peer = sample_data
    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


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
        json={
            "id": session_name,
            "peer_names": [test_peer.name]
        },
    )
    assert create_response.status_code == 200
    created_session = create_response.json()
    assert created_session["id"] == session_name

    # Now get sessions for the peer and validate the session is returned
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    print(data)
    assert "items" in data
    # Check that the created session is in the returned items
    session_ids = [item["id"] for item in data["items"]]
    assert session_name in session_ids
    assert response.status_code == 200
    data = response.json()
    assert "items" in data

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/sessions",
        json={},
    )
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
                    "metadata": {"message_key": "message_value"}
                },
                {
                    "content": "Second message",
                    "peer_id": test_peer.name,
                    "metadata": {"message_key": "message_value2"}
                }
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


def test_chat(client, sample_data):
    test_workspace, test_peer = sample_data
    target_peer = str(generate_nanoid())
    session_id = str(generate_nanoid())
    
    # Test chat endpoint
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat?session_id={session_id}&target={target_peer}",
        json={
            "queries": "Hello, how are you?",
            "stream": False
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    # The response should contain the workspace, peer, target, and session info
    assert test_workspace.name in data["content"]
    assert test_peer.name in data["content"]
    assert target_peer in data["content"]
    assert session_id in data["content"]
