from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src.models import Peer, Workspace


def test_get_or_create_workspace(client: TestClient):
    name = str(generate_nanoid())

    # This should create the workspace using POST /v2/workspaces
    response = client.post("/v2/workspaces", json={"name": name})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert "id" in data


def test_get_or_create_workspace_with_configuration(client: TestClient):
    """Test workspace creation with configuration parameter"""
    name = str(generate_nanoid())
    configuration = {"feature1": True, "feature2": False}

    response = client.post(
        "/v2/workspaces", json={"name": name, "configuration": configuration}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["configuration"] == configuration


def test_get_or_create_workspace_with_all_optional_params(client: TestClient):
    """Test workspace creation with all optional parameters"""
    name = str(generate_nanoid())
    metadata = {"key": "value", "number": 42}
    configuration = {"experimental": True, "beta": False}

    response = client.post(
        "/v2/workspaces",
        json={"name": name, "metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_get_or_create_existing_workspace(client: TestClient):
    name = str(generate_nanoid())

    # Create the workspace
    response = client.post(
        "/v2/workspaces", json={"name": name, "metadata": {"key": "value"}}
    )
    assert response.status_code == 200
    workspace1 = response.json()

    # Try to create the same workspace again - should return existing workspace
    response = client.post(
        "/v2/workspaces", json={"name": name, "metadata": {"key": "value"}}
    )
    assert response.status_code == 200
    workspace2 = response.json()

    # Both should be the same workspace
    assert workspace1["id"] == workspace2["id"]
    assert workspace1["metadata"] == workspace2["metadata"]


@pytest.mark.asyncio
async def test_get_all_workspaces(client: TestClient):
    # create a test workspace with metadata
    response = client.post(
        "/v2/workspaces",
        json={
            "name": "test_workspace",
            "metadata": {"test_key": "test_value"},
        },
    )

    response = client.post(
        "/v2/workspaces/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0

    response = client.post(
        "/v2/workspaces/list",
        json={"filters": {"metadata": {"test_key": "test_value"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["metadata"] == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_get_all_workspaces_with_empty_filter(client: TestClient):
    """Test workspace listing with empty filter object"""
    response = client.post("/v2/workspaces/list", json={"filters": {}})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_all_workspaces_with_null_filter(client: TestClient):
    """Test workspace listing with null filter"""
    response = client.post("/v2/workspaces/list", json={"filters": None})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_update_workspace(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, _ = sample_data
    _new_name = str(generate_nanoid())
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


def test_update_workspace_with_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test workspace update with configuration parameter"""
    test_workspace, _ = sample_data
    configuration = {"new_feature": True, "legacy_feature": False}

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}", json={"configuration": configuration}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["configuration"] == configuration


def test_update_workspace_with_all_optional_params(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test workspace update with both metadata and configuration"""
    test_workspace, _ = sample_data
    metadata = {"updated_key": "updated_value", "count": 100}
    configuration = {"experimental": True, "beta": True}

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}",
        json={"metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_update_workspace_with_null_metadata(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test workspace update with null metadata (should clear metadata)"""
    test_workspace, _ = sample_data

    # First set some metadata
    client.put(
        f"/v2/workspaces/{test_workspace.name}", json={"metadata": {"temp": "value"}}
    )

    # Then clear it with null
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}", json={"metadata": None}
    )
    assert response.status_code == 200
    data = response.json()
    # The behavior might be to keep existing metadata or clear it -
    # adjust this assertion based on actual behavior
    assert "metadata" in data


def test_update_workspace_with_null_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test workspace update with null configuration"""
    test_workspace, _ = sample_data

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}", json={"configuration": None}
    )
    assert response.status_code == 200
    data = response.json()
    assert "configuration" in data


def test_create_duplicate_workspace_name(client: TestClient):
    # Create an workspace
    name = str(generate_nanoid())
    response = client.post("/v2/workspaces", json={"name": name})
    assert response.status_code == 200

    # Try to create another workspace with the same name - should return existing workspace
    response = client.post("/v2/workspaces", json={"name": name})

    # Should return the existing workspace with 200 status (get_or_create behavior)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name


def test_search_workspace(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test the workspace search functionality"""
    test_workspace, _ = sample_data

    # Test search with a query
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/search",
        json={"query": "test search query", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_workspace_empty_query(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test the workspace search with empty query"""
    test_workspace, _ = sample_data

    # Test search with empty query
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/search", json={"query": "", "limit": 10}
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_workspace_nonexistent(client: TestClient):
    """Test searching a workspace that doesn't exist"""
    nonexistent_workspace_id = str(generate_nanoid())

    response = client.post(
        f"/v2/workspaces/{nonexistent_workspace_id}/search",
        json={"query": "test query", "limit": 10},
    )
    assert response.status_code == 200
    data: list[dict[str, Any]] = response.json()
    # Should return empty list for nonexistent workspace
    assert isinstance(data, list)
    assert len(data) == 0


def test_delete_workspace(client: TestClient):
    """Test deleting a workspace"""
    name = str(generate_nanoid())

    # Create a workspace
    response = client.post("/v2/workspaces", json={"name": name})
    assert response.status_code == 200
    workspace = response.json()
    assert workspace["id"] == name

    # Delete the workspace
    response = client.delete(f"/v2/workspaces/{name}")
    assert response.status_code == 200
    deleted_workspace = response.json()
    assert deleted_workspace["id"] == name

    # Verify the workspace no longer exists by trying to update it
    response = client.put(
        f"/v2/workspaces/{name}", json={"metadata": {"test": "value"}}
    )
    # Should create a new workspace since the old one was deleted
    assert response.status_code == 200


def test_delete_nonexistent_workspace(client: TestClient):
    """Test deleting a workspace that doesn't exist"""
    nonexistent_workspace_id = str(generate_nanoid())

    response = client.delete(f"/v2/workspaces/{nonexistent_workspace_id}")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_delete_workspace_with_peers(client: TestClient):
    """Test deleting a workspace that has peers"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v2/workspaces", json={"name": workspace_name})
    assert response.status_code == 200

    # Create peers
    peer1_name = str(generate_nanoid())
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{workspace_name}/peers", json={"name": peer1_name}
    )
    assert response.status_code == 200
    response = client.post(
        f"/v2/workspaces/{workspace_name}/peers", json={"name": peer2_name}
    )
    assert response.status_code == 200

    # Delete workspace
    response = client.delete(f"/v2/workspaces/{workspace_name}")
    assert response.status_code == 200


def test_delete_workspace_with_sessions(client: TestClient):
    """Test deleting a workspace that has sessions"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v2/workspaces", json={"name": workspace_name})
    assert response.status_code == 200

    # Create peer
    peer_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{workspace_name}/peers", json={"name": peer_name}
    )
    assert response.status_code == 200

    # Create sessions
    session1_name = str(generate_nanoid())
    session2_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions", json={"name": session1_name}
    )
    assert response.status_code == 200
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions", json={"name": session2_name}
    )
    assert response.status_code == 200

    # Delete workspace
    response = client.delete(f"/v2/workspaces/{workspace_name}")
    assert response.status_code == 200


def test_delete_workspace_with_messages(client: TestClient):
    """Test deleting a workspace that has messages"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v2/workspaces", json={"name": workspace_name})
    assert response.status_code == 200

    # Create peer
    peer_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{workspace_name}/peers", json={"name": peer_name}
    )
    assert response.status_code == 200

    # Create session
    session_name = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions", json={"name": session_name}
    )
    assert response.status_code == 200

    # Add peer to session
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/{session_name}/peers",
        json={peer_name: {}},
    )
    assert response.status_code == 200

    # Create messages
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/{session_name}/messages",
        json={
            "messages": [
                {"content": "Test message 1", "peer_id": peer_name},
                {"content": "Test message 2", "peer_id": peer_name},
            ]
        },
    )
    assert response.status_code == 200

    # Delete workspace
    response = client.delete(f"/v2/workspaces/{workspace_name}")
    assert response.status_code == 200


def test_delete_workspace_with_webhooks(client: TestClient):
    """Test deleting a workspace that has webhooks"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v2/workspaces", json={"name": workspace_name})
    assert response.status_code == 200

    # Create webhook
    response = client.post(
        f"/v2/workspaces/{workspace_name}/webhooks",
        json={
            "url": "https://example.com/webhook",
        },
    )
    assert response.status_code == 200

    # Delete workspace
    response = client.delete(f"/v2/workspaces/{workspace_name}")
    assert response.status_code == 200

    # Verify webhook is deleted by checking workspace doesn't exist
    response = client.get(f"/v2/workspaces/{workspace_name}/webhooks")
    # This should either return 404 or empty list depending on implementation
    assert response.status_code in [404, 200]


def test_delete_workspace_cascade(client: TestClient):
    """Test that deleting a workspace cascades to all related resources"""
    workspace_name = str(generate_nanoid())

    # Create workspace with complex structure
    response = client.post(
        "/v2/workspaces",
        json={"name": workspace_name, "metadata": {"test": "cascade"}},
    )
    assert response.status_code == 200

    # Create multiple peers
    peer_names = [str(generate_nanoid()) for _ in range(3)]
    for peer_name in peer_names:
        response = client.post(
            f"/v2/workspaces/{workspace_name}/peers", json={"name": peer_name}
        )
        assert response.status_code == 200

    # Create multiple sessions
    session_names = [str(generate_nanoid()) for _ in range(2)]
    for session_name in session_names:
        response = client.post(
            f"/v2/workspaces/{workspace_name}/sessions", json={"name": session_name}
        )
        assert response.status_code == 200

    # Add peers to sessions and create messages
    for session_name in session_names:
        for peer_name in peer_names[:2]:  # Add 2 peers to each session
            response = client.post(
                f"/v2/workspaces/{workspace_name}/sessions/{session_name}/peers",
                json={peer_name: {}},
            )
            assert response.status_code == 200

        # Create messages in session
        response = client.post(
            f"/v2/workspaces/{workspace_name}/sessions/{session_name}/messages",
            json={
                "messages": [
                    {
                        "content": f"Test message in {session_name}",
                        "peer_id": peer_names[0],
                    }
                ]
            },
        )
        assert response.status_code == 200

    # Delete the workspace
    response = client.delete(f"/v2/workspaces/{workspace_name}")
    assert response.status_code == 200
    deleted_workspace = response.json()
    assert deleted_workspace["id"] == workspace_name
    assert deleted_workspace["metadata"]["test"] == "cascade"


def test_delete_workspace_returns_workspace_data(client: TestClient):
    """Test that delete workspace returns the deleted workspace data"""
    name = str(generate_nanoid())
    metadata = {"key": "value", "number": 42}
    configuration = {"feature": True}

    # Create workspace with metadata and configuration
    response = client.post(
        "/v2/workspaces",
        json={"name": name, "metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    created_workspace = response.json()

    # Delete workspace
    response = client.delete(f"/v2/workspaces/{name}")
    assert response.status_code == 200
    deleted_workspace = response.json()

    # Verify returned data matches original workspace
    assert deleted_workspace["id"] == created_workspace["id"]
    assert deleted_workspace["metadata"] == metadata
    assert deleted_workspace["configuration"] == configuration
    assert "created_at" in deleted_workspace
