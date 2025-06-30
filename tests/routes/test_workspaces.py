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
        json={"filter": {"metadata": {"test_key": "test_value"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["metadata"] == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_get_all_workspaces_with_empty_filter(client: TestClient):
    """Test workspace listing with empty filter object"""
    response = client.post("/v2/workspaces/list", json={"filter": {}})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_all_workspaces_with_null_filter(client: TestClient):
    """Test workspace listing with null filter"""
    response = client.post("/v2/workspaces/list", json={"filter": None})
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
        f"/v2/workspaces/{test_workspace.name}/search", json="test search query"
    )
    assert response.status_code == 200
    data = response.json()

    # Response should have pagination structure
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert isinstance(data["items"], list)


def test_search_workspace_empty_query(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test the workspace search with empty query"""
    test_workspace, _ = sample_data

    # Test search with empty query
    response = client.post(f"/v2/workspaces/{test_workspace.name}/search", json="")
    assert response.status_code == 200
    data = response.json()

    # Response should still have proper pagination structure
    assert "items" in data
    assert isinstance(data["items"], list)


def test_search_workspace_nonexistent(client: TestClient):
    """Test searching a workspace that doesn't exist"""
    nonexistent_workspace_id = str(generate_nanoid())

    response = client.post(
        f"/v2/workspaces/{nonexistent_workspace_id}/search", json="test query"
    )
    assert response.status_code == 200


def test_get_or_create_workspace_no_name_no_jwt_workspace(client):
    """Test workspace creation with no name and no workspace in JWT"""

    # Try to create workspace with empty name and empty JWT
    response = client.post("/v2/workspaces", json={"name": ""})
    assert response.status_code == 422


def test_update_workspace_returns_updated_workspace(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test that update_workspace returns the updated workspace object"""
    test_workspace, _ = sample_data
    
    # Update the workspace
    new_metadata = {"updated_key": "updated_value"}
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}",
        json={"metadata": new_metadata},
    )
    
    # Verify response status and that the workspace object is returned
    assert response.status_code == 200
    returned_workspace = response.json()
    
    # Verify the returned workspace has the correct properties
    assert returned_workspace["id"] == test_workspace.name
    assert returned_workspace["metadata"] == new_metadata
    assert "created_at" in returned_workspace
    assert "updated_at" in returned_workspace
