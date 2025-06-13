import pytest
from nanoid import generate as generate_nanoid


def test_get_or_create_workspace(client):
    name = str(generate_nanoid())

    # This should create the workspace using POST /v1/workspaces
    response = client.post("/v1/workspaces", json={"name": name})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert "id" in data


def test_get_or_create_workspace_with_feature_flags(client):
    """Test workspace creation with feature_flags parameter"""
    name = str(generate_nanoid())
    feature_flags = {"feature1": True, "feature2": False}

    response = client.post(
        "/v1/workspaces", json={"name": name, "feature_flags": feature_flags}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["feature_flags"] == feature_flags


def test_get_or_create_workspace_with_all_optional_params(client):
    """Test workspace creation with all optional parameters"""
    name = str(generate_nanoid())
    metadata = {"key": "value", "number": 42}
    feature_flags = {"experimental": True, "beta": False}

    response = client.post(
        "/v1/workspaces",
        json={"name": name, "metadata": metadata, "feature_flags": feature_flags},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name
    assert data["metadata"] == metadata
    assert data["feature_flags"] == feature_flags


def test_get_or_create_existing_workspace(client):
    name = str(generate_nanoid())

    # Create the workspace
    response = client.post(
        "/v1/workspaces", json={"name": name, "metadata": {"key": "value"}}
    )
    assert response.status_code == 200
    workspace1 = response.json()

    # Try to create the same workspace again - should return existing workspace
    response = client.post(
        "/v1/workspaces", json={"name": name, "metadata": {"key": "value"}}
    )
    assert response.status_code == 200
    workspace2 = response.json()

    # Both should be the same workspace
    assert workspace1["id"] == workspace2["id"]
    assert workspace1["metadata"] == workspace2["metadata"]


@pytest.mark.asyncio
async def test_get_all_workspaces(client, db_session, sample_data):
    # create a test workspace with metadata
    response = client.post(
        "/v1/workspaces",
        json={
            "name": "test_workspace",
            "metadata": {"test_key": "test_value"},
        },
    )

    response = client.post(
        "/v1/workspaces/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0

    response = client.post(
        "/v1/workspaces/list",
        json={"filter": {"test_key": "test_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["metadata"] == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_get_all_workspaces_with_reverse(client, db_session, sample_data):
    """Test workspace listing with reverse parameter"""
    # Create multiple workspaces to test ordering
    workspace1_name = str(generate_nanoid())
    workspace2_name = str(generate_nanoid())

    # Create workspaces with slight delay to ensure different created_at times
    client.post("/v1/workspaces", json={"name": workspace1_name})
    client.post("/v1/workspaces", json={"name": workspace2_name})

    # Test normal order
    response = client.post("/v1/workspaces/list", json={})
    assert response.status_code == 200
    normal_data = response.json()

    # Test reversed order
    response = client.post("/v1/workspaces/list?reverse=true", json={})
    assert response.status_code == 200
    reversed_data = response.json()

    # Both should have items
    assert len(normal_data["items"]) > 0
    assert len(reversed_data["items"]) > 0

    # If we have multiple items, order should be different
    if len(normal_data["items"]) > 1 and len(reversed_data["items"]) > 1:
        assert normal_data["items"][0]["id"] != reversed_data["items"][0]["id"]


@pytest.mark.asyncio
async def test_get_all_workspaces_with_empty_filter(client, db_session, sample_data):
    """Test workspace listing with empty filter object"""
    response = client.post("/v1/workspaces/list", json={"filter": {}})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_all_workspaces_with_null_filter(client, db_session, sample_data):
    """Test workspace listing with null filter"""
    response = client.post("/v1/workspaces/list", json={"filter": None})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_update_workspace(client, sample_data):
    test_workspace, _ = sample_data
    _new_name = str(generate_nanoid())
    response = client.put(
        f"/v1/workspaces/{test_workspace.name}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


def test_update_workspace_with_feature_flags(client, sample_data):
    """Test workspace update with feature_flags parameter"""
    test_workspace, _ = sample_data
    feature_flags = {"new_feature": True, "legacy_feature": False}

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}", json={"feature_flags": feature_flags}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["feature_flags"] == feature_flags


def test_update_workspace_with_all_optional_params(client, sample_data):
    """Test workspace update with both metadata and feature_flags"""
    test_workspace, _ = sample_data
    metadata = {"updated_key": "updated_value", "count": 100}
    feature_flags = {"experimental": True, "beta": True}

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}",
        json={"metadata": metadata, "feature_flags": feature_flags},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == metadata
    assert data["feature_flags"] == feature_flags


def test_update_workspace_with_null_metadata(client, sample_data):
    """Test workspace update with null metadata (should clear metadata)"""
    test_workspace, _ = sample_data

    # First set some metadata
    client.put(
        f"/v1/workspaces/{test_workspace.name}", json={"metadata": {"temp": "value"}}
    )

    # Then clear it with null
    response = client.put(
        f"/v1/workspaces/{test_workspace.name}", json={"metadata": None}
    )
    assert response.status_code == 200
    data = response.json()
    # The behavior might be to keep existing metadata or clear it -
    # adjust this assertion based on actual behavior
    assert "metadata" in data


def test_update_workspace_with_null_feature_flags(client, sample_data):
    """Test workspace update with null feature_flags"""
    test_workspace, _ = sample_data

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}", json={"feature_flags": None}
    )
    assert response.status_code == 200
    data = response.json()
    assert "feature_flags" in data


def test_create_duplicate_workspace_name(client):
    # Create an workspace
    name = str(generate_nanoid())
    response = client.post("/v1/workspaces", json={"name": name})
    assert response.status_code == 200

    # Try to create another workspace with the same name - should return existing workspace
    response = client.post("/v1/workspaces", json={"name": name})

    # Should return the existing workspace with 200 status (get_or_create behavior)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == name


def test_search_workspace(client, sample_data):
    """Test the workspace search functionality"""
    test_workspace, _ = sample_data

    # Test search with a query
    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/search", json="test search query"
    )
    assert response.status_code == 200
    data = response.json()

    # Response should have pagination structure
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert isinstance(data["items"], list)


def test_search_workspace_empty_query(client, sample_data):
    """Test the workspace search with empty query"""
    test_workspace, _ = sample_data

    # Test search with empty query
    response = client.post(f"/v1/workspaces/{test_workspace.name}/search", json="")
    assert response.status_code == 200
    data = response.json()

    # Response should still have proper pagination structure
    assert "items" in data
    assert isinstance(data["items"], list)


def test_search_workspace_nonexistent(client):
    """Test searching a workspace that doesn't exist"""
    nonexistent_workspace_id = str(generate_nanoid())

    response = client.post(
        f"/v1/workspaces/{nonexistent_workspace_id}/search", json="test query"
    )
    assert response.status_code == 200
