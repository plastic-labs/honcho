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


def test_update_workspace(client, sample_data):
    test_workspace, _ = sample_data
    new_name = str(generate_nanoid())
    response = client.put(
        f"/v1/workspaces/{test_workspace.name}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


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
