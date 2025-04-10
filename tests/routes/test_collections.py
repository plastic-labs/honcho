import pytest
from nanoid import generate as generate_nanoid


def test_create_collection(client, sample_data) -> None:
    test_app, test_user = sample_data
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_collection"
    assert data["metadata"] == {}
    assert "id" in data


def test_get_collection_by_id(client, sample_data) -> None:
    test_app, test_user = sample_data
    # Make the collection
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    # Get the collection
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections?collection_id={data['id']}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_collection"
    assert data["metadata"] == {}
    assert "id" in data


def test_get_collection_by_name(client, sample_data) -> None:
    test_app, test_user = sample_data
    # Make the collection
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    # Get the collection
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/name/test_collection"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_collection"
    assert data["metadata"] == {}
    assert "id" in data


def test_get_collections(client, sample_data) -> None:
    test_app, test_user = sample_data
    # Make Sample Collections
    client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": str(generate_nanoid()), "metadata": {"test": "key"}},
    )
    client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": str(generate_nanoid()), "metadata": {"test": "key"}},
    )
    client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": str(generate_nanoid()), "metadata": {"test": "key2"}},
    )

    # Get the Collections
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/list",
        json={"filter": {"test": "key"}},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["items"][0]["metadata"] == {"test": "key"}
    assert data["items"][1]["metadata"] == {"test": "key"}


def test_update_collection(client, sample_data) -> None:
    test_app, test_user = sample_data
    # Make the collection
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    # Update the collection
    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{data['id']}",
        json={"name": "test_collection_updated", "metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_collection_updated"
    assert data["metadata"] == {"new_key": "new_value"}
    assert "id" in data


def test_delete_collection(client, sample_data) -> None:
    test_app, test_user = sample_data
    # Make the collection
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    # Delete the collection
    response = client.delete(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{data['id']}"
    )
    assert response.status_code == 200
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections?collection_id={data['id']}"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cannot_delete_honcho_collection(client, sample_data, db_session) -> None:
    test_app, test_user = sample_data
    # Make the protected "honcho" collection
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "honcho", "metadata": {}},
    )
    assert response.status_code == 422  # Should get validation error when trying to create

    # Try to create it directly through API to test delete protection
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    collection_id = data["id"]
    
    # Update the collection to have the reserved name
    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}",
        json={"name": "honcho", "metadata": {}},
    )
    assert response.status_code == 422  # Should get validation error
    
    # Create the protected collection using the internal method directly with db_session
    from src.crud import create_user_protected_collection
    
    # Create the protected collection
    honcho_collection = await create_user_protected_collection(
        db_session, app_id=test_app.public_id, user_id=test_user.public_id
    )
    await db_session.flush()
    
    # Get the protected collection
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/name/honcho"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "honcho"
    protected_id = data["id"]
    
    # Try to delete the protected collection
    response = client.delete(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{protected_id}"
    )
    assert response.status_code == 422  # Should get validation error
    assert "reserved name" in response.json()["detail"].lower()
