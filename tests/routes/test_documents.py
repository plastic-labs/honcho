def test_create_document(client, sample_data):
    test_app, test_user = sample_data
    # Create a collection
    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    # Create a document
    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{data['id']}/documents",
        json={"content": "test_text", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "test_text"
    assert data["metadata"] == {}
    assert "id" in data


def test_get_document(client, sample_data):
    test_app, test_user = sample_data
    # Create a collection
    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    collection = response.json()
    # Create a document
    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection['id']}/documents",
        json={"content": "test_text", "metadata": {}},
    )
    assert response.status_code == 200
    document = response.json()
    # Get the document
    response = client.get(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection['id']}/documents/{document['id']}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "test_text"
    assert data["metadata"] == {}
    assert "id" in data


def test_update_document(client, sample_data):
    test_app, test_user = sample_data
    # Create a collection
    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    # Create a document
    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{data['id']}/documents",
        json={"content": "test_text", "metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    # Update the document
    response = client.put(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{data['id']}/documents/{data['id']}",
        json={"content": "test_text_updated", "metadata": {"new_key": "new_value"}},
    )


def test_delete_document(client, sample_data):
    test_app, test_user = sample_data
    # Create a collection
    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )
    assert response.status_code == 200
    collection = response.json()
    # Create a document
    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection['id']}/documents",
        json={"content": "test_text", "metadata": {}},
    )
    assert response.status_code == 200
    document = response.json()
    # Delete the document
    response = client.delete(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection['id']}/documents/{document['id']}"
    )
    assert response.status_code == 200
    response = client.get(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection['id']}/documents/{document['id']}"
    )
    assert response.status_code == 404
