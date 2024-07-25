import uuid


def test_create_user(client, sample_data):
    test_app, _ = sample_data
    name = str(uuid.uuid4())
    response = client.post(
        f"/apps/{test_app.id}/users",
        json={"name": name, "metadata": {"user_key": "user_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == name
    assert data["metadata"] == {"user_key": "user_value"}
    assert "id" in data


def test_get_user_by_id(client, sample_data):
    test_app, test_user = sample_data
    response = client.get(f"/apps/{test_app.id}/users/{test_user.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_user.name
    assert data["id"] == str(test_user.id)


def test_get_user_by_name(client, sample_data):
    test_app, test_user = sample_data
    response = client.get(f"/apps/{test_app.id}/users/name/{test_user.name}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_user.name
    assert data["id"] == str(test_user.id)


def test_get_or_create_user(client, sample_data):
    test_app, _ = sample_data
    name = str(uuid.uuid4())
    response = client.get(f"/apps/{test_app.id}/users/name/{name}")
    assert response.status_code == 404
    response = client.get(f"/apps/{test_app.id}/users/get_or_create/{name}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == name
    assert "id" in data


# def test_get_users(client, sample_data):
#     test_app, _ = sample_data
#     response = client.get(f"/apps/{test_app.id}/users")
#     assert response.status_code == 200
#     data = response.json()
#     assert "items" in data
#     assert len(data["items"]) > 0


def test_update_user(client, sample_data):
    test_app, test_user = sample_data
    new_name = str(uuid.uuid4())
    response = client.put(
        f"/apps/{test_app.id}/users/{test_user.id}",
        json={"name": new_name, "metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    print(new_name)
    print(data)
    assert data["name"] == new_name
    assert data["metadata"] == {"new_key": "new_value"}
