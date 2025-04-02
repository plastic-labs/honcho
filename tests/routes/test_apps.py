from nanoid import generate as generate_nanoid


def test_create_app(client):
    name = str(generate_nanoid())
    response = client.post(
        "/v1/apps", json={"name": name, "metadata": {"key": "value"}}
    )
    print(response)
    assert response.status_code == 200
    data = response.json()
    print("===================")
    print(data)
    print("===================")
    assert data["name"] == name
    assert data["metadata"] == {"key": "value"}
    assert "id" in data


def test_create_app_no_metadata(client):
    name = str(generate_nanoid())
    response = client.post("/v1/apps", json={"name": name})
    print(response)
    assert response.status_code == 200
    data = response.json()
    print("===================")
    print(data)
    print("===================")
    assert data["name"] == name
    assert data["metadata"] == {}
    assert "id" in data


def test_get_or_create_app(client):
    name = str(generate_nanoid())
    # Should return a ResourceNotFoundException with 404 status
    response = client.get(f"/v1/apps/name/{name}")
    assert response.status_code == 404
    assert "detail" in response.json()

    # This should create the app
    response = client.get(f"/v1/apps/get_or_create/{name}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == name
    assert "id" in data


def test_get_or_create_existing_app(client):
    name = str(generate_nanoid())

    # App doesn't exist yet
    response = client.get(f"/v1/apps/name/{name}")
    assert response.status_code == 404

    # Create the app
    response = client.post(
        "/v1/apps", json={"name": name, "metadata": {"key": "value"}}
    )
    assert response.status_code == 200
    app1 = response.json()

    # Now get_or_create should find the existing app
    response = client.get(f"/v1/apps/get_or_create/{name}")
    assert response.status_code == 200
    app2 = response.json()

    # Both should be the same app
    assert app1["name"] == app2["name"]
    assert app1["id"] == app2["id"]
    assert app1["metadata"] == app2["metadata"]


def test_get_app_by_id(client, sample_data):
    test_app, _ = sample_data
    response = client.get(f"/v1/apps/{test_app.public_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_app.name
    assert data["id"] == str(test_app.public_id)


def test_get_all_apps(client, sample_data):
    response = client.get("/v1/apps/all")
    assert response.status_code == 200

    # Create an app
    test_app, _ = sample_data
    response = client.get("/v1/apps/all")
    print(response)
    assert response.status_code == 200
    data = response.json()
    assert test_app.name in [app["name"] for app in data]
    assert test_app.public_id in [app["id"] for app in data]


def test_get_app_by_name(client, sample_data):
    test_app, _ = sample_data
    response = client.get(f"/v1/apps/name/{test_app.name}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_app.name
    assert data["id"] == str(test_app.public_id)


def test_update_app(client, sample_data):
    test_app, _ = sample_data
    new_name = str(generate_nanoid())
    response = client.put(
        f"/v1/apps/{test_app.public_id}",
        json={"name": new_name, "metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == new_name
    assert data["metadata"] == {"new_key": "new_value"}


def test_create_duplicate_app_name(client):
    # Create an app
    name = str(generate_nanoid())
    response = client.post("/v1/apps", json={"name": name})
    assert response.status_code == 200

    # Try to create another app with the same name
    response = client.post("/v1/apps", json={"name": name})

    # Should get a ConflictException with 409 status
    assert response.status_code == 409
    data = response.json()
    assert "detail" in data
