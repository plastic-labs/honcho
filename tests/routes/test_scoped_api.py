from nanoid import generate as generate_nanoid

from src.security import JWTParams, create_jwt


def test_create_app_with_auth(auth_client):
    name = str(generate_nanoid())
    response = auth_client.post(
        "/v1/apps", json={"name": name, "metadata": {"key": "value"}}
    )

    # Check expected behavior based on auth type
    if auth_client.auth_type != "admin":
        print(f"Auth type: {auth_client.auth_type}")
        assert response.status_code == 401
        return

    assert response.status_code == 200


def test_get_or_create_app_with_auth(auth_client):
    name = str(generate_nanoid())
    # Should return a ResourceNotFoundException with 404 status
    response = auth_client.get(f"/v1/apps/name/{name}")

    if auth_client.auth_type != "admin":
        print(f"Auth type: {auth_client.auth_type}")
        assert response.status_code == 401
        return

    assert response.status_code == 404

    response = auth_client.get(f"/v1/apps/get_or_create/{name}")

    if auth_client.auth_type != "admin":
        assert response.status_code == 401
        return

    assert response.status_code == 200


def test_get_app_by_id_with_auth(auth_client, sample_data):
    test_app, _ = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

    response = auth_client.get(f"/v1/apps/{test_app.public_id}")

    # Admin JWT or JWT with matching app_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_get_app_by_name_with_auth(auth_client, sample_data):
    test_app, _ = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id in the JWT
        # Note that this will still fail because name route requires admin
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

    response = auth_client.get(f"/v1/apps/name/{test_app.name}")

    # Only admin JWT should be allowed
    if auth_client.auth_type == "admin":
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_update_app_with_auth(auth_client, sample_data):
    test_app, _ = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id in the JWT
        # Note that this will still fail because name route requires admin
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

    new_name = str(generate_nanoid())
    response = auth_client.put(
        f"/v1/apps/{test_app.public_id}",
        json={"name": new_name, "metadata": {"new_key": "new_value"}},
    )

    # Only admin JWT or JWT with matching app_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_create_user_with_auth(auth_client, sample_data):
    test_app, _ = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

    name = str(generate_nanoid())
    response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users",
        json={"name": name, "metadata": {"user_key": "user_value"}},
    )

    # Only admin JWT or JWT with matching app_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401


## XX more user-level tests


def test_create_session_with_auth(auth_client, sample_data):
    test_app, test_user = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id and user_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id, us=test_user.public_id))}"
        )

    response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={},
    )

    # Only admin JWT or JWT with matching app_id and user_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401

    # Remove app_id from header and make sure user-scoped key works too
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(us=test_user.public_id))}"
        )

        response = auth_client.post(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
            json={},
        )

        assert response.status_code == 200


## XX more session-level tests


def test_create_collection(auth_client, sample_data) -> None:
    test_app, test_user = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id and user_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id, us=test_user.public_id))}"
        )

    response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection", "metadata": {}},
    )

    # Only admin JWT or JWT with matching app_id and user_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401

    # Remove app_id from header and make sure user-scoped key works too
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(us=test_user.public_id))}"
        )

        response = auth_client.post(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
            json={},
        )

        assert response.status_code == 200


## XX more collection-level tests
