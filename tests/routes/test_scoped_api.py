from nanoid import generate as generate_nanoid

from src.security import JWTParams, create_jwt


def test_create_app_with_auth(auth_client):
    name = str(generate_nanoid())

    response = auth_client.post(
        "/v1/apps", json={"name": name, "metadata": {"key": "value"}}
    )

    # Check expected behavior based on auth type
    if auth_client.auth_type != "admin":
        assert response.status_code == 401
        return

    assert response.status_code == 200


def test_auth_response_time(auth_client):
    name = str(generate_nanoid())

    import time

    start_time = time.time()

    response = auth_client.post(
        "/v1/apps", json={"name": name, "metadata": {"key": "value"}}
    )

    end_time = time.time()
    response_time = end_time - start_time
    print(
        f"Server response time for client {auth_client.auth_type}: {response_time:.6f} seconds"
    )

    # Check expected behavior based on auth type
    if auth_client.auth_type != "admin":
        assert response.status_code == 401
        return

    assert response.status_code == 200


def test_get_or_create_app_with_auth(auth_client):
    name = str(generate_nanoid())
    # Should return a ResourceNotFoundException with 404 status
    response = auth_client.get(f"/v1/apps/name/{name}")

    if auth_client.auth_type != "admin":
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

    response = auth_client.get(f"/v1/apps?app_id={test_app.public_id}")

    # Admin JWT or JWT with matching app_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_get_app_from_token(auth_client, sample_data):
    test_app, _ = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

    response = auth_client.get("/v1/apps")

    if auth_client.auth_type == "empty":
        assert response.status_code == 200
        assert response.json()["id"] == test_app.public_id
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


def test_update_app_with_wrong_auth(auth_client, sample_data):
    test_app, _ = sample_data

    different_app = str(generate_nanoid())

    if auth_client.auth_type == "empty":
        # For non-admin, include the *wrong* app_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=different_app))}"
        )

    new_name = str(generate_nanoid())
    response = auth_client.put(
        f"/v1/apps/{test_app.public_id}",
        json={"name": new_name, "metadata": {"new_key": "new_value"}},
    )

    # Only admin JWT or JWT with matching app_id should be allowed
    if auth_client.auth_type == "admin":
        assert response.status_code == 200
    else:
        # wrong app_id should be rejected
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


def test_get_user_by_id_with_auth(auth_client, sample_data):
    test_app, test_user = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

    response = auth_client.get(
        f"/v1/apps/{test_app.public_id}/users?user_id={test_user.public_id}"
    )

    # Admin JWT or JWT with matching app_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401

    # Test with user-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(us=test_user.public_id))}"
        )

        response = auth_client.get(
            f"/v1/apps/{test_app.public_id}/users?user_id={test_user.public_id}"
        )

        assert response.status_code == 200

        response2 = auth_client.get(f"/v1/apps/{test_app.public_id}/users")

        assert response2.status_code == 200

        print(response2.json())

        assert response2.json()["id"] == test_user.public_id


def test_get_user_by_name_with_auth(auth_client, sample_data):
    test_app, test_user = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

    response = auth_client.get(
        f"/v1/apps/{test_app.public_id}/users/name/{test_user.name}"
    )

    # Admin JWT or JWT with matching app_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_update_user_with_auth(auth_client, sample_data):
    test_app, test_user = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the app_id in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

    new_name = str(generate_nanoid())
    response = auth_client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}",
        json={"name": new_name, "metadata": {"updated_key": "updated_value"}},
    )

    # Admin JWT or JWT with matching app_id should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401

    # Test with user-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(us=test_user.public_id))}"
        )

        response = auth_client.put(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}",
            json={
                "name": str(generate_nanoid()),
                "metadata": {"user_key": "user_value"},
            },
        )

        assert response.status_code == 200


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


def test_get_session_by_id_with_auth(auth_client, sample_data):
    test_app, test_user = sample_data

    # First create a session
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id, us=test_user.public_id))}"
        )

    create_response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={},
    )

    if auth_client.auth_type not in ["admin", "empty"]:
        assert create_response.status_code == 401
        return

    assert create_response.status_code == 200
    session_id = create_response.json()["id"]

    # Test with app and user scoped JWT
    response = auth_client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions?session_id={session_id}"
    )
    assert response.status_code == 200

    # Test with session-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(se=session_id))}"
        )

        response = auth_client.get(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions?session_id={session_id}"
        )
        assert response.status_code == 200

        response2 = auth_client.get(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions"
        )

        assert response2.status_code == 200

        assert response2.json()["id"] == session_id


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
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
            json={"name": "test_collection2", "metadata": {}},
        )

        assert response.status_code == 200

    # Remove user_id from header and make sure app-scoped key works too
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id))}"
        )

        response = auth_client.post(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
            json={"name": "test_collection3", "metadata": {}},
        )

        assert response.status_code == 200


def test_get_collection_by_id_with_auth(auth_client, sample_data) -> None:
    test_app, test_user = sample_data

    # First create a collection
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id, us=test_user.public_id))}"
        )

    create_response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection_get", "metadata": {}},
    )

    if auth_client.auth_type not in ["admin", "empty"]:
        assert create_response.status_code == 401
        return

    assert create_response.status_code == 200
    collection_id = create_response.json()["id"]

    # Test with app and user scoped JWT
    response = auth_client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections?collection_id={collection_id}"
    )
    assert response.status_code == 200

    # Test with collection-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(co=collection_id))}"
        )

        response = auth_client.get(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections?collection_id={collection_id}"
        )
        assert response.status_code == 200

        # Test auto resolution of ID
        response2 = auth_client.get(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections"
        )
        assert response2.status_code == 200
        assert response2.json()["id"] == collection_id


def test_get_collection_by_name_with_auth(auth_client, sample_data) -> None:
    test_app, test_user = sample_data
    collection_name = f"test_collection_{generate_nanoid()}"

    # First create a collection
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id, us=test_user.public_id))}"
        )

    create_response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": collection_name, "metadata": {}},
    )

    if auth_client.auth_type not in ["admin", "empty"]:
        assert create_response.status_code == 401
        return

    assert create_response.status_code == 200

    # Test with app and user scoped JWT
    response = auth_client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/name/{collection_name}"
    )
    assert response.status_code == 200


def test_create_document_with_auth(auth_client, sample_data) -> None:
    test_app, test_user = sample_data

    # First create a collection
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id, us=test_user.public_id))}"
        )

    create_collection_response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection_docs", "metadata": {}},
    )

    if auth_client.auth_type not in ["admin", "empty"]:
        assert create_collection_response.status_code == 401
        return

    assert create_collection_response.status_code == 200
    collection_id = create_collection_response.json()["id"]

    # Create document with app and user scoped JWT
    response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents",
        json={"content": "Test document content", "metadata": {"doc_key": "doc_value"}},
    )
    assert response.status_code == 200

    # Test with collection-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(co=collection_id))}"
        )

        response = auth_client.post(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents",
            json={"content": "Test document with collection JWT", "metadata": {}},
        )
        assert response.status_code == 200


def test_get_document_with_auth(auth_client, sample_data) -> None:
    test_app, test_user = sample_data

    # First create a collection and document
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(ap=test_app.public_id, us=test_user.public_id))}"
        )

    create_collection_response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections",
        json={"name": "test_collection_get_doc", "metadata": {}},
    )

    if auth_client.auth_type not in ["admin", "empty"]:
        assert create_collection_response.status_code == 401
        return

    assert create_collection_response.status_code == 200
    collection_id = create_collection_response.json()["id"]

    create_doc_response = auth_client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents",
        json={"content": "Test document for retrieval", "metadata": {}},
    )
    assert create_doc_response.status_code == 200
    document_id = create_doc_response.json()["id"]

    # Get document with app and user scoped JWT
    response = auth_client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents/{document_id}"
    )
    assert response.status_code == 200

    # Test with collection-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(co=collection_id))}"
        )

        response = auth_client.get(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/collections/{collection_id}/documents/{document_id}"
        )
        assert response.status_code == 200
