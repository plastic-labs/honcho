from src.security import JWTParams, create_admin_jwt, create_jwt


def test_create_key_no_params(auth_client):
    """Test creating a key with no parameters"""
    response = auth_client.post("/v1/keys")

    # Only admin JWT should be allowed
    if auth_client.auth_type == "admin":
        # key with no params should fail
        assert response.status_code == 422
    else:
        assert response.status_code == 401


def test_create_key_with_params(auth_client, sample_data):
    """Test creating a key with specific parameters"""
    test_app, test_user = sample_data

    if auth_client.auth_type != "admin":
        return  # Skip test if not admin authentication

    # Test with app_id
    response = auth_client.post("/v1/keys", params={"app_id": test_app.public_id})
    assert response.status_code == 200
    assert "key" in response.json()

    # Test with app_id and user_id
    response = auth_client.post(
        "/v1/keys",
        params={"app_id": test_app.public_id, "user_id": test_user.public_id},
    )
    assert response.status_code == 200
    assert "key" in response.json()

    # Test with session_id and collection_id
    response = auth_client.post(
        "/v1/keys",
        params={
            "app_id": test_app.public_id,
            "user_id": test_user.public_id,
            "session_id": "test-session",
            "collection_id": "test-collection",
        },
    )
    assert response.status_code == 200
    assert "key" in response.json()


def test_revoke_key_with_auth(auth_client, sample_data):
    """Test revoking a key with different auth types"""

    try:
        original_auth = auth_client.headers["Authorization"]
    except KeyError:
        original_auth = None

    # Create a test key to revoke using admin auth
    test_app, test_user = sample_data
    auth_client.headers["Authorization"] = f"Bearer {create_admin_jwt()}"
    create_response = auth_client.post(
        "/v1/keys",
        params={
            "app_id": test_app.public_id,
            "user_id": test_user.public_id,
            "session_id": "test-session",
            "collection_id": "test-collection",
        },
    )
    assert create_response.status_code == 200
    key = create_response.json()["key"]
    auth_client.headers.pop("Authorization")

    if original_auth:
        auth_client.headers["Authorization"] = original_auth

    response = auth_client.post("/v1/keys/revoke?key=" + key)

    if auth_client.auth_type != "admin":
        # Try to revoke with non-admin auth
        assert response.status_code == 401
    else:
        assert response.status_code == 200
        assert response.json() == {"revoked": key}


def test_revoke_nonexistent_key(auth_client):
    """Test revoking a key that doesn't exist"""

    key = create_jwt(JWTParams(us="newkey"))

    response = auth_client.post("/v1/keys/revoke?key=" + key)

    if auth_client.auth_type != "admin":
        assert response.status_code == 401
    else:
        # Should still return 200 for a key that doesn't exist:
        # we want to be able to revoke any possible key
        assert response.status_code == 200
        assert response.json() == {"revoked": key}


def test_rotate_jwt_secret(auth_client):
    """Test rotating the JWT secret"""
    if auth_client.auth_type != "admin":
        # Only admin can rotate JWT secret
        response = auth_client.post("/v1/keys/rotate")
        assert response.status_code == 401
        return

    # Test rotation with no new secret (auto-generated)
    response = auth_client.post("/v1/keys/rotate")
    assert response.status_code == 200
    assert "key" in response.json()
    assert "created_at" in response.json()

    # Verify the new key works for authentication
    new_key = response.json()["key"]
    original_auth = auth_client.headers["Authorization"]
    auth_client.headers["Authorization"] = f"Bearer {new_key}"

    # Try to use the new key to access a protected endpoint
    keys_response = auth_client.get("/v1/keys/")
    assert keys_response.status_code == 200

    # Restore original auth
    auth_client.headers["Authorization"] = original_auth


def test_rotate_jwt_with_custom_secret(auth_client):
    """Test rotating the JWT secret with a custom secret"""
    if auth_client.auth_type != "admin":
        return  # Skip test if not admin authentication

    # Test rotation with a specific new secret
    new_secret = "test_custom_secret_for_rotation"
    response = auth_client.post(f"/v1/keys/rotate?new_secret={new_secret}")
    assert response.status_code == 200
    assert "key" in response.json()

    # Verify the new key works
    new_key = response.json()["key"]
    original_auth = auth_client.headers["Authorization"]
    auth_client.headers["Authorization"] = f"Bearer {new_key}"

    keys_response = auth_client.get("/v1/keys/")
    assert keys_response.status_code == 200

    # Verify that old key doesn't work
    auth_client.headers["Authorization"] = original_auth
    keys_response = auth_client.get("/v1/keys/")
    assert keys_response.status_code == 401

    # Restore original auth
    auth_client.headers["Authorization"] = original_auth


def test_rotate_jwt_disabled_auth(client):
    """Test rotating JWT secret when auth is disabled"""
    # Temporarily modify USE_AUTH to simulate disabled auth
    import src.routers.keys as keys_module

    original_use_auth = keys_module.USE_AUTH
    keys_module.USE_AUTH = False

    try:
        response = client.post("/v1/keys/rotate")
        assert response.status_code == 405
        assert "detail" in response.json()
        assert "disabled" in response.json()["detail"].lower()
    finally:
        # Restore original USE_AUTH value
        keys_module.USE_AUTH = original_use_auth
