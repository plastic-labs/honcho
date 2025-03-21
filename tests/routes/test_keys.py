from src.security import JWTParams, create_admin_jwt, create_jwt


def test_create_key_with_auth(auth_client):
    """Test creating a key with different auth types"""
    response = auth_client.post("/v1/keys")

    # Only admin JWT should be allowed
    if auth_client.auth_type == "admin":
        assert response.status_code == 200
        assert "key" in response.json()
        assert "created_at" in response.json()
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

    if auth_client.auth_type != "admin":
        # Try to revoke with non-admin auth
        response = auth_client.post("/v1/keys/revoke?key=" + key)
        assert response.status_code == 401
    else:
        response = auth_client.post("/v1/keys/revoke?key=" + key)
        assert response.status_code == 200
        assert response.json() == {"revoked": key}


def test_revoke_nonexistent_key(auth_client):
    """Test revoking a key that doesn't exist"""
    if auth_client.auth_type != "admin":
        return  # Skip test if not admin authentication

    key = create_jwt(JWTParams(us="hello"))

    response = auth_client.post("/v1/keys/revoke?key=" + key)
    # Should return 404 for a key that doesn't exist
    assert response.status_code == 404
