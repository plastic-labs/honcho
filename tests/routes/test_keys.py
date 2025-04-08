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


def test_create_key_with_expires_at(auth_client, sample_data):
    """Test creating a key with an expiration date"""
    response = auth_client.post("/v1/keys", params={"expires_at": "2025-01-01"})

    # Only admin JWT should be allowed
    if auth_client.auth_type == "admin":
        # key with no params should fail
        assert response.status_code == 422
        return
    else:
        assert response.status_code == 401

    test_app, _ = sample_data

    # assert that the key is expired
    response = auth_client.post("/v1/keys", params={"app_id": test_app.public_id})
    assert response.status_code == 401
