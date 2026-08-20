import datetime

import pytest

from src.exceptions import AuthenticationException
from src.models import Peer, Workspace
from src.security import verify_jwt
from tests.conftest import AuthClient


def test_create_key_no_params(auth_client: AuthClient):
    """Test creating a key with no parameters"""
    response = auth_client.post("/v3/keys")

    # Only admin JWT should be allowed
    if auth_client.auth_type == "admin":
        # key with no params should fail
        assert response.status_code == 422
    else:
        assert response.status_code == 401


def test_create_key_with_params(
    auth_client: AuthClient, sample_data: tuple[Workspace, Peer]
):
    """Test creating a key with specific parameters"""
    test_workspace, test_peer = sample_data

    if auth_client.auth_type != "admin":
        return  # Skip test if not admin authentication

    # Test with app_id
    response = auth_client.post(
        "/v3/keys", params={"workspace_id": test_workspace.name}
    )
    assert response.status_code == 200
    assert "key" in response.json()

    # Test with app_id and user_id
    response = auth_client.post(
        "/v3/keys",
        params={"workspace_id": test_workspace.name, "peer_id": test_peer.name},
    )
    assert response.status_code == 200
    assert "key" in response.json()

    # Test with session_id and collection_id
    response = auth_client.post(
        "/v3/keys",
        params={
            "workspace_id": test_workspace.name,
            "peer_id": test_peer.name,
            "session_id": "test-session",
            "collection_id": "test-collection",
        },
    )
    assert response.status_code == 200
    assert "key" in response.json()


def test_create_key_with_expires_at(
    auth_client: AuthClient, sample_data: tuple[Workspace, Peer]
):
    """Test creating a key with an expiration date"""
    response = auth_client.post("/v3/keys", params={"expires_at": "2025-01-01"})

    # Only admin JWT should be allowed
    if auth_client.auth_type != "admin":
        assert response.status_code == 401
        return

    # key with no params should fail
    assert response.status_code == 422

    test_workspace, _ = sample_data

    # Future expiry: mint succeeds and the token verifies (NumericDate exp, #1016)
    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    response = auth_client.post(
        "/v3/keys",
        params={"workspace_id": test_workspace.name, "expires_at": future.isoformat()},
    )
    assert response.status_code == 200
    verify_jwt(response.json()["key"])  # must not raise "Invalid JWT"

    # Past expiry: mint succeeds but verification reports expired
    past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    response = auth_client.post(
        "/v3/keys",
        params={"workspace_id": test_workspace.name, "expires_at": past.isoformat()},
    )
    assert response.status_code == 200
    with pytest.raises(AuthenticationException, match="JWT expired"):
        verify_jwt(response.json()["key"])
