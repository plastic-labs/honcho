from nanoid import generate as generate_nanoid

from src.security import JWTParams, create_jwt


def test_create_workspace_with_auth(auth_client):
    name = str(generate_nanoid())

    response = auth_client.post(
        "/v2/workspaces", json={"name": name, "metadata": {"key": "value"}}
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
        "/v2/workspaces", json={"name": name, "metadata": {"key": "value"}}
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


def test_get_or_create_workspace_with_auth(auth_client):
    name = str(generate_nanoid())

    response = auth_client.post(
        "/v2/workspaces", json={"name": name, "metadata": {"key": "value"}}
    )

    if auth_client.auth_type != "admin":
        assert response.status_code == 401
        return

    assert response.status_code == 200


def test_get_workspace_with_auth(auth_client, sample_data):
    test_workspace, _ = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the workspace name in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
        )

    response = auth_client.post("/v2/workspaces", json={"name": test_workspace.name})

    # Admin JWT or JWT with matching workspace should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_update_workspace_with_auth(auth_client, sample_data):
    test_workspace, _ = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the workspace name in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
        )

    new_name = str(generate_nanoid())
    response = auth_client.put(
        f"/v2/workspaces/{test_workspace.name}",
        json={"name": new_name, "metadata": {"new_key": "new_value"}},
    )

    # Only admin JWT or JWT with matching workspace should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_update_workspace_with_wrong_auth(auth_client, sample_data):
    test_workspace, _ = sample_data

    different_workspace = str(generate_nanoid())

    if auth_client.auth_type == "empty":
        # For non-admin, include the *wrong* workspace name in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=different_workspace))}"
        )

    new_name = str(generate_nanoid())
    response = auth_client.put(
        f"/v2/workspaces/{test_workspace.name}",
        json={"name": new_name, "metadata": {"new_key": "new_value"}},
    )

    # Only admin JWT or JWT with matching workspace should be allowed
    if auth_client.auth_type == "admin":
        assert response.status_code == 200
    else:
        # wrong workspace should be rejected
        assert response.status_code == 401


def test_create_peer_with_auth(auth_client, sample_data):
    test_workspace, _ = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the workspace name in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
        )

    name = str(generate_nanoid())
    response = auth_client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": name, "metadata": {"peer_key": "peer_value"}},
    )

    # Only admin JWT or JWT with matching workspace should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401


def test_get_peer_by_name_with_auth(auth_client, sample_data):
    test_workspace, test_peer = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the workspace name in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
        )

    # Use POST /list endpoint to get peers
    response = auth_client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"name": test_peer.name}},
    )

    # Admin JWT or JWT with matching workspace should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401

    # Test with peer-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(p=test_peer.name))}"
        )

        # Get specific peer using get_or_create endpoint
        response = auth_client.post(
            f"/v2/workspaces/{test_workspace.name}/peers", json={"name": test_peer.name}
        )

        assert response.status_code == 200


def test_update_peer_with_auth(auth_client, sample_data):
    test_workspace, test_peer = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the workspace name in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
        )

    new_name = str(generate_nanoid())
    response = auth_client.put(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
        json={"name": new_name, "metadata": {"updated_key": "updated_value"}},
    )

    # Admin JWT or JWT with matching workspace should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401

    # Test with peer-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(p=test_peer.name, w=test_workspace.name))}"
        )

        response = auth_client.put(
            f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}",
            json={
                "name": str(generate_nanoid()),
                "metadata": {"peer_key": "peer_value"},
            },
        )

        assert response.status_code == 200


def test_create_session_with_auth(auth_client, sample_data):
    test_workspace, test_peer = sample_data

    if auth_client.auth_type == "empty":
        # For non-admin, include the workspace name in the JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
        )

    session_name = str(generate_nanoid())
    response = auth_client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"name": session_name, "peer_names": {test_peer.name: {}}},
    )

    # Only admin JWT or JWT with matching workspace should be allowed
    if auth_client.auth_type in ["admin", "empty"]:
        assert response.status_code == 200
    else:
        assert response.status_code == 401

    # Test with peer-scoped JWT
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(p=test_peer.name))}"
        )

        session_name2 = str(generate_nanoid())
        response = auth_client.post(
            f"/v2/workspaces/{test_workspace.name}/sessions",
            json={"name": session_name2, "peer_names": {test_peer.name: {}}},
        )

        assert response.status_code == 200


def test_get_session_by_name_with_auth(auth_client, sample_data):
    test_workspace, test_peer = sample_data

    # First create a session
    if auth_client.auth_type == "empty":
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
        )

    session_name = str(generate_nanoid())
    create_response = auth_client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"name": session_name, "peer_names": {test_peer.name: {}}},
    )

    if auth_client.auth_type not in ["admin", "empty"]:
        assert create_response.status_code == 401
        return

    assert create_response.status_code == 200

    # Test with workspace scoped JWT - get the same session
    response = auth_client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions", json={"name": session_name}
    )
    assert response.status_code == 200

    if auth_client.auth_type == "empty":
        # Test with session-scoped JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(s=session_name))}"
        )

        response = auth_client.post(
            f"/v2/workspaces/{test_workspace.name}/sessions",
            json={"name": session_name},
        )
        assert response.status_code == 200

        # Test with wrong session_name (should be 401 since we have a session-scoped JWT)
        assert (
            auth_client.post(
                f"/v2/workspaces/{test_workspace.name}/sessions",
                json={"name": generate_nanoid()},
            ).status_code
            == 401
        )

        # Test with peer-scoped JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(p=test_peer.name))}"
        )

        assert (
            auth_client.post(
                f"/v2/workspaces/{test_workspace.name}/sessions",
                json={"name": session_name},
            ).status_code
            == 200
        )

        # Test with workspace-scoped JWT
        auth_client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=test_workspace.name))}"
        )

        assert (
            auth_client.post(
                f"/v2/workspaces/{test_workspace.name}/sessions",
                json={"name": session_name},
            ).status_code
            == 200
        )

        # Test with wrong session_name using DELETE endpoint (should be 404 since session doesn't exist)
        wrong_session_name = generate_nanoid()
        assert (
            auth_client.delete(
                f"/v2/workspaces/{test_workspace.name}/sessions/{wrong_session_name}"
            ).status_code
            == 404
        )
