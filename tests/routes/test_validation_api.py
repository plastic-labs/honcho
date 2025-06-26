from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src.models import Peer, Workspace


def test_workspace_validations_api(client: TestClient):
    # Test name too short
    response = client.post("/v2/workspaces", json={"name": "", "metadata": {}})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test name too long
    response = client.post("/v2/workspaces", json={"name": "a" * 101, "metadata": {}})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at most 100 characters"
    assert error["type"] == "string_too_long"

    # Test invalid metadata type
    response = client.post(
        "/v2/workspaces", json={"name": "test", "metadata": "not a dict"}
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "metadata"]
    assert error["type"] == "dict_type"


def test_peer_validations_api(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, _ = sample_data

    # Test name too short
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers", json={"name": "", "metadata": {}}
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test name too long
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": "a" * 101, "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at most 100 characters"
    assert error["type"] == "string_too_long"


def test_message_validations_api(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    # Create a test session first
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 200

    long_content = "a" * 50001
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"content": long_content, "peer_id": test_peer.name, "metadata": {}}
            ]
        },
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "String should have at most 50000 characters" in error["msg"]
    assert error["type"] == "string_too_long"


def test_session_validations_api(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    # Create a test session first
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"test_key": "test_value"},
            "configuration": {"test_flag": "test_value"},
        },
    )
    assert session_response.status_code == 200

    # Test invalid metadata type
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": "not a dict"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "metadata"]
    assert error["type"] == "dict_type"

    # Test empty update
    # This should work but not change the session's metadata or configuration
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={},
    )
    assert response.status_code == 200

    # Test that the session's metadata and configuration are not changed
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test_key": "test_value"}
    assert data["configuration"] == {"test_flag": "test_value"}


def test_agent_query_validations_api(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    # Create a session first since agent queries are session-based
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 200

    # Test valid string query (under 10000 chars)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"queries": "a" * 9999, "stream": False},
    )
    assert response.status_code == 200

    # Test string query too long (over 10000 chars)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"queries": "a" * 10001, "stream": False},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "queries"]
    assert error["msg"] == "Value error, Query too long"
    assert error["type"] == "value_error"

    # Test valid list query (under 25 items, each under 10000 chars)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"queries": ["a" * 9999 for _ in range(25)], "stream": False},
    )
    assert response.status_code == 200

    # Test list too long (over 25 items)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"queries": ["test" for _ in range(26)], "stream": False},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "queries"]
    assert error["type"] == "value_error"

    # Test list item too long (item over 10000 chars)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"queries": ["a" * 10001], "stream": False},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "queries"]
    assert error["msg"] == "Value error, One or more queries too long"
    assert error["type"] == "value_error"

    # Test that strings over 20 chars are allowed
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"queries": "a" * 100, "stream": False},  # 100 chars should be fine
    )
    assert response.status_code == 200


def test_required_field_validations_api(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 200

    # Test missing required content in message
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={"messages": [{"peer_id": test_peer.name, "metadata": {}}]},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "content" in str(error["loc"])
    assert error["type"] == "missing"

    # Test missing required peer_id in message
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={"messages": [{"content": "test", "metadata": {}}]},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "peer_id" in str(error["loc"]) or "peer_name" in str(error["loc"])
    assert error["type"] == "missing"


def test_filter_validations_api(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    # Create a session first
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 200

    # Test invalid filter type in message list (at session level)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": "not a dict"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "filter"]
    assert error["type"] == "dict_type"
