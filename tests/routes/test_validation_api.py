from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src.config import settings
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

    # Test content too long
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "a" * (settings.MAX_MESSAGE_SIZE + 1),
                    "peer_id": test_peer.name,
                    "metadata": {},
                }
            ]
        },
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "content" in str(error["loc"])
    assert (
        error["msg"]
        == f"String should have at most {settings.MAX_MESSAGE_SIZE} characters"
    )
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
    # Create a session first since agent query are session-based
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
        json={"query": "a" * 9999, "stream": False},
    )
    assert response.status_code == 200

    # Test string query too long (over 10000 chars)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"query": "a" * 10001, "stream": False},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "query"]
    assert error["msg"] == "String should have at most 10000 characters"
    assert error["type"] == "string_too_long"

    # Test that strings over 20 chars are allowed
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"query": "a" * 100, "stream": False},  # 100 chars should be fine
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
        json={"filters": "not a dict"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "filters"]
    assert error["type"] == "dict_type"
