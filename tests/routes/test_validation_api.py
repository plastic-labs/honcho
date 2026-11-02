import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models import Peer, Workspace
from src.schemas.configuration import MessageConfiguration


def test_workspace_validations_api(client: TestClient):
    # Test name too short
    response = client.post("/v3/workspaces", json={"name": "", "metadata": {}})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test name too long
    response = client.post("/v3/workspaces", json={"name": "a" * 513, "metadata": {}})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at most 512 characters"
    assert error["type"] == "string_too_long"

    # Test invalid metadata type
    response = client.post(
        "/v3/workspaces", json={"name": "test", "metadata": "not a dict"}
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "metadata"]
    assert error["type"] == "dict_type"


def test_lone_surrogate_input_returns_422(client: TestClient):
    # Valid JSON, invalid Unicode: the echoed `input` must not break rendering
    response = client.post(
        "/v3/workspaces",
        content=b'{"name": "emoji \\ud83d cut", "metadata": {}}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["type"] == "string_unicode"


def test_peer_validations_api(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, _ = sample_data

    # Test name too short
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers", json={"name": "", "metadata": {}}
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at least 1 character"
    assert error["type"] == "string_too_short"

    # Test name too long
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers",
        json={"name": "a" * 513, "metadata": {}},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "name"]
    assert error["msg"] == "String should have at most 512 characters"
    assert error["type"] == "string_too_long"


def test_message_validations_api(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data
    # Create a test session first
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 201

    # Test content too long
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"test_key": "test_value"},
            "configuration": {"test_flag": "test_value"},
        },
    )
    assert session_response.status_code == 201

    # Test invalid metadata type
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={"metadata": "not a dict"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "metadata"]
    assert error["type"] == "dict_type"

    # Test empty update
    # This should work but not change the session's metadata or configuration
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}",
        json={},
    )
    assert response.status_code == 200

    # Test that the session's metadata and configuration are not changed
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
        },
    )
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["metadata"] == {"test_key": "test_value"}
    assert data["configuration"] == {"test_flag": "test_value"}


@patch("src.routers.peers.tracked_db")
def test_agent_query_validations_api(
    mock_tracked_db: AsyncMock,
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
):
    mock_tracked_db.return_value.__aenter__.return_value = db_session

    test_workspace, test_peer = sample_data
    # Create a session first since agent query are session-based
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 201

    # Test valid string query (under 10000 chars)
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
        params={"session_id": session_id, "target": "test_target"},
        json={"query": "a" * 9999, "stream": False},
    )
    assert response.status_code == 200

    # Test string query too long (over 10000 chars)
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
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
        f"/v3/workspaces/{test_workspace.name}/peers/{test_peer.name}/chat",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 201

    # Test missing required content in message
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={"messages": [{"peer_id": test_peer.name, "metadata": {}}]},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "content" in str(error["loc"])
    assert error["type"] == "missing"

    # Test missing required peer_id in message
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
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
        f"/v3/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 201

    # Test invalid filter type in message list (at session level)
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filters": "not a dict"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "filters"]
    assert error["type"] == "dict_type"


@pytest.mark.parametrize(
    ("method", "path", "body", "path_param"),
    [
        ("PUT", "/v3/workspaces/bad:id", {}, "workspace_id"),
        ("POST", "/v3/workspaces/bad:id/peers", {"id": "p"}, "workspace_id"),
        ("POST", "/v3/workspaces/bad:id/sessions", {"id": "s"}, "workspace_id"),
        ("POST", "/v3/workspaces/bad:id/scopes", {"id": "sc"}, "workspace_id"),
        ("PUT", "/v3/workspaces/{ws}/sessions/bad:id", {}, "session_id"),
        (
            "POST",
            "/v3/workspaces/{ws}/sessions/bad:id/messages",
            {"messages": [{"content": "hi", "peer_id": "{peer}"}]},
            "session_id",
        ),
    ],
)
def test_get_or_create_path_id_failing_schema_returns_422(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    method: str,
    path: str,
    body: dict[str, Any],
    path_param: str,
):
    # The handler builds a create schema from the path ID, so the pattern
    # violation is raised inside it, not by request parsing
    test_workspace, test_peer = sample_data
    body = json.loads(json.dumps(body).replace("{peer}", test_peer.name))
    response = client.request(method, path.format(ws=test_workspace.name), json=body)
    assert response.status_code == 422, response.text
    error = response.json()["detail"][0]
    assert error["type"] == "string_pattern_mismatch"
    assert error["loc"] == ["path", path_param]
    assert "input" not in error


def test_nul_in_configuration_is_stripped(client: TestClient):
    # Configuration allows extra keys, stored as JSONB, which rejects NUL
    response = client.post(
        "/v3/workspaces",
        json={"name": "nul-config", "configuration": {"k\u0000": "a\u0000b"}},
    )
    assert response.status_code in (200, 201)
    assert response.json()["configuration"]["k"] == "ab"


@pytest.mark.parametrize("depth", [970, 1200])
@pytest.mark.parametrize("nested_in_reasoning", [False, True])
def test_deep_unknown_message_configuration_is_ignored(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    depth: int,
    nested_in_reasoning: bool,
):
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    base = f"/v3/workspaces/{workspace.name}/sessions/{session_id}"
    assert (
        client.post(
            f"/v3/workspaces/{workspace.name}/sessions",
            json={"id": session_id, "peers": {peer.name: {}}},
        ).status_code
        == 201
    )

    # Build raw JSON: json.dumps can itself hit Python's recursion limit.
    unknown = '"x":' + "[" * depth + "0" + "]" * depth
    reasoning = '"enabled":true,"custom_instructions":"a\\u0000b"'
    configuration = (
        '{"reasoning":{' + reasoning + "," + unknown + "}}"
        if nested_in_reasoning
        else "{" + unknown + ',"reasoning":{' + reasoning + "}}"
    )
    body = (
        '{"messages":[{"content":"hi","peer_id":'
        + json.dumps(peer.name)
        + ',"configuration":'
        + configuration
        + "}]}"
    )
    response = client.post(
        f"{base}/messages",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 201, response.text
    assert MessageConfiguration.model_validate(json.loads(configuration)).model_dump(
        exclude_none=True
    ) == {"reasoning": {"enabled": True, "custom_instructions": "ab"}}
    listed = client.post(f"{base}/messages/list", json={})
    assert listed.status_code == 200, listed.text
    assert listed.json()["items"][0]["id"] == response.json()[0]["id"]


def test_nul_in_id_filter_returns_422(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.get(
        f"/v3/workspaces/{test_workspace.name}/queue/status",
        params={"observer_id": "\u0000"},
    )
    assert response.status_code == 422


def test_page_beyond_bigint_offset_returns_422(client: TestClient):
    response = client.post("/v3/workspaces/list", params={"page": 2**62}, json={})
    assert response.status_code == 422


def test_lone_surrogate_in_free_form_json_returns_422(client: TestClient):
    # Pydantic accepts the surrogate inside an Any-typed dict; Postgres JSONB
    # then rejects it, which must surface as a 422 rather than a 500.
    response = client.post(
        "/v3/workspaces",
        content=b'{"id": "surrogate-ws", "metadata": {"k": "\\ud800"}}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
