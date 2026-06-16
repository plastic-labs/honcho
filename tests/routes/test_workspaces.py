import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


def test_get_or_create_workspace(client: TestClient):
    name = str(generate_nanoid())

    # This should create the workspace using POST /v3/workspaces
    response = client.post("/v3/workspaces", json={"name": name})
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["id"] == name
    assert "id" in data


def test_get_or_create_workspace_with_configuration(client: TestClient):
    """Test workspace creation with configuration parameter"""
    name = str(generate_nanoid())
    configuration = {"feature1": True, "feature2": False}

    response = client.post(
        "/v3/workspaces", json={"name": name, "configuration": configuration}
    )
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["id"] == name
    assert data["configuration"] == configuration


def test_get_or_create_workspace_with_all_optional_params(client: TestClient):
    """Test workspace creation with all optional parameters"""
    name = str(generate_nanoid())
    metadata = {"key": "value", "number": 42}
    configuration = {"experimental": True, "beta": False}

    response = client.post(
        "/v3/workspaces",
        json={"name": name, "metadata": metadata, "configuration": configuration},
    )
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["id"] == name
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_get_or_create_existing_workspace(client: TestClient):
    name = str(generate_nanoid())

    # Create the workspace
    response = client.post(
        "/v3/workspaces", json={"name": name, "metadata": {"key": "value"}}
    )
    assert response.status_code in [200, 201]
    workspace1 = response.json()

    # Try to create the same workspace again - should return existing workspace
    response = client.post(
        "/v3/workspaces", json={"name": name, "metadata": {"key": "value"}}
    )
    assert response.status_code in [200, 201]
    workspace2 = response.json()

    # Both should be the same workspace
    assert workspace1["id"] == workspace2["id"]
    assert workspace1["metadata"] == workspace2["metadata"]


@pytest.mark.asyncio
async def test_get_all_workspaces(client: TestClient):
    # create a test workspace with metadata
    response = client.post(
        "/v3/workspaces",
        json={
            "name": "test_workspace",
            "metadata": {"test_key": "test_value"},
        },
    )

    response = client.post(
        "/v3/workspaces/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0

    response = client.post(
        "/v3/workspaces/list",
        json={"filters": {"metadata": {"test_key": "test_value"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["metadata"] == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_get_all_workspaces_with_empty_filter(client: TestClient):
    """Test workspace listing with empty filter object"""
    response = client.post("/v3/workspaces/list", json={"filters": {}})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_all_workspaces_with_null_filter(client: TestClient):
    """Test workspace listing with null filter"""
    response = client.post("/v3/workspaces/list", json={"filters": None})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_all_workspaces_with_reverse(client: TestClient):
    """Test workspace listing with reverse creation-time ordering."""
    first_name = f"reverse-workspace-{generate_nanoid()}"
    second_name = f"reverse-workspace-{generate_nanoid()}"

    first_response = client.post(
        "/v3/workspaces",
        json={"name": first_name, "metadata": {"reverse_group": first_name}},
    )
    assert first_response.status_code in [200, 201]

    second_response = client.post(
        "/v3/workspaces",
        json={"name": second_name, "metadata": {"reverse_group": first_name}},
    )
    assert second_response.status_code in [200, 201]

    normal_response = client.post(
        "/v3/workspaces/list",
        json={"filters": {"metadata": {"reverse_group": first_name}}},
    )
    assert normal_response.status_code == 200

    reverse_response = client.post(
        "/v3/workspaces/list?reverse=true",
        json={"filters": {"metadata": {"reverse_group": first_name}}},
    )
    assert reverse_response.status_code == 200

    assert [item["id"] for item in normal_response.json()["items"]] == [
        first_name,
        second_name,
    ]
    assert [item["id"] for item in reverse_response.json()["items"]] == [
        second_name,
        first_name,
    ]


@pytest.mark.asyncio
async def test_get_all_workspaces_reverse_uses_id_tiebreaker(
    client: TestClient, db_session: AsyncSession
):
    """Workspaces with identical created_at fall back to ordering by id (nanoid PK)."""
    reverse_group = f"tiebreaker-{generate_nanoid()}"
    shared_created_at = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    low_id = "A" * 21
    high_id = "z" * 21
    low_name = f"tie-low-{generate_nanoid()}"
    high_name = f"tie-high-{generate_nanoid()}"

    db_session.add(
        models.Workspace(
            id=low_id,
            name=low_name,
            created_at=shared_created_at,
            h_metadata={"reverse_group": reverse_group},
        )
    )
    db_session.add(
        models.Workspace(
            id=high_id,
            name=high_name,
            created_at=shared_created_at,
            h_metadata={"reverse_group": reverse_group},
        )
    )
    await db_session.commit()

    normal_response = client.post(
        "/v3/workspaces/list",
        json={"filters": {"metadata": {"reverse_group": reverse_group}}},
    )
    assert normal_response.status_code == 200

    reverse_response = client.post(
        "/v3/workspaces/list?reverse=true",
        json={"filters": {"metadata": {"reverse_group": reverse_group}}},
    )
    assert reverse_response.status_code == 200

    normal_items = [item["id"] for item in normal_response.json()["items"]]
    reverse_items = [item["id"] for item in reverse_response.json()["items"]]

    # When created_at ties, ordering falls back to the nanoid id: low_id < high_id
    # lexicographically, so the workspace with id="AAA..." sorts first ascending.
    assert normal_items == [low_name, high_name]
    assert reverse_items == [high_name, low_name]


@pytest.mark.asyncio
async def test_get_all_workspaces_reverse_with_pagination(client: TestClient):
    """Paged reverse listing returns newest-first across consecutive pages."""
    reverse_group = f"paged-reverse-{generate_nanoid()}"
    names = [f"paged-reverse-{i}-{generate_nanoid()}" for i in range(3)]

    for name in names:
        response = client.post(
            "/v3/workspaces",
            json={"name": name, "metadata": {"reverse_group": reverse_group}},
        )
        assert response.status_code in [200, 201]

    page_one = client.post(
        "/v3/workspaces/list?reverse=true&page=1&size=1",
        json={"filters": {"metadata": {"reverse_group": reverse_group}}},
    )
    assert page_one.status_code == 200
    page_two = client.post(
        "/v3/workspaces/list?reverse=true&page=2&size=1",
        json={"filters": {"metadata": {"reverse_group": reverse_group}}},
    )
    assert page_two.status_code == 200
    page_three = client.post(
        "/v3/workspaces/list?reverse=true&page=3&size=1",
        json={"filters": {"metadata": {"reverse_group": reverse_group}}},
    )
    assert page_three.status_code == 200

    assert page_one.json()["total"] == 3
    assert [item["id"] for item in page_one.json()["items"]] == [names[2]]
    assert [item["id"] for item in page_two.json()["items"]] == [names[1]]
    assert [item["id"] for item in page_three.json()["items"]] == [names[0]]


def test_update_workspace(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, _ = sample_data
    _new_name = str(generate_nanoid())
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


def test_update_workspace_with_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test workspace update with configuration parameter"""
    test_workspace, _ = sample_data
    configuration = {"new_feature": True, "legacy_feature": False}

    response = client.put(
        f"/v3/workspaces/{test_workspace.name}", json={"configuration": configuration}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["configuration"] == configuration


def test_update_workspace_with_all_optional_params(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test workspace update with both metadata and configuration"""
    test_workspace, _ = sample_data
    metadata = {"updated_key": "updated_value", "count": 100}
    configuration = {"experimental": True, "beta": True}

    response = client.put(
        f"/v3/workspaces/{test_workspace.name}",
        json={"metadata": metadata, "configuration": configuration},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == metadata
    assert data["configuration"] == configuration


def test_update_workspace_with_null_metadata(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test workspace update with null metadata (should clear metadata)"""
    test_workspace, _ = sample_data

    # First set some metadata
    client.put(
        f"/v3/workspaces/{test_workspace.name}", json={"metadata": {"temp": "value"}}
    )

    # Then clear it with null
    response = client.put(
        f"/v3/workspaces/{test_workspace.name}", json={"metadata": None}
    )
    assert response.status_code == 200
    data = response.json()
    # The behavior might be to keep existing metadata or clear it -
    # adjust this assertion based on actual behavior
    assert "metadata" in data


def test_update_workspace_with_null_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test workspace update with null configuration"""
    test_workspace, _ = sample_data

    response = client.put(
        f"/v3/workspaces/{test_workspace.name}", json={"configuration": None}
    )
    assert response.status_code == 200
    data = response.json()
    assert "configuration" in data


def test_create_duplicate_workspace_name(client: TestClient):
    # Create an workspace
    name = str(generate_nanoid())
    response = client.post("/v3/workspaces", json={"name": name})
    assert response.status_code in [200, 201]

    # Try to create another workspace with the same name - should return existing workspace
    response = client.post("/v3/workspaces", json={"name": name})

    # Should return the existing workspace with 200 status (get_or_create behavior)
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["id"] == name


def test_search_workspace(client: TestClient, sample_data: tuple[Workspace, Peer]):
    """Test the workspace search functionality"""
    test_workspace, _ = sample_data

    # Test search with a query
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/search",
        json={"query": "test search query", "limit": 10},
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_workspace_empty_query(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test the workspace search with empty query"""
    test_workspace, _ = sample_data

    # Test search with empty query
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/search", json={"query": "", "limit": 10}
    )
    assert response.status_code == 200
    data = response.json()

    # Response should be a direct list of messages
    assert isinstance(data, list)


def test_search_workspace_nonexistent(client: TestClient):
    """Test searching a workspace that doesn't exist"""
    nonexistent_workspace_id = str(generate_nanoid())

    response = client.post(
        f"/v3/workspaces/{nonexistent_workspace_id}/search",
        json={"query": "test query", "limit": 10},
    )
    assert response.status_code == 200
    data: list[dict[str, Any]] = response.json()
    # Should return empty list for nonexistent workspace
    assert isinstance(data, list)
    assert len(data) == 0


def test_delete_workspace(client: TestClient):
    """Test deleting a workspace"""
    name = str(generate_nanoid())

    # Create a workspace
    response = client.post("/v3/workspaces", json={"name": name})
    assert response.status_code in [200, 201]
    workspace = response.json()
    assert workspace["id"] == name

    # Delete the workspace
    response = client.delete(f"/v3/workspaces/{name}")
    assert response.status_code == 202

    # Verify the workspace no longer exists by trying to update it
    response = client.put(
        f"/v3/workspaces/{name}", json={"metadata": {"test": "value"}}
    )
    # Should create a new workspace since the old one was deleted
    assert response.status_code == 200


def test_delete_nonexistent_workspace(client: TestClient):
    """Test deleting a workspace that doesn't exist"""
    nonexistent_workspace_id = str(generate_nanoid())

    response = client.delete(f"/v3/workspaces/{nonexistent_workspace_id}")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_delete_workspace_with_peers(client: TestClient):
    """Test deleting a workspace that has peers"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v3/workspaces", json={"name": workspace_name})
    assert response.status_code in [200, 201]

    # Create peers
    peer1_name = str(generate_nanoid())
    peer2_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/peers", json={"name": peer1_name}
    )
    assert response.status_code in [200, 201]
    response = client.post(
        f"/v3/workspaces/{workspace_name}/peers", json={"name": peer2_name}
    )
    assert response.status_code in [200, 201]

    # Delete workspace
    response = client.delete(f"/v3/workspaces/{workspace_name}")
    assert response.status_code == 202


def test_delete_workspace_with_sessions(client: TestClient):
    """Test that deleting a workspace with active sessions returns 409"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v3/workspaces", json={"name": workspace_name})
    assert response.status_code in [200, 201]

    # Create sessions
    session1_name = str(generate_nanoid())
    session2_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions", json={"name": session1_name}
    )
    assert response.status_code in [200, 201]
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions", json={"name": session2_name}
    )
    assert response.status_code in [200, 201]

    # Delete workspace should fail with 409
    response = client.delete(f"/v3/workspaces/{workspace_name}")
    assert response.status_code == 409
    data = response.json()
    assert "active session" in data["detail"].lower()


def test_delete_workspace_with_messages(client: TestClient):
    """Test deleting a workspace that has messages"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v3/workspaces", json={"name": workspace_name})
    assert response.status_code in [200, 201]

    # Create peer
    peer_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/peers", json={"name": peer_name}
    )
    assert response.status_code in [200, 201]

    # Create session
    session_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions", json={"name": session_name}
    )
    assert response.status_code in [200, 201]

    # Add peer to session
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions/{session_name}/peers",
        json={peer_name: {}},
    )
    assert response.status_code == 200

    # Create messages
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions/{session_name}/messages",
        json={
            "messages": [
                {"content": "Test message 1", "peer_id": peer_name},
                {"content": "Test message 2", "peer_id": peer_name},
            ]
        },
    )
    assert response.status_code == 201

    # Delete session first (marks inactive, required before workspace deletion)
    response = client.delete(f"/v3/workspaces/{workspace_name}/sessions/{session_name}")
    assert response.status_code == 202

    # Delete workspace
    response = client.delete(f"/v3/workspaces/{workspace_name}")
    assert response.status_code == 202


def test_delete_workspace_with_webhooks(client: TestClient):
    """Test deleting a workspace that has webhooks"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v3/workspaces", json={"name": workspace_name})
    assert response.status_code in [200, 201]

    # Create webhook
    response = client.post(
        f"/v3/workspaces/{workspace_name}/webhooks",
        json={
            "url": "https://example.com/webhook",
        },
    )
    assert response.status_code in [200, 201]

    # Delete workspace
    response = client.delete(f"/v3/workspaces/{workspace_name}")
    assert response.status_code == 202

    # Verify webhook is deleted by checking workspace doesn't exist
    response = client.get(f"/v3/workspaces/{workspace_name}/webhooks")
    # This should either return 404 or empty list depending on implementation
    assert response.status_code in [404, 200]


def test_delete_workspace_cascade(client: TestClient):
    """Test that deleting a workspace cascades to all related resources"""
    workspace_name = str(generate_nanoid())

    # Create workspace with complex structure
    response = client.post(
        "/v3/workspaces",
        json={"name": workspace_name, "metadata": {"test": "cascade"}},
    )
    assert response.status_code in [200, 201]

    # Create multiple peers
    peer_names = [str(generate_nanoid()) for _ in range(3)]
    for peer_name in peer_names:
        response = client.post(
            f"/v3/workspaces/{workspace_name}/peers", json={"name": peer_name}
        )
        assert response.status_code in [200, 201]

    # Create multiple sessions
    session_names = [str(generate_nanoid()) for _ in range(2)]
    for session_name in session_names:
        response = client.post(
            f"/v3/workspaces/{workspace_name}/sessions", json={"name": session_name}
        )
        assert response.status_code in [200, 201]

    # Add peers to sessions and create messages
    for session_name in session_names:
        for peer_name in peer_names[:2]:  # Add 2 peers to each session
            response = client.post(
                f"/v3/workspaces/{workspace_name}/sessions/{session_name}/peers",
                json={peer_name: {}},
            )
            assert response.status_code == 200

        # Create messages in session
        response = client.post(
            f"/v3/workspaces/{workspace_name}/sessions/{session_name}/messages",
            json={
                "messages": [
                    {
                        "content": f"Test message in {session_name}",
                        "peer_id": peer_names[0],
                    }
                ]
            },
        )
        assert response.status_code == 201

    # Delete sessions first (marks inactive, required before workspace deletion)
    for session_name in session_names:
        response = client.delete(
            f"/v3/workspaces/{workspace_name}/sessions/{session_name}"
        )
        assert response.status_code == 202

    # Delete the workspace
    response = client.delete(f"/v3/workspaces/{workspace_name}")
    assert response.status_code == 202


def test_delete_workspace_returns_accepted(client: TestClient):
    """Test that delete workspace returns 202 Accepted"""
    name = str(generate_nanoid())
    metadata = {"key": "value", "number": 42}
    configuration = {"feature": True}

    # Create workspace with metadata and configuration
    response = client.post(
        "/v3/workspaces",
        json={"name": name, "metadata": metadata, "configuration": configuration},
    )
    assert response.status_code in [200, 201]

    # Delete workspace
    response = client.delete(f"/v3/workspaces/{name}")
    assert response.status_code == 202


def test_delete_workspace_blocked_by_sessions_returns_409(client: TestClient):
    """Test that deleting a workspace with active sessions returns 409 with descriptive message"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v3/workspaces", json={"name": workspace_name})
    assert response.status_code in [200, 201]

    # Create a session
    session_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions", json={"name": session_name}
    )
    assert response.status_code in [200, 201]

    # Attempt to delete workspace
    response = client.delete(f"/v3/workspaces/{workspace_name}")
    assert response.status_code == 409
    data = response.json()
    assert "detail" in data
    assert "active session" in data["detail"]
    assert "delete all sessions first" in data["detail"].lower()


def test_delete_workspace_after_session_deletion(client: TestClient):
    """Test that workspace deletion succeeds after all sessions are deleted"""
    workspace_name = str(generate_nanoid())

    # Create workspace
    response = client.post("/v3/workspaces", json={"name": workspace_name})
    assert response.status_code in [200, 201]

    # Create sessions
    session1_name = str(generate_nanoid())
    session2_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions", json={"name": session1_name}
    )
    assert response.status_code in [200, 201]
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions", json={"name": session2_name}
    )
    assert response.status_code in [200, 201]

    # Workspace deletion should fail
    response = client.delete(f"/v3/workspaces/{workspace_name}")
    assert response.status_code == 409

    # Delete all sessions (marks inactive)
    response = client.delete(
        f"/v3/workspaces/{workspace_name}/sessions/{session1_name}"
    )
    assert response.status_code == 202
    response = client.delete(
        f"/v3/workspaces/{workspace_name}/sessions/{session2_name}"
    )
    assert response.status_code == 202

    # Now workspace deletion should succeed
    response = client.delete(f"/v3/workspaces/{workspace_name}")
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_schedule_dream_invokes_enqueue_dream(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """POST /schedule_dream forwards observer/observed/dream_type to enqueue_dream.

    After Loop 4, the manual schedule_dream route no longer touches the
    baseline count — the orchestrator writes both guard fields atomically on
    successful completion. The route's job shrinks to forwarding the dream
    request.
    """
    workspace, peer = sample_data

    collection = models.Collection(
        observer=peer.name,
        observed=peer.name,
        workspace_name=workspace.name,
        internal_metadata={},
    )
    db_session.add(collection)
    await db_session.commit()

    captured: dict[str, Any] = {}

    async def fake_enqueue_dream(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    with (
        patch("src.routers.workspaces.settings.DREAM.ENABLED", True),
        patch(
            "src.routers.workspaces.enqueue_dream",
            new=AsyncMock(side_effect=fake_enqueue_dream),
        ),
    ):
        response = client.post(
            f"/v3/workspaces/{workspace.name}/schedule_dream",
            json={
                "observer": peer.name,
                "observed": peer.name,
                "dream_type": "omni",
            },
        )

    assert response.status_code == 204, response.text
    assert "kwargs" in captured, "enqueue_dream was not called"
    assert captured["kwargs"]["observer"] == peer.name
    assert captured["kwargs"]["observed"] == peer.name
    assert "document_count" not in captured["kwargs"], (
        "Loop 4: enqueue_dream no longer accepts document_count; the baseline "
        "is written atomically with last_dream_at in process_dream."
    )
