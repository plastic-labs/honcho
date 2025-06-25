"""
Tests for advanced filter functionality including logical operators,
comparison operators, and wildcards across multiple models.
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Peer, Workspace


@pytest.mark.asyncio
async def test_logical_operators_and_filters(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test AND, OR, NOT logical operators in filters"""
    test_workspace, test_peer = sample_data

    # Create multiple peers with different metadata
    peer1_name = str(generate_nanoid())
    peer2_name = str(generate_nanoid())
    peer3_name = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer1_name,
            "metadata": {
                "role": "admin",
                "department": "engineering",
                "level": "senior",
            },
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer2_name,
            "metadata": {
                "role": "user",
                "department": "engineering",
                "level": "junior",
            },
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer3_name,
            "metadata": {"role": "admin", "department": "sales", "level": "senior"},
        },
    )

    # Test AND operator - peers who are admin AND in engineering
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": [
                    {"metadata": {"role": "admin"}},
                    {"metadata": {"department": "engineering"}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer1_name

    # Test OR operator - peers who are admin OR in engineering
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "OR": [
                    {"metadata": {"role": "admin"}},
                    {"metadata": {"department": "engineering"}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3  # All three peers match

    # Test NOT operator - peers who are NOT admin
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"NOT": [{"metadata": {"role": "admin"}}]}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer2_name in found_names  # Only peer2 is not admin
    assert peer1_name not in found_names
    assert peer3_name not in found_names


@pytest.mark.asyncio
async def test_comparison_operators_filters(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test comparison operators (gte, lte, gt, lt, ne, in, contains, icontains)"""
    test_workspace, test_peer = sample_data

    # Create session with messages containing different metadata
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 200

    # Create messages with numeric metadata for comparison tests
    messages_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Message with score 10",
                    "peer_id": test_peer.name,
                    "metadata": {
                        "score": 10,
                        "category": "high",
                        "tags": ["important", "urgent"],
                    },
                },
                {
                    "content": "Message with score 5",
                    "peer_id": test_peer.name,
                    "metadata": {"score": 5, "category": "medium", "tags": ["normal"]},
                },
                {
                    "content": "Message with score 1",
                    "peer_id": test_peer.name,
                    "metadata": {"score": 1, "category": "low", "tags": ["minor"]},
                },
            ]
        },
    )
    assert messages_response.status_code == 200

    # Test gte (greater than or equal)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"gte": 5}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2  # Messages with score 10 and 5

    # Test lte (less than or equal)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"lte": 5}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2  # Messages with score 5 and 1

    # Test gt (greater than)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"gt": 5}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1  # Only message with score 10

    # Test lt (less than)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"lt": 5}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1  # Only message with score 1

    # Test ne (not equal)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"ne": 5}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2  # Messages with score 10 and 1

    # Test in operator
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"category": {"in": ["high", "low"]}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2  # Messages with category "high" and "low"

    # Test contains operator for text content
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"content": {"contains": "score 10"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert "score 10" in data["items"][0]["content"]

    # Test icontains operator (case-insensitive)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"content": {"icontains": "MESSAGE"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3  # All messages contain "message" (case-insensitive)


@pytest.mark.asyncio
async def test_wildcard_filters(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test wildcard (*) filters that match everything for a field"""
    test_workspace, test_peer = sample_data

    # Create peers with different names
    peer1_name = str(generate_nanoid())
    peer2_name = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer1_name, "metadata": {"type": "bot"}},
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {"type": "human"}},
    )

    # Test wildcard for peer_id field
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": [
                    {"id": "*"},
                    {"metadata": {"type": "bot"}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Should find the bot peer (wildcard doesn't filter anything)
    found_names = [item["id"] for item in data["items"]]
    assert peer1_name in found_names

    # Test wildcard in comparison operators
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "id": {"in": ["*"]}  # Wildcard in comparison should also match all
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Since wildcard should be ignored, this should return all peers
    assert len(data["items"]) >= 3  # At least the 3 peers we know about


@pytest.mark.asyncio
async def test_complex_nested_filters(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test complex nested logical operations"""
    test_workspace, test_peer = sample_data

    # Create session and messages for complex filtering
    session_id = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Create messages with various metadata combinations
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Priority urgent task",
                    "peer_id": test_peer.name,
                    "metadata": {
                        "priority": "urgent",
                        "status": "open",
                        "assignee": "alice",
                    },
                },
                {
                    "content": "Normal task for bob",
                    "peer_id": test_peer.name,
                    "metadata": {
                        "priority": "normal",
                        "status": "open",
                        "assignee": "bob",
                    },
                },
                {
                    "content": "Completed urgent task",
                    "peer_id": test_peer.name,
                    "metadata": {
                        "priority": "urgent",
                        "status": "completed",
                        "assignee": "alice",
                    },
                },
                {
                    "content": "Low priority task",
                    "peer_id": test_peer.name,
                    "metadata": {
                        "priority": "low",
                        "status": "open",
                        "assignee": "charlie",
                    },
                },
            ]
        },
    )

    # Complex filter: (urgent OR normal priority) AND open status AND NOT assigned to charlie
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={
            "filter": {
                "AND": [
                    {
                        "OR": [
                            {"metadata": {"priority": "urgent"}},
                            {"metadata": {"priority": "normal"}},
                        ]
                    },
                    {"metadata": {"status": "open"}},
                    {"NOT": [{"metadata": {"assignee": "charlie"}}]},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2  # Should match first two messages

    # Verify the correct messages were returned
    contents = [item["content"] for item in data["items"]]
    assert "Priority urgent task" in contents
    assert "Normal task for bob" in contents
    assert "Completed urgent task" not in contents  # Wrong status
    assert "Low priority task" not in contents  # Wrong assignee and priority


@pytest.mark.asyncio
async def test_filters_across_different_models(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test that filters work consistently across different models"""
    test_workspace, test_peer = sample_data

    # Test workspace filters
    workspace_name = str(generate_nanoid())
    client.post(
        "/v2/workspaces",
        json={
            "name": workspace_name,
            "metadata": {"environment": "production", "version": "2.0"},
        },
    )

    response = client.post(
        "/v2/workspaces/list",
        json={
            "filter": {
                "AND": [
                    {"metadata": {"environment": "production"}},
                    {"metadata": {"version": {"gte": "2.0"}}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert workspace_name in found_names

    # Test session filters with comparison operators
    session1_id = str(generate_nanoid())
    session2_id = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session1_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"duration": 30, "type": "meeting"},
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session2_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"duration": 60, "type": "workshop"},
        },
    )

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={
            "filter": {
                "OR": [
                    {"metadata": {"duration": {"gte": 45}}},
                    {"metadata": {"type": "meeting"}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    found_sessions = [item["id"] for item in data["items"]]
    assert session1_id in found_sessions  # Matches type=meeting
    assert session2_id in found_sessions  # Matches duration>=45


@pytest.mark.asyncio
async def test_filter_edge_cases(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test edge cases and error handling for filters"""
    test_workspace, test_peer = sample_data

    # Test empty logical operators
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": []  # Empty AND should not crash
            }
        },
    )
    assert response.status_code == 200

    # Test nested empty operators
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"OR": [{"AND": []}, {"id": test_peer.name}]}},
    )
    assert response.status_code == 200

    # Test filter with non-existent columns (should be ignored gracefully)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"non_existent_column": "value"}},
    )
    assert response.status_code == 422

    # Test mixed wildcards and regular values
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": [
                    {"id": "*"},  # Wildcard
                    {"id": test_peer.name},  # Regular value
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Should find the specific peer since AND combines conditions
    found_names = [item["id"] for item in data["items"]]
    assert test_peer.name in found_names


@pytest.mark.asyncio
async def test_backward_compatibility(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test that old simple filter format still works"""
    test_workspace, test_peer = sample_data

    # Create peer with metadata
    peer_name = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer_name,
            "metadata": {"role": "admin", "department": "engineering"},
        },
    )

    # Test old-style simple equality filter
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"metadata": {"role": "admin"}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer_name in found_names

    # Test multiple field simple filter (implicit AND)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"metadata": {"role": "admin"}, "id": peer_name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer_name


@pytest.mark.asyncio
async def test_range_queries_with_dates(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test range queries that might be used with date fields"""
    test_workspace, test_peer = sample_data

    # Create sessions with date-like metadata
    session1_id = str(generate_nanoid())
    session2_id = str(generate_nanoid())
    session3_id = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session1_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"created_date": "2024-01-15", "priority": 5},
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session2_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"created_date": "2024-02-20", "priority": 3},
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session3_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"created_date": "2024-03-10", "priority": 8},
        },
    )

    # Test date range query
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={
            "filter": {
                "metadata": {"created_date": {"gte": "2024-02-01", "lte": "2024-02-28"}}
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    found_sessions = [item["id"] for item in data["items"]]
    assert session2_id in found_sessions
    assert session1_id not in found_sessions
    assert session3_id not in found_sessions

    # Test combining date and numeric filters
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={
            "filter": {
                "AND": [
                    {"metadata": {"created_date": {"gte": "2024-01-01"}}},
                    {"metadata": {"priority": {"gt": 4}}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    found_sessions = [item["id"] for item in data["items"]]
    assert session1_id in found_sessions  # priority 5
    assert session3_id in found_sessions  # priority 8
    assert session2_id not in found_sessions  # priority 3


@pytest.mark.asyncio
async def test_all_workspace_columns_filtering(client: TestClient):
    """Test filtering by all available workspace columns"""
    # Create additional workspaces for testing
    workspace1_name = str(generate_nanoid())
    workspace2_name = str(generate_nanoid())

    client.post(
        "/v2/workspaces",
        json={
            "name": workspace1_name,
            "metadata": {"env": "dev", "version": "1.0", "active": True},
            "configuration": {"max_sessions": 100, "timeout": 30},
        },
    )
    client.post(
        "/v2/workspaces",
        json={
            "name": workspace2_name,
            "metadata": {"env": "prod", "version": "2.0", "active": False},
            "configuration": {"max_sessions": 500, "timeout": 60},
        },
    )

    # Test filtering by id (maps to name internally)
    response = client.post(
        "/v2/workspaces/list",
        json={"filter": {"id": workspace1_name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == workspace1_name

    # Test filtering by id with comparison operators
    response = client.post(
        "/v2/workspaces/list",
        json={"filter": {"id": {"in": [workspace1_name, workspace2_name]}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert workspace1_name in found_names
    assert workspace2_name in found_names

    # Test filtering by metadata (maps to h_metadata internally)
    response = client.post(
        "/v2/workspaces/list",
        json={"filter": {"metadata": {"env": "dev"}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert workspace1_name in found_names
    assert workspace2_name not in found_names

    # Test filtering by metadata with comparison operators
    response = client.post(
        "/v2/workspaces/list",
        json={"filter": {"metadata": {"version": {"gte": "2.0"}}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert workspace2_name in found_names
    assert workspace1_name not in found_names

    # Test filtering by configuration
    response = client.post(
        "/v2/workspaces/list",
        json={"filter": {"configuration": {"max_sessions": {"gte": 500}}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert workspace2_name in found_names
    assert workspace1_name not in found_names

    # Test filtering by created_at (datetime field)
    response = client.post(
        "/v2/workspaces/list",
        json={"filter": {"created_at": {"gte": "2020-01-01"}}},
    )
    assert response.status_code == 200
    # Should return all workspaces created after 2020


@pytest.mark.asyncio
async def test_all_peer_columns_filtering(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test filtering by all available peer columns"""
    test_workspace, test_peer = sample_data

    # Create additional peers for testing
    peer1_name = str(generate_nanoid())
    peer2_name = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer1_name,
            "metadata": {"role": "user", "level": 1, "active": True},
            "configuration": {"notifications": True, "theme": "dark"},
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer2_name,
            "metadata": {"role": "admin", "level": 5, "active": False},
            "configuration": {"notifications": False, "theme": "light"},
        },
    )

    # Test filtering by id (maps to name internally)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"id": peer1_name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer1_name

    # Test filtering by workspace_id (maps to workspace_name internally)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"workspace_id": test_workspace.name}},
    )
    assert response.status_code == 200
    data = response.json()
    # Should return all peers in the workspace
    assert len(data["items"]) >= 3  # test_peer + peer1 + peer2

    # Test filtering by metadata
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"metadata": {"role": "admin"}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer2_name in found_names
    assert peer1_name not in found_names

    # Test filtering by metadata with comparison operators
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"metadata": {"level": {"gte": 3}}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer2_name in found_names
    assert peer1_name not in found_names

    # Test filtering by configuration
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"configuration": {"notifications": True}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer1_name in found_names
    assert peer2_name not in found_names

    # Test filtering by created_at
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"created_at": {"gte": "2020-01-01"}}},
    )
    assert response.status_code == 200
    # Should return all peers created after 2020


@pytest.mark.asyncio
async def test_all_session_columns_filtering(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test filtering by all available session columns"""
    test_workspace, test_peer = sample_data

    # Create additional sessions for testing
    session1_id = str(generate_nanoid())
    session2_id = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session1_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"type": "chat", "priority": 1, "active": True},
            "configuration": {"auto_save": True, "timeout": 30},
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session2_id,
            "peer_names": {test_peer.name: {}},
            "metadata": {"type": "support", "priority": 5, "active": False},
            "configuration": {"auto_save": False, "timeout": 60},
        },
    )

    # Test filtering by id (maps to name internally)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"id": session1_id}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == session1_id

    # Test filtering by workspace_id (maps to workspace_name internally)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"workspace_id": test_workspace.name}},
    )
    assert response.status_code == 200
    data = response.json()
    # Should return all sessions in the workspace
    assert len(data["items"]) >= 2

    # Test filtering by is_active (boolean field)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"is_active": True}},
    )
    assert response.status_code == 200
    data = response.json()
    found_sessions = [item["id"] for item in data["items"]]
    assert session1_id in found_sessions
    # session2 should not be in results since is_active=False by default in get_sessions

    # Test filtering by metadata
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"metadata": {"type": "chat"}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_sessions = [item["id"] for item in data["items"]]
    assert session1_id in found_sessions

    # Test filtering by metadata with comparison operators
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"metadata": {"priority": {"gte": 3}}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_sessions = [item["id"] for item in data["items"]]
    assert session2_id in found_sessions
    assert session1_id not in found_sessions

    # Test filtering by configuration
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"configuration": {"auto_save": True}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_sessions = [item["id"] for item in data["items"]]
    assert session1_id in found_sessions

    # Test filtering by created_at
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"created_at": {"gte": "2020-01-01"}}},
    )
    assert response.status_code == 200
    # Should return all sessions created after 2020


@pytest.mark.asyncio
async def test_all_message_columns_filtering(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test filtering by all available message columns"""
    test_workspace, test_peer = sample_data

    # Create session and messages for testing
    session_id = str(generate_nanoid())
    session_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )
    assert session_response.status_code == 200

    # Create messages with various data
    messages_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Hello world message",
                    "peer_id": test_peer.name,
                    "metadata": {"type": "greeting", "priority": 1, "urgent": True},
                },
                {
                    "content": "Technical support request",
                    "peer_id": test_peer.name,
                    "metadata": {"type": "support", "priority": 5, "urgent": False},
                },
                {
                    "content": "Follow up message",
                    "peer_id": test_peer.name,
                    "metadata": {"type": "followup", "priority": 3, "urgent": True},
                },
            ]
        },
    )
    assert messages_response.status_code == 200
    created_messages = messages_response.json()

    # Test filtering by id (for messages, this should map to public_id)
    message_id = created_messages[0]["id"]
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"id": message_id}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == message_id

    # Test filtering by session_id (maps to session_name internally)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"session_id": session_id}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3  # All messages in the session

    # Test filtering by peer_id (maps to peer_name internally)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"peer_id": test_peer.name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3  # All messages from the peer

    # Test filtering by workspace_id (maps to workspace_name internally)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"workspace_id": test_workspace.name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3  # All messages in the workspace

    # Test filtering by content (text field)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"content": "Hello world message"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert "Hello world" in data["items"][0]["content"]

    # Test filtering by content with contains operator
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"content": {"contains": "support"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert "support" in data["items"][0]["content"]

    # Test filtering by metadata
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"type": "greeting"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1

    # Test filtering by metadata with comparison operators
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"priority": {"gte": 3}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2  # priority 5 and 3

    # Test filtering by token_count (integer field) - this should exist after message creation
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"token_count": {"gte": 0}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3  # All messages should have token_count >= 0

    # Test filtering by created_at
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"created_at": {"gte": "2020-01-01"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3  # All messages created after 2020


@pytest.mark.asyncio
async def test_id_field_interpolation_consistency(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test that id field interpolation works consistently across all models"""
    test_workspace, test_peer = sample_data

    # Create test data
    workspace_name = str(generate_nanoid())
    peer_name = str(generate_nanoid())
    session_id = str(generate_nanoid())

    # Create workspace
    client.post(
        "/v2/workspaces",
        json={"name": workspace_name, "metadata": {"test": "value"}},
    )

    # Create peer
    client.post(
        f"/v2/workspaces/{workspace_name}/peers",
        json={"name": peer_name, "metadata": {"test": "value"}},
    )

    # Create session
    client.post(
        f"/v2/workspaces/{workspace_name}/sessions",
        json={
            "id": session_id,
            "peer_names": {peer_name: {}},
            "metadata": {"test": "value"},
        },
    )

    # Create message
    messages_response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Test message",
                    "peer_id": peer_name,
                    "metadata": {"test": "value"},
                }
            ]
        },
    )
    message_id = messages_response.json()[0]["id"]

    # Test that filtering by "id" returns the expected items for each model

    # Workspace: id should map to name
    response = client.post(
        "/v2/workspaces/list",
        json={"filter": {"id": workspace_name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == workspace_name

    # Peer: id should map to name
    response = client.post(
        f"/v2/workspaces/{workspace_name}/peers/list",
        json={"filter": {"id": peer_name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == peer_name

    # Session: id should map to name
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/list",
        json={"filter": {"id": session_id}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == session_id

    # Message: id should map to public_id (not name since messages don't have a user-visible name)
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/{session_id}/messages/list",
        json={"filter": {"id": message_id}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == message_id


@pytest.mark.asyncio
async def test_foreign_key_field_interpolation(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test that foreign key _id fields map to _name fields correctly"""
    test_workspace, test_peer = sample_data

    # Create test data
    peer_name = str(generate_nanoid())
    session_id = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer_name, "metadata": {"role": "test"}},
    )

    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={
            "id": session_id,
            "peer_names": {peer_name: {}},
            "metadata": {"type": "test"},
        },
    )

    # Create message
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Test message",
                    "peer_id": peer_name,
                    "metadata": {"test": "value"},
                }
            ]
        },
    )

    # Test workspace_id filtering for peers (maps to workspace_name)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"workspace_id": test_workspace.name}},
    )
    assert response.status_code == 200
    data = response.json()
    peer_names = [item["id"] for item in data["items"]]
    assert peer_name in peer_names
    assert test_peer.name in peer_names

    # Test workspace_id filtering for sessions (maps to workspace_name)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/list",
        json={"filter": {"workspace_id": test_workspace.name}},
    )
    assert response.status_code == 200
    data = response.json()
    session_names = [item["id"] for item in data["items"]]
    assert session_id in session_names

    # Test session_id filtering for messages (maps to session_name)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"session_id": session_id}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1

    # Test peer_id filtering for messages (maps to peer_name)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"peer_id": peer_name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1

    # Test workspace_id filtering for messages (maps to workspace_name)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"workspace_id": test_workspace.name}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_metadata_field_interpolation(client: TestClient):
    """Test that metadata field maps to h_metadata column correctly"""
    # Create test data with metadata
    workspace_name = str(generate_nanoid())
    peer_name = str(generate_nanoid())
    session_id = str(generate_nanoid())

    client.post(
        "/v2/workspaces",
        json={
            "name": workspace_name,
            "metadata": {"env": "test", "version": "1.0", "active": True},
        },
    )

    client.post(
        f"/v2/workspaces/{workspace_name}/peers",
        json={
            "name": peer_name,
            "metadata": {"role": "user", "level": 5, "premium": True},
        },
    )

    client.post(
        f"/v2/workspaces/{workspace_name}/sessions",
        json={
            "id": session_id,
            "peer_names": {peer_name: {}},
            "metadata": {"type": "chat", "duration": 120, "archived": False},
        },
    )

    client.post(
        f"/v2/workspaces/{workspace_name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Test message",
                    "peer_id": peer_name,
                    "metadata": {
                        "sentiment": "positive",
                        "confidence": 0.95,
                        "flagged": False,
                    },
                }
            ]
        },
    )

    # Test metadata filtering for all models

    # Workspace metadata
    response = client.post(
        "/v2/workspaces/list",
        json={"filter": {"metadata": {"env": "test"}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert workspace_name in found_names

    # Peer metadata
    response = client.post(
        f"/v2/workspaces/{workspace_name}/peers/list",
        json={"filter": {"metadata": {"role": "user"}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer_name in found_names

    # Session metadata
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/list",
        json={"filter": {"metadata": {"type": "chat"}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert session_id in found_names

    # Message metadata
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"sentiment": "positive"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1

    # Test metadata with comparison operators

    # Numeric metadata comparison
    response = client.post(
        f"/v2/workspaces/{workspace_name}/peers/list",
        json={"filter": {"metadata": {"level": {"gte": 3}}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer_name in found_names

    # Boolean metadata comparison
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/list",
        json={"filter": {"metadata": {"archived": {"ne": True}}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert session_id in found_names

    # Float metadata comparison
    response = client.post(
        f"/v2/workspaces/{workspace_name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"confidence": {"gte": 0.9}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_nonexistent_columns_ignored_gracefully(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test that filtering by non-existent columns is ignored gracefully"""
    test_workspace, test_peer = sample_data

    # Create test data
    peer_name = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer_name, "metadata": {"role": "test"}},
    )

    # Test filtering by non-existent columns
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"nonexistent_column": "value"}},
    )
    assert response.status_code == 422

    # Test combining real and non-existent columns
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": [
                    {"metadata": {"role": "test"}},  # Real filter
                    {"fake_column": "fake_value"},  # Non-existent filter
                ]
            }
        },
    )
    assert response.status_code == 422

    # Test with complex nested filters containing non-existent columns
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "OR": [
                    {"nonexistent_field": "value"},
                    {"metadata": {"role": "test"}},
                ]
            }
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_not_logic_correctness(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test that NOT logic works correctly - THIS WILL LIKELY FAIL due to broken NOT logic"""
    test_workspace, test_peer = sample_data

    # Create peers with specific metadata
    peer1_name = str(generate_nanoid())
    peer2_name = str(generate_nanoid())
    peer3_name = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer1_name,
            "metadata": {"role": "admin", "department": "engineering"},
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer2_name,
            "metadata": {"role": "user", "department": "engineering"},
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer3_name, "metadata": {"role": "admin", "department": "sales"}},
    )

    # Test multiple NOT conditions - this exposes the bug
    # User expectation: NOT admin AND NOT engineering = exclude admin users AND exclude engineering users
    # Current broken code: NOT(admin AND engineering) = exclude users who are BOTH admin AND engineering
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "NOT": [
                    {"metadata": {"role": "admin"}},
                    {"metadata": {"department": "engineering"}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]

    # This assertion will FAIL with current broken logic
    # We expect only peers that are NOT admin AND NOT in engineering
    # With broken logic, we get peers that are not (admin AND engineering) = peer2 and peer3
    # With correct logic, we should get only peers that are (NOT admin) AND (NOT engineering) = none of our test peers
    assert peer1_name not in found_names  # admin + engineering - should be excluded
    assert (
        peer2_name not in found_names
    )  # user + engineering - should be excluded (engineering)
    assert peer3_name not in found_names  # admin + sales - should be excluded (admin)


@pytest.mark.asyncio
async def test_jsonb_type_casting_edge_cases(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test edge cases in JSONB type casting for numeric comparisons"""
    test_workspace, test_peer = sample_data

    session_id = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Create messages with various data types in metadata
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Boolean true",
                    "peer_id": test_peer.name,
                    "metadata": {"active": True, "score": 10},
                },
                {
                    "content": "Boolean false",
                    "peer_id": test_peer.name,
                    "metadata": {"active": False, "score": 5.5},
                },
                {
                    "content": "String number",
                    "peer_id": test_peer.name,
                    "metadata": {"active": "true", "score": "15"},
                },
                {
                    "content": "Large number",
                    "peer_id": test_peer.name,
                    "metadata": {"active": True, "score": 999999999},
                },
            ]
        },
    )

    # Test boolean comparisons - PostgreSQL stores booleans as "true"/"false" strings
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"active": {"ne": False}}}},
    )
    assert response.status_code == 200
    data = response.json()
    # This might fail if boolean casting isn't handled correctly
    assert len(data["items"]) == 3  # All except the false one

    # Test string vs numeric comparison
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"gt": 10}}}},
    )
    assert response.status_code == 200
    data = response.json()
    # Should handle both numeric 15 (from string) and 999999999
    contents = [item["content"] for item in data["items"]]
    assert "String number" in contents or "Large number" in contents

    # Test large number handling
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"gte": 999999999}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert "Large number" in data["items"][0]["content"]


@pytest.mark.asyncio
async def test_real_datetime_column_filtering(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test filtering on actual datetime columns (not just metadata strings)"""
    test_workspace, test_peer = sample_data

    # Create peers at different times (this uses created_at datetime column)
    peer1_name = str(generate_nanoid())
    peer2_name = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer1_name, "metadata": {"created": "early"}},
    )

    # Wait a bit to ensure different timestamps
    import time

    time.sleep(0.1)

    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer2_name, "metadata": {"created": "late"}},
    )

    # Test datetime filtering with various formats
    now = datetime.now(timezone.utc)
    one_minute_ago = now - timedelta(minutes=1)

    # Test ISO format
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"created_at": {"gte": one_minute_ago.isoformat()}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer1_name in found_names
    assert peer2_name in found_names

    # Test date-only format
    today = datetime.now(timezone.utc).date().isoformat()
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"created_at": {"gte": today}}},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_invalid_datetime_handling(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test handling of invalid datetime strings"""
    test_workspace, test_peer = sample_data

    # Test malicious datetime strings (should be rejected by validation)
    malicious_datetimes = [
        "2024-01-01'; DROP TABLE peers; --",
        "2024-01-01 OR 1=1",
        "'; SELECT * FROM users; --",
        "2024-01-01' UNION SELECT password FROM auth",
    ]

    for malicious_dt in malicious_datetimes:
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/peers/list",
            json={"filter": {"created_at": {"gte": malicious_dt}}},
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_nested_jsonb_filtering(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test complex nested JSONB object filtering"""
    test_workspace, test_peer = sample_data

    session_id = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Create messages with deeply nested metadata
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Complex nested data",
                    "peer_id": test_peer.name,
                    "metadata": {
                        "user": {
                            "profile": {"level": 5, "premium": True},
                            "settings": {"notifications": True},
                        },
                        "tags": ["important", "urgent"],
                        "scores": [1, 2, 3, 4, 5],
                    },
                },
                {
                    "content": "Simple data",
                    "peer_id": test_peer.name,
                    "metadata": {
                        "user": {"profile": {"level": 1, "premium": False}},
                        "tags": ["normal"],
                    },
                },
            ]
        },
    )

    # Test nested object filtering - this might not work as expected
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"user": {"profile": {"premium": True}}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert "Complex nested data" in data["items"][0]["content"]


@pytest.mark.asyncio
async def test_multiple_operators_same_field(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test multiple comparison operators on the same field"""
    test_workspace, test_peer = sample_data

    session_id = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Create messages with scores for range testing
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Low score",
                    "peer_id": test_peer.name,
                    "metadata": {"score": 1},
                },
                {
                    "content": "Mid score",
                    "peer_id": test_peer.name,
                    "metadata": {"score": 5},
                },
                {
                    "content": "High score",
                    "peer_id": test_peer.name,
                    "metadata": {"score": 10},
                },
                {
                    "content": "Very high",
                    "peer_id": test_peer.name,
                    "metadata": {"score": 15},
                },
            ]
        },
    )

    # Test range query: score >= 3 AND score <= 8
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"gte": 3, "lte": 8}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1  # Only "Mid score" with score 5
    assert "Mid score" in data["items"][0]["content"]


@pytest.mark.asyncio
async def test_empty_and_null_filter_handling(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test handling of empty and null filters"""
    test_workspace, test_peer = sample_data

    # Test completely empty filter
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list", json={"filter": {}}
    )
    assert response.status_code == 200
    data = response.json()
    # Should return all peers
    assert len(data["items"]) >= 1

    # Test null filter
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list", json={"filter": None}
    )
    assert response.status_code == 200

    # Test empty comparison dict
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"metadata": {"role": {}}}},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_unicode_and_special_characters(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test filtering with unicode and special characters"""
    test_workspace, test_peer = sample_data

    # Create peer with unicode metadata
    # NOTE: peer names are validated to only contain alphanumerics
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": test_peer.name,
            "metadata": {
                "description": "Héllo Wörld! 你好世界 🌍",
                "tags": ["spëcial", "ünîcode", "émojis🎉"],
            },
        },
    )

    # Test unicode in metadata contains
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"metadata": {"description": {"icontains": "wörld"}}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert test_peer.name in found_names


@pytest.mark.asyncio
async def test_case_sensitivity_edge_cases(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test case sensitivity in various contexts"""
    test_workspace, test_peer = sample_data

    # Create peers with mixed case data
    peer1_name = str(generate_nanoid())
    peer2_name = str(generate_nanoid())

    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer1_name,
            "metadata": {"Role": "Admin", "Department": "ENGINEERING"},
        },
    )
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={
            "name": peer2_name,
            "metadata": {"role": "admin", "department": "engineering"},
        },
    )

    # Test exact case matching (should be case sensitive for JSONB)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"metadata": {"Role": "Admin"}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer1_name in found_names
    assert peer2_name not in found_names  # Different case

    # Test icontains for case insensitive search
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"metadata": {"Department": {"icontains": "engineering"}}}},
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]
    assert peer1_name in found_names  # Should match ENGINEERING


@pytest.mark.asyncio
async def test_malformed_filter_structures(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test handling of malformed filter structures"""
    test_workspace, test_peer = sample_data

    malformed_filters = [
        # Invalid logical operator structures
        {"AND": "not_a_list"},
        {"OR": {"invalid": "structure"}},
        {"NOT": None},
        # Invalid comparison structures
        {"metadata": {"score": {"gte": [1, 2, 3]}}},  # Array instead of single value
        # Mixed valid/invalid
        {"AND": [{"valid_field": "value"}, "invalid_structure"]},
    ]

    for malformed_filter in malformed_filters:
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/peers/list",
            json={"filter": malformed_filter},
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_performance_with_complex_filters(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test performance with very complex nested filters"""
    test_workspace, test_peer = sample_data

    # Create a very complex filter structure
    complex_filter = {
        "AND": [
            {
                "OR": [
                    {"metadata": {"type": "A"}},
                    {"metadata": {"type": "B"}},
                    {"metadata": {"type": "C"}},
                ]
            },
            {
                "NOT": [
                    {
                        "AND": [
                            {"metadata": {"status": "inactive"}},
                            {"metadata": {"priority": {"lt": 5}}},
                        ]
                    }
                ]
            },
            {
                "OR": [
                    {"metadata": {"score": {"gte": 80}}},
                    {
                        "AND": [
                            {"metadata": {"premium": True}},
                            {"metadata": {"level": {"in": [1, 2, 3, 4, 5]}}},
                        ]
                    },
                ]
            },
        ]
    }

    # This should complete without timeout
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": complex_filter},
    )
    assert response.status_code == 200
    # Should return results in reasonable time (this is more of a performance test)


@pytest.mark.asyncio
async def test_filter_precedence_and_grouping(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test that filter precedence and grouping works as expected"""
    test_workspace, test_peer = sample_data

    # Create test data to verify precedence
    peers_data = [
        {
            "name": str(generate_nanoid()),
            "metadata": {"role": "admin", "dept": "eng", "level": 1},
        },
        {
            "name": str(generate_nanoid()),
            "metadata": {"role": "user", "dept": "eng", "level": 2},
        },
        {
            "name": str(generate_nanoid()),
            "metadata": {"role": "admin", "dept": "sales", "level": 3},
        },
        {
            "name": str(generate_nanoid()),
            "metadata": {"role": "user", "dept": "sales", "level": 4},
        },
    ]

    for peer_data in peers_data:
        client.post(f"/v2/workspaces/{test_workspace.name}/peers", json=peer_data)

    # Test: (admin OR user) AND (eng OR high level)
    # This should test that grouping works correctly
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": [
                    {
                        "OR": [
                            {"metadata": {"role": "admin"}},
                            {"metadata": {"role": "user"}},
                        ]
                    },
                    {
                        "OR": [
                            {"metadata": {"dept": "eng"}},
                            {"metadata": {"level": {"gte": 3}}},
                        ]
                    },
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    found_names = [item["id"] for item in data["items"]]

    # Should match:
    # - admin+eng+1: matches (admin OR user) AND (eng OR level>=3) ✓
    # - user+eng+2: matches (admin OR user) AND (eng OR level>=3) ✓
    # - admin+sales+3: matches (admin OR user) AND (eng OR level>=3) ✓
    # - user+sales+4: matches (admin OR user) AND (eng OR level>=3) ✓
    # So all should match
    assert len(found_names) == 4


@pytest.mark.asyncio
async def test_jsonb_contains_vs_equality_semantics(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test the difference between JSONB contains and equality"""
    test_workspace, test_peer = sample_data

    session_id = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Create messages with different JSONB structures
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Exact match",
                    "peer_id": test_peer.name,
                    "metadata": {"role": "admin"},  # Simple key-value
                },
                {
                    "content": "Superset",
                    "peer_id": test_peer.name,
                    "metadata": {
                        "role": "admin",
                        "department": "engineering",
                    },  # Contains role=admin plus more
                },
                {
                    "content": "Different",
                    "peer_id": test_peer.name,
                    "metadata": {"role": "user"},  # Different value
                },
            ]
        },
    )

    # Test JSONB contains behavior - should match both exact and superset
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"role": "admin"}}},
    )
    assert response.status_code == 200
    data = response.json()
    contents = [item["content"] for item in data["items"]]
    assert "Exact match" in contents
    assert "Superset" in contents  # JSONB contains should match this
    assert "Different" not in contents


@pytest.mark.asyncio
async def test_wildcard_edge_cases_comprehensive(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test comprehensive edge cases for wildcard behavior"""
    test_workspace, test_peer = sample_data

    peer_name = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/peers",
        json={"name": peer_name, "metadata": {"role": "admin", "level": 5}},
    )

    # Test wildcard with comparison operators
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {"metadata": {"level": {"gte": "*"}}}  # Wildcard in comparison
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Wildcard in comparison should be ignored, returning all peers
    assert len(data["items"]) >= 2

    # Test wildcard in array (in operator)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "metadata": {"role": {"in": ["*", "admin"]}}
            }  # Mixed wildcard and value
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Should return all peers since "*" in array means match all
    assert len(data["items"]) >= 2

    # Test multiple wildcards
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": [
                    {"id": "*"},
                    {"metadata": {"role": "*"}},
                    {"metadata": {"level": {"gte": 3}}},  # Only this should filter
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Should only apply the level filter
    found_names = [item["id"] for item in data["items"]]
    assert peer_name in found_names


@pytest.mark.asyncio
async def test_error_logging_and_debugging(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test that errors are logged appropriately for debugging"""
    test_workspace, test_peer = sample_data

    # Test scenarios that should generate an error
    problematic_filters = [
        {"nonexistent_column": "value"},
        {"created_at": {"gte": "invalid-date"}},
    ]

    for filter_dict in problematic_filters:
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/peers/list",
            json={"filter": filter_dict},
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_boundary_conditions_numeric(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test boundary conditions for numeric comparisons"""
    test_workspace, test_peer = sample_data

    session_id = str(generate_nanoid())
    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions",
        json={"id": session_id, "peer_names": {test_peer.name: {}}},
    )

    # Create messages with boundary values
    boundary_values = [
        {"content": "Zero", "metadata": {"score": 0}},
        {"content": "Negative", "metadata": {"score": -1}},
        {"content": "Float", "metadata": {"score": 3.14159}},
        {"content": "Large int", "metadata": {"score": 2147483647}},  # Max 32-bit int
        {
            "content": "Very large",
            "metadata": {"score": 9223372036854775807},
        },  # Max 64-bit int
        {"content": "Small float", "metadata": {"score": 0.0000001}},
    ]

    client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages",
        json={
            "messages": [
                {
                    "content": msg["content"],
                    "peer_id": test_peer.name,
                    "metadata": msg["metadata"],
                }
                for msg in boundary_values
            ]
        },
    )

    # Test boundary conditions
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"gte": 0}}}},
    )
    assert response.status_code == 200
    data = response.json()
    # Should include zero and all positive values
    contents = [item["content"] for item in data["items"]]
    assert "Zero" in contents
    assert "Negative" not in contents

    # Test floating point precision
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{session_id}/messages/list",
        json={"filter": {"metadata": {"score": {"gt": 3.14}}}},
    )
    assert response.status_code == 200
    data = response.json()
    contents = [item["content"] for item in data["items"]]
    assert "Float" in contents  # 3.14159 > 3.14
