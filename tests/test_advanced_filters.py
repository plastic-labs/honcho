"""
Tests for advanced filter functionality including logical operators,
comparison operators, and wildcards across multiple models.
"""

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

    # Test wildcard for name field - should match all peers regardless of name
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": [
                    {"name": "*"},  # Wildcard matches all names
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
                "name": {"in": ["*"]}  # Wildcard in comparison should also match all
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
        json={"filter": {"OR": [{"AND": []}, {"name": test_peer.name}]}},
    )
    assert response.status_code == 200

    # Test filter with non-existent columns (should be ignored gracefully)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={"filter": {"non_existent_column": "value"}},
    )
    assert response.status_code == 200  # Should not crash

    # Test mixed wildcards and regular values
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/peers/list",
        json={
            "filter": {
                "AND": [
                    {"name": "*"},  # Wildcard
                    {"name": test_peer.name},  # Regular value
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
        json={"filter": {"metadata": {"role": "admin"}, "name": peer_name}},
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
