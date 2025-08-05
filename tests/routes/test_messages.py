import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


@pytest.mark.asyncio
async def test_create_message(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages",
        json={
            "messages": [
                {
                    "content": "Test message",
                    "peer_id": test_peer.name,
                    "metadata": {"message_key": "message_value"},
                }
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    message = data[0]
    assert message["content"] == "Test message"
    assert message["peer_id"] == test_peer.name
    assert message["session_id"] == test_session.name
    assert message["metadata"] == {"message_key": "message_value"}
    assert "id" in message


@pytest.mark.asyncio
async def test_create_batch_messages_with_metadata(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test batch message creation with metadata for each message"""
    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages",
        json={
            "messages": [
                {
                    "content": "Message 1",
                    "peer_id": test_peer.name,
                    "metadata": {"type": "question", "priority": "high"},
                },
                {
                    "content": "Message 2",
                    "peer_id": test_peer.name,
                    "metadata": {"type": "answer", "priority": "low"},
                },
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    # Check first message
    assert data[0]["content"] == "Message 1"
    assert data[0]["metadata"] == {"type": "question", "priority": "high"}

    # Check second message
    assert data[1]["content"] == "Message 2"
    assert data[1]["metadata"] == {"type": "answer", "priority": "low"}


@pytest.mark.asyncio
async def test_create_batch_messages_without_metadata(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test batch message creation without metadata (should default to empty dict)"""
    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages",
        json={
            "messages": [
                {
                    "content": "Message without metadata",
                    "peer_id": test_peer.name,
                }
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "Message without metadata"
    assert data[0]["metadata"] == {}


@pytest.mark.asyncio
async def test_create_batch_messages_with_null_metadata(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test batch message creation with null metadata"""
    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages",
        json={
            "messages": [
                {
                    "content": "Message with null metadata",
                    "peer_id": test_peer.name,
                    "metadata": None,
                }
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "Message with null metadata"
    assert data[0]["metadata"] == {}


@pytest.mark.asyncio
async def test_get_messages(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["peer_id"] == test_peer.name
    assert data["items"][0]["session_id"] == test_session.name
    assert data["items"][0]["metadata"] == {}


@pytest.mark.asyncio
async def test_get_messages_with_reverse(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test getting messages with reverse parameter"""
    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    # Create multiple messages to test ordering
    test_message1 = models.Message(
        session_name=test_session.name,
        content="First message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    test_message2 = models.Message(
        session_name=test_session.name,
        content="Second message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    db_session.add(test_message1)
    db_session.add(test_message2)
    await db_session.commit()

    # Test normal order
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={},
    )
    assert response.status_code == 200
    normal_data = response.json()

    # Test reversed order
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list?reverse=true",
        json={},
    )
    assert response.status_code == 200
    reversed_data = response.json()

    # Both should have items
    assert len(normal_data["items"]) >= 2
    assert len(reversed_data["items"]) >= 2

    # Order should be different (first item in normal should be last in reversed)
    if len(normal_data["items"]) > 1 and len(reversed_data["items"]) > 1:
        assert normal_data["items"][0]["id"] != reversed_data["items"][0]["id"]


@pytest.mark.asyncio
async def test_get_messages_with_empty_filter(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test getting messages with empty filter object"""
    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={"filter": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_messages_with_null_filter(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test getting messages with null filter"""
    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={"filter": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_messages_no_body(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test getting messages without any request body"""
    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list"
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_filtered_messages(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data

    # Create a test session and messages
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"key": "value"},
    )
    test_message2 = models.Message(
        session_name=test_session.name,
        content="Test message 2",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"key": "value2"},
    )
    db_session.add(test_message)
    db_session.add(test_message2)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={"filter": {"metadata": {"key": "value2"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["content"] == "Test message 2"
    assert data["items"][0]["peer_id"] == test_peer.name
    assert data["items"][0]["session_id"] == test_session.name
    assert data["items"][0]["metadata"] == {"key": "value2"}


@pytest.mark.asyncio
async def test_get_filtered_messages_with_complex_filter(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test getting messages with complex metadata filter"""
    test_workspace, test_peer = sample_data

    # Create a test session and messages
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message1 = models.Message(
        session_name=test_session.name,
        content="Message 1",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"type": "question", "priority": "high", "category": "technical"},
    )
    test_message2 = models.Message(
        session_name=test_session.name,
        content="Message 2",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"type": "answer", "priority": "high", "category": "technical"},
    )
    test_message3 = models.Message(
        session_name=test_session.name,
        content="Message 3",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"type": "question", "priority": "low", "category": "general"},
    )
    db_session.add(test_message1)
    db_session.add(test_message2)
    db_session.add(test_message3)
    await db_session.commit()

    # Test old-style filter (backward compatibility)
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={"filter": {"metadata": {"priority": "high", "category": "technical"}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    # Should return messages 1 and 2 (both have high priority and technical category)
    assert len(data["items"]) >= 2

    # Test new-style filter with AND operator
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={
            "filter": {
                "AND": [
                    {"metadata": {"priority": "high"}},
                    {"metadata": {"category": "technical"}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2

    # Test OR filter to get high priority OR question type
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={
            "filter": {
                "OR": [
                    {"metadata": {"priority": "high"}},
                    {"metadata": {"type": "question"}},
                ]
            }
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 3  # All messages should match


@pytest.mark.asyncio
async def test_update_message(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


@pytest.mark.asyncio
async def test_update_message_with_complex_metadata(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test updating message with complex metadata structure"""
    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    db_session.add(test_message)
    await db_session.commit()

    complex_metadata = {
        "tags": ["important", "follow-up"],
        "score": 8.5,
        "nested": {"category": "technical", "subcategory": "api"},
        "processed": True,
    }

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}",
        json={"metadata": complex_metadata},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == complex_metadata


@pytest.mark.asyncio
async def test_update_message_empty_metadata(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    # note that this should not change the metadata of the message.
    # this test is to ensure that the metadata is not changed when it is set to None.

    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"test_key": "test_value"},
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}",
        json={"metadata": None},
    )
    assert response.status_code == 200

    # now ensure that the metadata is not changed
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_update_message_with_empty_dict_metadata(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test updating message with empty dictionary metadata"""
    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"old_key": "old_value"},
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}",
        json={"metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {}


@pytest.mark.asyncio
async def test_get_single_message(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    test_workspace, test_peer = sample_data

    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Test message"
    assert data["peer_id"] == test_peer.name
    assert data["session_id"] == test_session.name
    assert data["workspace_id"] == test_workspace.name
    assert data["id"] == test_message.public_id


@pytest.mark.asyncio
async def test_get_nonexistent_message(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test getting a message that doesn't exist"""
    test_workspace, _ = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    nonexistent_message_id = str(generate_nanoid())
    response = client.get(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{nonexistent_message_id}"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_nonexistent_message(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test updating a message that doesn't exist"""
    test_workspace, _ = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    nonexistent_message_id = str(generate_nanoid())
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{nonexistent_message_id}",
        json={"metadata": {"key": "value"}},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_messages_for_nonexistent_session(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test creating messages for a session that doesn't exist - should create the session"""
    test_workspace, test_peer = sample_data

    nonexistent_session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{nonexistent_session_id}/messages",
        json={
            "messages": [
                {
                    "content": "Test message",
                    "peer_id": test_peer.name,
                }
            ]
        },
    )
    # Should create the session and return 200 with the created message
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "Test message"
    assert data[0]["workspace_id"] == test_workspace.name
    assert data[0]["session_id"] == nonexistent_session_id
    assert data[0]["peer_id"] == test_peer.name


@pytest.mark.asyncio
async def test_get_messages_for_nonexistent_session(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test getting messages for a session that doesn't exist - should return empty list"""
    test_workspace, _ = sample_data

    nonexistent_session_id = str(generate_nanoid())
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{nonexistent_session_id}/messages/list",
        json={},
    )
    # Should return 200 with empty results (session doesn't exist = no messages)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 0


@pytest.mark.asyncio
async def test_create_empty_batch_messages(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test creating an empty batch of messages"""
    test_workspace, _ = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages",
        json={"messages": []},
    )
    # Should return 422 for validation error (empty list not allowed)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_batch_messages_max_limit(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    """Test creating batch messages at the maximum limit (100 messages)"""
    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    # Create exactly 100 messages (the maximum allowed)
    messages = [
        {"content": f"Message {i}", "peer_id": test_peer.name, "metadata": {"index": i}}
        for i in range(100)
    ]

    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages",
        json={"messages": messages},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 100
    assert data[0]["content"] == "Message 0"
    assert data[99]["content"] == "Message 99"
