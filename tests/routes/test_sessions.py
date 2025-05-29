import pytest

from src import models  # Import your SQLAlchemy models


def test_create_session(client, sample_data):
    test_app, test_user = sample_data
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {}
    assert "id" in data


def test_create_session_with_metadata(client, sample_data):
    test_app, test_user = sample_data
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={
            "metadata": {"session_key": "session_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"session_key": "session_value"}
    assert "id" in data


@pytest.mark.asyncio
async def test_get_sessions(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={
            "metadata": {"test_key": "test_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test_key": "test_value"}
    assert "id" in data

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"filter": {"test_key": "test_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["metadata"] == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_get_sessions_filter_functionality(client, db_session, sample_data):
    """Test comprehensive filter functionality for sessions endpoint"""
    test_app, test_user = sample_data
    
    # Create multiple sessions with different metadata
    session1_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"category": "work", "priority": "high", "status": "active"}},
    )
    assert session1_response.status_code == 200
    session1_id = session1_response.json()["id"]
    
    session2_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"category": "personal", "priority": "low", "tags": ["vacation", "planning"]}},
    )
    assert session2_response.status_code == 200
    session2_id = session2_response.json()["id"]
    
    session3_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"category": "work", "priority": "medium", "nested": {"level": 1, "type": "project"}}},
    )
    assert session3_response.status_code == 200
    session3_id = session3_response.json()["id"]
    
    # Test 1: Filter by single key-value pair
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"filter": {"category": "work"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    work_sessions = [item for item in data["items"] if item["id"] in [session1_id, session3_id]]
    assert len(work_sessions) == 2
    
    # Test 2: Filter by multiple key-value pairs (AND logic)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"filter": {"category": "work", "priority": "high"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == session1_id
    
    # Test 3: Filter by nested object
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"filter": {"nested": {"level": 1}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == session3_id
    
    # Test 4: Filter by array element
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"filter": {"tags": ["vacation", "planning"]}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == session2_id
    
    # Test 5: Filter with no matches
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"filter": {"nonexistent": "value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 0
    
    # Test 6: Empty filter (should return all sessions)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"filter": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) >= 3
    
    # Test 7: No filter parameter (should return all sessions)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) >= 3


@pytest.mark.asyncio
async def test_get_sessions_is_active_filter(client, db_session, sample_data):
    """Test is_active filtering functionality"""
    test_app, test_user = sample_data
    
    # Create an active session
    active_session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"type": "active_test"}},
    )
    assert active_session_response.status_code == 200
    active_session_id = active_session_response.json()["id"]
    
    # Create another session and delete it (mark as inactive)
    inactive_session_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"type": "inactive_test"}},
    )
    assert inactive_session_response.status_code == 200
    inactive_session_id = inactive_session_response.json()["id"]
    
    # Delete the second session to make it inactive
    delete_response = client.delete(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{inactive_session_id}"
    )
    assert delete_response.status_code == 200
    
    # Test 1: Filter for active sessions only
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"is_active": True},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should include the active session but not the inactive one
    active_sessions = [item for item in data["items"] if item["id"] == active_session_id]
    inactive_sessions = [item for item in data["items"] if item["id"] == inactive_session_id]
    
    assert len(active_sessions) == 1
    assert len(inactive_sessions) == 0
    assert active_sessions[0]["is_active"] is True
    
    # Test 2: Filter for inactive sessions only
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"is_active": False},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should include only the inactive session
    all_active = [item for item in data["items"] if item["id"] == active_session_id]
    all_inactive = [item for item in data["items"] if item["id"] == inactive_session_id]
    
    assert len(all_active) == 0
    assert len(all_inactive) == 1
    assert all_inactive[0]["is_active"] is False
    
    # Test 3: Default behavior (no is_active filter should return all sessions)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should include both active and inactive sessions by default
    default_active = [item for item in data["items"] if item["id"] == active_session_id]
    default_inactive = [item for item in data["items"] if item["id"] == inactive_session_id]
    
    assert len(default_active) == 1
    assert len(default_inactive) == 1
    assert default_active[0]["is_active"] is True
    assert default_inactive[0]["is_active"] is False


@pytest.mark.asyncio
async def test_get_sessions_reverse_order(client, db_session, sample_data):
    """Test reverse order functionality"""
    test_app, test_user = sample_data
    
    # Create sessions in sequence
    session1_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"order": "first"}},
    )
    assert session1_response.status_code == 200
    session1_id = session1_response.json()["id"]
    
    session2_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"order": "second"}},
    )
    assert session2_response.status_code == 200
    session2_id = session2_response.json()["id"]
    
    # Test 1: Normal order (oldest first)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Find positions of our test sessions
    session1_pos = next(i for i, item in enumerate(data["items"]) if item["id"] == session1_id)
    session2_pos = next(i for i, item in enumerate(data["items"]) if item["id"] == session2_id)
    assert session1_pos < session2_pos  # First session should come before second
    
    # Test 2: Reverse order (newest first)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={},
        params={"reverse": True},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Find positions of our test sessions in reverse order
    session1_pos_rev = next(i for i, item in enumerate(data["items"]) if item["id"] == session1_id)
    session2_pos_rev = next(i for i, item in enumerate(data["items"]) if item["id"] == session2_id)
    assert session2_pos_rev < session1_pos_rev  # Second session should come before first in reverse


@pytest.mark.asyncio
async def test_get_sessions_combined_filters(client, db_session, sample_data):
    """Test combining filter, is_active, and reverse parameters"""
    test_app, test_user = sample_data
    
    # Create sessions with specific metadata
    session1_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"project": "alpha", "status": "completed"}},
    )
    assert session1_response.status_code == 200
    session1_id = session1_response.json()["id"]
    
    session2_response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions",
        json={"metadata": {"project": "alpha", "status": "active"}},
    )
    assert session2_response.status_code == 200
    session2_id = session2_response.json()["id"]
    
    # Delete first session to make it inactive
    delete_response = client.delete(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{session1_id}"
    )
    assert delete_response.status_code == 200
    
    # Test: Filter by project + active sessions only + reverse order
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/list",
        json={"filter": {"project": "alpha"}, "is_active": True},
        params={"reverse": True},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should only return the active session with project "alpha"
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == session2_id
    assert data["items"][0]["is_active"] is True
    assert data["items"][0]["metadata"]["project"] == "alpha"


@pytest.mark.asyncio
async def test_empty_update_session(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id, h_metadata={}, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}",
        json={},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_delete_metadata(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        h_metadata={"default": "value"},
        app_id=test_app.public_id,
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}",
        json={"metadata": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {}


@pytest.mark.asyncio
async def test_update_session(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id, h_metadata={}, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


@pytest.mark.asyncio
async def test_delete_session(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id, h_metadata={}, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()
    response = client.delete(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}"
    )
    assert response.status_code == 200
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions?session_id={test_session.public_id}"
    )
    data = response.json()
    assert data["is_active"] is False


@pytest.mark.asyncio
async def test_clone_session(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        h_metadata={"test": "key"},
        app_id=test_app.public_id,
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        h_metadata={"key": "value"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    test_message2 = models.Message(
        session_id=test_session.public_id,
        content="Test message 2",
        is_user=True,
        h_metadata={"key": "value2"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    db_session.add(test_message)
    db_session.add(test_message2)
    await db_session.commit()

    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/clone",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test": "key"}

    print(data)

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{data['id']}/messages/list",
        json={},
    )

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert len(data["items"]) == 2

    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["is_user"] is True
    assert data["items"][0]["metadata"] == {"key": "value"}

    assert data["items"][1]["content"] == "Test message 2"
    assert data["items"][1]["is_user"] is True
    assert data["items"][1]["metadata"] == {"key": "value2"}


@pytest.mark.asyncio
async def test_partial_clone_session(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        h_metadata={"test": "key"},
        app_id=test_app.public_id,
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        h_metadata={"key": "value"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    test_message2 = models.Message(
        session_id=test_session.public_id,
        content="Test message 2",
        is_user=True,
        h_metadata={"key": "value2"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )

    test_message3 = models.Message(
        session_id=test_session.public_id,
        content="Test message 2",
        is_user=True,
        h_metadata={"key": "value2"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )

    db_session.add(test_message)
    db_session.add(test_message2)
    db_session.add(test_message3)
    await db_session.commit()

    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/clone?message_id={test_message2.public_id}",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test": "key"}

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{data['id']}/messages/list",
        json={},
    )

    data = response.json()
    assert len(data["items"]) > 0
    assert len(data["items"]) == 2

    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["is_user"] is True
    assert data["items"][0]["metadata"] == {"key": "value"}

    assert data["items"][1]["content"] == "Test message 2"
    assert data["items"][1]["is_user"] is True
    assert data["items"][1]["metadata"] == {"key": "value2"}


@pytest.mark.asyncio
async def test_deep_clone_session(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        h_metadata={"test": "key"},
        app_id=test_app.public_id,
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        h_metadata={"key": "value"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    test_message2 = models.Message(
        session_id=test_session.public_id,
        content="Test message 2",
        is_user=True,
        h_metadata={"key": "value2"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    db_session.add(test_message)
    db_session.add(test_message2)
    await db_session.commit()

    test_metamessage_1 = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage 1",
        h_metadata={},
        label="test_type",
        app_id=test_app.public_id,
    )
    test_metamessage_2 = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage 2",
        h_metadata={},
        label="test_type",
        app_id=test_app.public_id,
    )
    test_metamessage_3 = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=test_message2.public_id,
        content="Test Metamessage 3",
        h_metadata={},
        label="test_type",
        app_id=test_app.public_id,
    )
    test_metamessage_4 = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=test_message2.public_id,
        content="Test Metamessage 4",
        h_metadata={},
        app_id=test_app.public_id,
        label="test_type_2",
    )

    db_session.add(test_metamessage_1)
    db_session.add(test_metamessage_2)
    db_session.add(test_metamessage_3)
    db_session.add(test_metamessage_4)
    await db_session.commit()

    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/clone?deep_copy=true",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test": "key"}

    cloned_session_id = data["id"]

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{cloned_session_id}/messages/list",
        json={},
    )

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert len(data["items"]) == 2

    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["is_user"] is True
    assert data["items"][0]["metadata"] == {"key": "value"}

    assert data["items"][1]["content"] == "Test message 2"
    assert data["items"][1]["is_user"] is True
    assert data["items"][1]["metadata"] == {"key": "value2"}

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages/list",
        json={"session_id": cloned_session_id},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) > 0
    assert len(data["items"]) == 4
    assert data["items"][0]["content"] == "Test Metamessage 1"
    assert data["items"][0]["label"] == "test_type"
    assert data["items"][0]["metamessage_type"] == "test_type"
    assert data["items"][0]["metadata"] == {}
    assert data["items"][1]["content"] == "Test Metamessage 2"
    assert data["items"][1]["label"] == "test_type"
    assert data["items"][1]["metamessage_type"] == "test_type"
    assert data["items"][1]["metadata"] == {}
    assert data["items"][2]["content"] == "Test Metamessage 3"
    assert data["items"][2]["label"] == "test_type"
    assert data["items"][2]["metamessage_type"] == "test_type"
    assert data["items"][2]["metadata"] == {}
    assert data["items"][3]["content"] == "Test Metamessage 4"
    assert data["items"][3]["label"] == "test_type_2"
    assert data["items"][3]["metamessage_type"] == "test_type_2"
    assert data["items"][3]["metadata"] == {}


@pytest.mark.asyncio
async def test_partial_deep_clone_session(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        h_metadata={"test": "key"},
        app_id=test_app.public_id,
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        h_metadata={"key": "value"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    test_message2 = models.Message(
        session_id=test_session.public_id,
        content="Test message 2",
        is_user=True,
        h_metadata={"key": "value2"},
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    db_session.add(test_message)
    db_session.add(test_message2)
    await db_session.commit()

    test_metamessage_1 = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage 1",
        h_metadata={},
        app_id=test_app.public_id,
        label="test_type",
    )
    test_metamessage_2 = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage 2",
        h_metadata={},
        app_id=test_app.public_id,
        label="test_type",
    )
    test_metamessage_3 = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=test_message2.public_id,
        content="Test Metamessage 3",
        h_metadata={},
        app_id=test_app.public_id,
        label="test_type",
    )
    test_metamessage_4 = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=test_message2.public_id,
        content="Test Metamessage 4",
        h_metadata={},
        app_id=test_app.public_id,
        label="test_type_2",
    )

    db_session.add(test_metamessage_1)
    db_session.add(test_metamessage_2)
    db_session.add(test_metamessage_3)
    db_session.add(test_metamessage_4)
    await db_session.commit()

    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/clone?deep_copy=true&message_id={test_message.public_id}",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"test": "key"}

    cloned_session_id = data["id"]

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{cloned_session_id}/messages/list",
        json={},
    )

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert len(data["items"]) == 1

    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["is_user"] is True
    assert data["items"][0]["metadata"] == {"key": "value"}

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages/list",
        json={"session_id": cloned_session_id},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) > 0
    assert len(data["items"]) == 2
    assert data["items"][0]["content"] == "Test Metamessage 1"
    assert data["items"][0]["label"] == "test_type"
    assert data["items"][0]["metamessage_type"] == "test_type"
    assert data["items"][0]["metadata"] == {}
    assert data["items"][1]["content"] == "Test Metamessage 2"
    assert data["items"][1]["label"] == "test_type"
    assert data["items"][1]["metamessage_type"] == "test_type"
    assert data["items"][1]["metadata"] == {}
