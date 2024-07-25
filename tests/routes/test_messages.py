import pytest

from src import models  # Import your SQLAlchemy models


@pytest.mark.asyncio
async def test_create_message(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.id, location_id="test_location", metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions/{test_session.id}/messages",
        json={
            "content": "Test message",
            "is_user": True,
            "metadata": {"message_key": "message_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Test message"
    assert data["is_user"] is True
    assert data["metadata"] == {"message_key": "message_value"}
    assert "id" in data


@pytest.mark.asyncio
async def test_get_messages(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session and message
    test_session = models.Session(
        user_id=test_user.id, location_id="test_location", metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.id, content="Test message", is_user=True, metadata={}
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.get(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions/{test_session.id}/messages"
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["is_user"] is True
    assert data["items"][0]["metadata"] == {}


@pytest.mark.asyncio
async def test_update_message(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session and message
    test_session = models.Session(
        user_id=test_user.id, location_id="test_location", metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.id, content="Test message", is_user=True, metadata={}
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions/{test_session.id}/messages/{test_message.id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}
