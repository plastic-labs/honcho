import pytest

from src import models  # Import your SQLAlchemy models


@pytest.mark.asyncio
async def test_create_message(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.public_id)
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages",
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
    test_session = models.Session(user_id=test_user.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/get",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["is_user"] is True
    assert data["items"][0]["metadata"] == {}


@pytest.mark.asyncio
async def test_get_filtered_messages(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session and message
    test_session = models.Session(user_id=test_user.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        h_metadata={"key": "value"},
    )
    test_message2 = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        h_metadata={"key": "value2"},
    )
    db_session.add(test_message)
    db_session.add(test_message2)
    await db_session.commit()

    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/get",
        json={"filter": {"key": "value2"}},
    )
    assert response.status_code == 200
    data = response.json()
    print(data)
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["is_user"] is True
    assert data["items"][0]["metadata"] == {"key": "value2"}


@pytest.mark.asyncio
async def test_update_message(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session and message
    test_session = models.Session(user_id=test_user.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/{test_message.public_id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}
