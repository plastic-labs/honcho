import pytest

from src import models  # Import your SQLAlchemy models


@pytest.mark.asyncio
async def test_create_message(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.public_id, app_id=test_app.public_id)
    db_session.add(test_session)
    await db_session.commit()

    ts = "2020-01-01T12:34:56Z"
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages",
        json={
            "content": "Test message",
            "is_user": True,
            "metadata": {"message_key": "message_value"},
            "created_at": ts,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Test message"
    assert data["is_user"] is True
    assert data["metadata"] == {"message_key": "message_value"}
    assert "id" in data
    assert data["created_at"].startswith("2020-01-01T12:34:56")


@pytest.mark.asyncio
async def test_get_messages(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session and message
    test_session = models.Session(user_id=test_user.public_id, app_id=test_app.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id, content="Test message", is_user=True, app_id=test_app.public_id, user_id=test_user.public_id
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/list",
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
    test_session = models.Session(user_id=test_user.public_id, app_id=test_app.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
        h_metadata={"key": "value"},
    )
    test_message2 = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
        h_metadata={"key": "value2"},
    )
    db_session.add(test_message)
    db_session.add(test_message2)
    await db_session.commit()

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/list",
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
    test_session = models.Session(user_id=test_user.public_id, app_id=test_app.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id, content="Test message", is_user=True, app_id=test_app.public_id, user_id=test_user.public_id
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/{test_message.public_id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


@pytest.mark.asyncio
async def test_update_message_empty_metadata(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session and message
    test_session = models.Session(user_id=test_user.public_id, app_id=test_app.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id, content="Test message", is_user=True, app_id=test_app.public_id, user_id=test_user.public_id
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/{test_message.public_id}",
        json={"metadata": None},
    )
    assert response.status_code == 422
    data = response.json()
    print(data)
    # assert data["detail"] == "Message metadata cannot be empty"


@pytest.mark.asyncio
async def test_create_batch_messages(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.public_id, app_id=test_app.public_id)
    db_session.add(test_session)
    await db_session.commit()

    # Create batch of test messages
    ts = "2019-12-27T18:11:19Z"
    test_messages = {
        "messages": [
            {
                "content": f"Test message {i}",
                "is_user": i % 2 == 0,  # Alternating user/non-user messages
                "metadata": {"batch_index": i},
                "created_at": ts,
            }
            for i in range(3)
        ]
    }

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/batch",
        json=test_messages,
    )

    assert response.status_code == 200
    data = response.json()

    # Verify the response contains all messages
    assert len(data) == 3

    # Verify messages are in the correct order and have correct content
    for i, message in enumerate(data):
        assert message["content"] == f"Test message {i}"
        assert message["is_user"] == (i % 2 == 0)
        assert message["metadata"] == {"batch_index": i}
        assert message["created_at"].startswith(ts[:19])
        assert "id" in message
        assert message["session_id"] == test_session.public_id

    # Verify messages were actually saved to the database
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/list",
        json={},
    )
    assert response.status_code == 200
    saved_messages = response.json()["items"]
    assert len(saved_messages) == 3


@pytest.mark.asyncio
async def test_create_batch_messages_limit(client, db_session, sample_data):
    test_app, test_user = sample_data
    test_session = models.Session(user_id=test_user.public_id, app_id=test_app.public_id)
    db_session.add(test_session)
    await db_session.commit()

    # Create batch with more than 100 messages
    test_messages = {
        "messages": [
            {
                "content": f"Test message {i}",
                "is_user": i % 2 == 0,
                "metadata": {"batch_index": i},
            }
            for i in range(101)  # 101 messages
        ]
    }

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/batch",
        json=test_messages,
    )

    assert response.status_code == 422  # Validation error
    data = response.json()
    assert "messages" in data["detail"][0]["loc"]  # Error should mention messages field
