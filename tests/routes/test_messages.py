import pytest

from src import models  # Import your SQLAlchemy models


@pytest.mark.asyncio
async def test_create_message(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages",
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
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
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
async def test_messages_pagination(client, db_session, sample_data):
    """Test pagination of messages with different page sizes."""
    test_app, test_user = sample_data

    # Create a test session
    test_session = models.Session(
        app_id=test_app.public_id, user_id=test_user.public_id
    )
    db_session.add(test_session)
    await db_session.commit()

    # Create 50 test messages
    for i in range(50):
        test_message = models.Message(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,
            content=f"Pagination test message {i}",
            is_user=i % 2 == 0,  # Alternating user/non-user messages
            h_metadata={"index": i},
        )
        db_session.add(test_message)

    await db_session.commit()

    # Test case 1: Default pagination (page 1, default size)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()

    # Check pagination metadata
    assert data["page"] == 1
    assert "total" in data
    assert data["total"] == 50  # Total count should be 50

    # Test case 2: 5 pages of 10 items each
    for page in range(1, 6):
        response = client.post(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/list?page={page}&size=10",
            json={},
        )
        assert response.status_code == 200
        data = response.json()

        # Check pagination metadata
        assert data["page"] == page
        assert data["size"] == 10
        assert data["total"] == 50

        # Check items count (should be 10 for all pages)
        expected_items = 10
        assert len(data["items"]) == expected_items

        # Verify we have the correct page of items
        start_idx = (page - 1) * 10
        for i, item in enumerate(data["items"]):
            expected_idx = start_idx + i
            assert item["metadata"]["index"] == expected_idx

    # Test case 3: 2 pages of 25 items each
    for page in range(1, 3):
        response = client.post(
            f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/list?page={page}&size=25",
            json={},
        )
        assert response.status_code == 200
        data = response.json()

        # Check pagination metadata
        assert data["page"] == page
        assert data["size"] == 25
        assert data["total"] == 50

        # Check items count
        expected_items = 25
        assert len(data["items"]) == expected_items

        # Verify we have the correct page of items
        start_idx = (page - 1) * 25
        for i, item in enumerate(data["items"]):
            expected_idx = start_idx + i
            assert item["metadata"]["index"] == expected_idx

    # Test case 4: 1 page of 50 items
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/list?page=1&size=50",
        json={},
    )
    assert response.status_code == 200
    data = response.json()

    # Check pagination metadata
    assert data["page"] == 1
    assert data["size"] == 50
    assert data["total"] == 50

    # Check items count
    assert len(data["items"]) == 50

    # Verify all items are included
    all_indices = {item["metadata"]["index"] for item in data["items"]}
    assert all_indices == set(range(50))

    # Test case 5: Test with reverse=true (newest first, reverse order)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/messages/list?page=1&size=50&reverse=true",
        json={},
    )
    assert response.status_code == 200
    data = response.json()

    # Check items are in reverse order when reverse=true
    for i, item in enumerate(data["items"]):
        expected_idx = 49 - i
        assert item["metadata"]["index"] == expected_idx


@pytest.mark.asyncio
async def test_get_filtered_messages(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session and message
    test_session = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
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
    test_session = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
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
    test_session = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id,
        content="Test message",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
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
    test_session = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()

    # Create batch of test messages
    test_messages = {
        "messages": [
            {
                "content": f"Test message {i}",
                "is_user": i % 2 == 0,  # Alternating user/non-user messages
                "metadata": {"batch_index": i},
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
    test_session = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
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
