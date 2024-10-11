import pytest

from src import models  # Import your SQLAlchemy models


@pytest.mark.asyncio
async def test_create_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions/{test_session.id}/metamessages",
        json={
            "message_id": str(test_message.id),
            "content": "Test Metamessage",
            "metadata": {},
            "metamessage_type": "test_type",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message_id"] == str(test_message.id)
    assert data["content"] == "Test Metamessage"
    assert data["metadata"] == {}
    assert data["metamessage_type"] == "test_type"


@pytest.mark.asyncio
async def test_get_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()
    test_metamessage = models.Metamessage(
        message_id=test_message.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    db_session.add(test_metamessage)
    await db_session.commit()

    response = client.get(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions/{test_session.id}/metamessages/{test_metamessage.id}?message_id={test_message.id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message_id"] == str(test_message.id)
    assert data["content"] == "Test Metamessage"
    assert data["metadata"] == {}
    assert data["metamessage_type"] == "test_type"


@pytest.mark.asyncio
async def test_get_metamessages(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()
    test_metamessage_1 = models.Metamessage(
        message_id=test_message.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    test_metamessage_2 = models.Metamessage(
        message_id=test_message.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    test_metamessage_3 = models.Metamessage(
        message_id=test_message.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    test_metamessage_4 = models.Metamessage(
        message_id=test_message.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type_2",
    )
    db_session.add(test_metamessage_1)
    db_session.add(test_metamessage_2)
    db_session.add(test_metamessage_3)
    db_session.add(test_metamessage_4)
    await db_session.commit()

    response = client.get(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions/{test_session.id}/metamessages?metamessage_type=test_type"
    )

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert len(data["items"]) == 3
    assert data["items"][0]["content"] == "Test Metamessage"
    assert data["items"][0]["metamessage_type"] == "test_type"
    assert data["items"][0]["metadata"] == {}


@pytest.mark.asyncio
async def test_get_metamessage_by_user(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a 3 test sessions
    test_session_1 = models.Session(user_id=test_user.id)
    test_session_2 = models.Session(user_id=test_user.id)
    test_session_3 = models.Session(user_id=test_user.id)
    db_session.add(test_session_1)
    db_session.add(test_session_2)
    db_session.add(test_session_3)
    await db_session.commit()

    # Create a message in each session
    test_message_1 = models.Message(
        session_id=test_session_1.id, content="Test message", is_user=True
    )
    test_message_2 = models.Message(
        session_id=test_session_2.id, content="Test message", is_user=True
    )
    test_message_3 = models.Message(
        session_id=test_session_3.id, content="Test message", is_user=True
    )
    db_session.add(test_message_1)
    db_session.add(test_message_2)
    db_session.add(test_message_3)
    await db_session.commit()

    # create a metamessage on each message
    test_metamessage_1 = models.Metamessage(
        message_id=test_message_1.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    test_metamessage_2 = models.Metamessage(
        message_id=test_message_2.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    test_metamessage_3 = models.Metamessage(
        message_id=test_message_3.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    test_metamessage_4 = models.Metamessage(
        message_id=test_message_3.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type_2",
    )
    db_session.add(test_metamessage_1)
    db_session.add(test_metamessage_2)
    db_session.add(test_metamessage_3)
    db_session.add(test_metamessage_4)
    await db_session.commit()

    response = client.get(
        f"/apps/{test_app.id}/users/{test_user.id}/metamessages?metamessage_type=test_type"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) > 0
    assert len(data["items"]) == 3
    assert data["items"][0]["content"] == "Test Metamessage"
    assert data["items"][0]["metamessage_type"] == "test_type"
    assert data["items"][0]["metadata"] == {}


@pytest.mark.asyncio
async def test_update_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()
    test_metamessage = models.Metamessage(
        message_id=test_message.id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    db_session.add(test_metamessage)
    await db_session.commit()

    response = client.put(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions/{test_session.id}/metamessages/{test_metamessage.id}",
        json={"message_id": str(test_message.id), "metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}
