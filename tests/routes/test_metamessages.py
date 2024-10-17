import pytest

from src import models  # Import your SQLAlchemy models


@pytest.mark.asyncio
async def test_create_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/metamessages",
        json={
            "message_id": str(test_message.public_id),
            "content": "Test Metamessage",
            "metadata": {},
            "metamessage_type": "test_type",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message_id"] == str(test_message.public_id)
    assert data["content"] == "Test Metamessage"
    assert data["metadata"] == {}
    assert data["metamessage_type"] == "test_type"


@pytest.mark.asyncio
async def test_get_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()
    test_metamessage = models.Metamessage(
        message_id=test_message.public_id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    db_session.add(test_metamessage)
    await db_session.commit()

    response = client.get(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/metamessages/{test_metamessage.public_id}?message_id={test_message.public_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message_id"] == str(test_message.public_id)
    assert data["content"] == "Test Metamessage"
    assert data["metadata"] == {}
    assert data["metamessage_type"] == "test_type"


@pytest.mark.asyncio
async def test_update_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(user_id=test_user.public_id)
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id, content="Test message", is_user=True
    )
    db_session.add(test_message)
    await db_session.commit()
    test_metamessage = models.Metamessage(
        message_id=test_message.public_id,
        content="Test Metamessage",
        metadata={},
        metamessage_type="test_type",
    )
    db_session.add(test_metamessage)
    await db_session.commit()

    response = client.put(
        f"/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/metamessages/{test_metamessage.public_id}",
        json={
            "message_id": str(test_message.public_id),
            "metadata": {"new_key": "new_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}
