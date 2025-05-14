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
