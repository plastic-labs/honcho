import pytest

from src import models  # Import your SQLAlchemy models


@pytest.mark.asyncio
async def test_create_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
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
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages",
        json={
            "session_id": str(test_session.public_id),
            "message_id": str(test_message.public_id),
            "content": "Test Metamessage",
            "metadata": {},
            "label": "test_type",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == str(test_user.public_id)
    assert data["session_id"] == str(test_session.public_id)
    assert data["message_id"] == str(test_message.public_id)
    assert data["content"] == "Test Metamessage"
    assert data["metadata"] == {}
    assert data["label"] == "test_type"
    assert data["metamessage_type"] == "test_type"


@pytest.mark.asyncio
async def test_get_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
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
    test_metamessage = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type",
    )
    db_session.add(test_metamessage)
    await db_session.commit()

    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages/{test_metamessage.public_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == str(test_user.public_id)
    assert data["app_id"] == str(test_app.public_id)
    assert data["session_id"] == str(test_session.public_id)
    assert data["message_id"] == str(test_message.public_id)
    assert data["content"] == "Test Metamessage"
    assert data["metadata"] == {}
    assert data["label"] == "test_type"
    assert data["metamessage_type"] == "test_type"


@pytest.mark.asyncio
async def test_get_metamessages_by_session(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
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

    # Create metamessages for the same session
    test_metamessage_1 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type",
    )
    test_metamessage_2 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type",
    )
    test_metamessage_3 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type",
    )
    test_metamessage_4 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type_2",
    )
    db_session.add(test_metamessage_1)
    db_session.add(test_metamessage_2)
    db_session.add(test_metamessage_3)
    db_session.add(test_metamessage_4)
    await db_session.commit()

    # Filter by session and type
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages/list",
        json={
            "session_id": str(test_session.public_id),
            "label": "test_type",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 3
    assert data["items"][0]["content"] == "Test Metamessage"
    assert data["items"][0]["label"] == "test_type"
    assert data["items"][0]["metamessage_type"] == "test_type"
    assert data["items"][0]["session_id"] == str(test_session.public_id)
    assert data["items"][0]["metadata"] == {}
    assert data["items"][0]["app_id"] == str(test_app.public_id)


@pytest.mark.asyncio
async def test_get_metamessage_by_user(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create 3 test sessions
    test_session_1 = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    test_session_2 = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    test_session_3 = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    db_session.add(test_session_1)
    db_session.add(test_session_2)
    db_session.add(test_session_3)
    await db_session.commit()

    # Create a message in each session
    test_message_1 = models.Message(
        session_id=test_session_1.public_id,
        content="Test message",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    test_message_2 = models.Message(
        session_id=test_session_2.public_id,
        content="Test message",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    test_message_3 = models.Message(
        session_id=test_session_3.public_id,
        content="Test message",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    db_session.add(test_message_1)
    db_session.add(test_message_2)
    db_session.add(test_message_3)
    await db_session.commit()

    # Create metamessages across different sessions
    test_metamessage_1 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session_1.public_id,
        message_id=test_message_1.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type",
    )
    test_metamessage_2 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session_2.public_id,
        message_id=test_message_2.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type",
    )
    test_metamessage_3 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session_3.public_id,
        message_id=test_message_3.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type",
    )
    test_metamessage_4 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session_3.public_id,
        message_id=test_message_3.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type_2",
    )
    # Create a user-level metamessage (no session/message)
    test_metamessage_5 = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        content="User level metamessage",
        h_metadata={},
        label="test_type",
    )
    db_session.add(test_metamessage_1)
    db_session.add(test_metamessage_2)
    db_session.add(test_metamessage_3)
    db_session.add(test_metamessage_4)
    db_session.add(test_metamessage_5)
    await db_session.commit()

    # Filter only by type across all user's metamessages
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages/list",
        json={"label": "test_type"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 4  # All test_type metamessages for the user
    assert data["items"][0]["content"] in ["Test Metamessage", "User level metamessage"]
    assert data["items"][0]["label"] == "test_type"
    assert data["items"][0]["metamessage_type"] == "test_type"
    assert data["items"][0]["user_id"] == str(test_user.public_id)
    assert data["items"][0]["app_id"] == str(test_app.public_id)


@pytest.mark.asyncio
async def test_create_user_level_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data

    # Create a user-level metamessage (no session or message)
    response = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages",
        json={
            "content": "User level insight",
            "metadata": {"source": "user_profile"},
            "label": "user_insight",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == str(test_user.public_id)
    assert data["session_id"] is None
    assert data["message_id"] is None
    assert data["content"] == "User level insight"
    assert data["metadata"] == {"source": "user_profile"}
    assert data["metamessage_type"] == "user_insight"
    assert data["app_id"] == str(test_app.public_id)
    assert data["label"] == "user_insight"
    assert data["metamessage_type"] == "user_insight"


@pytest.mark.asyncio
async def test_update_metamessage(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
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
    test_metamessage = models.Metamessage(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        session_id=test_session.public_id,
        message_id=test_message.public_id,
        content="Test Metamessage",
        h_metadata={},
        label="test_type",
    )
    db_session.add(test_metamessage)
    await db_session.commit()

    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages/{test_metamessage.public_id}",
        json={
            "metadata": {"new_key": "new_value"},
            "label": "updated_type",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}
    assert data["label"] == "updated_type"
    assert data["metamessage_type"] == "updated_type"
    assert data["user_id"] == str(test_user.public_id)
    assert data["session_id"] == str(test_session.public_id)
    assert data["message_id"] == str(test_message.public_id)
    assert data["app_id"] == str(test_app.public_id)


@pytest.mark.asyncio
async def test_create_metamessage_with_label_and_alias(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a common session and message for both test cases
    test_session = models.Session(
        user_id=test_user.public_id, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.commit()
    test_message = models.Message(
        session_id=test_session.public_id,
        content="Shared message content",
        is_user=True,
        app_id=test_app.public_id,
        user_id=test_user.public_id,
    )
    db_session.add(test_message)
    await db_session.commit()

    common_payload_parts = {
        "session_id": str(test_session.public_id),
        "message_id": str(test_message.public_id),
        "metadata": {"source": "input_test"},
    }
    test_value_for_type = "input_consistency_test_type"

    # 1. Create metamessage using "label"
    response_with_label = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages",
        json={
            **common_payload_parts,
            "content": "Content created with label",
            "label": test_value_for_type,
        },
    )
    assert response_with_label.status_code == 200
    data_from_label_input = response_with_label.json()

    # 2. Create metamessage using "metamessage_type" (alias)
    response_with_alias = client.post(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/metamessages",
        json={
            **common_payload_parts,
            "content": "Content created with alias",
            "metamessage_type": test_value_for_type,  # Using alias
        },
    )
    assert response_with_alias.status_code == 200
    data_from_alias_input = response_with_alias.json()

    # Assertions for the first response (created with "label")
    assert data_from_label_input["content"] == "Content created with label"
    assert data_from_label_input["label"] == test_value_for_type
    assert data_from_label_input["metamessage_type"] == test_value_for_type
    assert data_from_label_input["metadata"] == common_payload_parts["metadata"]
    assert data_from_label_input["metamessage_type"] == test_value_for_type
    assert data_from_label_input["app_id"] == str(test_app.public_id)
    assert data_from_label_input["user_id"] == str(test_user.public_id)
    assert data_from_label_input["session_id"] == common_payload_parts["session_id"]
    assert data_from_label_input["message_id"] == common_payload_parts["message_id"]

    # Assertions for the second response (created with "metamessage_type")
    assert data_from_alias_input["content"] == "Content created with alias"
    assert (
        data_from_alias_input["label"] == test_value_for_type
    )  # Output should still be "label"
    assert data_from_alias_input["metamessage_type"] == test_value_for_type
    assert data_from_alias_input["metadata"] == common_payload_parts["metadata"]
    assert data_from_alias_input["metamessage_type"] == test_value_for_type
    assert data_from_alias_input["user_id"] == str(test_user.public_id)
    assert data_from_alias_input["session_id"] == common_payload_parts["session_id"]
    assert data_from_alias_input["message_id"] == common_payload_parts["message_id"]
    assert data_from_alias_input["app_id"] == str(test_app.public_id)
    # Key assertion: The output for the type/label field is consistent ("label") and has the correct value
    assert data_from_label_input["label"] == data_from_alias_input["label"]
    assert data_from_label_input["metamessage_type"] == data_from_alias_input["metamessage_type"]
    assert data_from_label_input["label"] == data_from_label_input["metamessage_type"]
    assert data_from_alias_input["label"] == data_from_alias_input["metamessage_type"]
