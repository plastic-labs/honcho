from src import models  # Import your SQLAlchemy models
from sqlalchemy.ext.asyncio import AsyncSession


def test_create_app(client):
    response = client.post(
        "/apps", json={"name": "New App", "metadata": {"key": "value"}}
    )
    print(response)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "New App"
    assert data["metadata"] == {"key": "value"}
    assert "id" in data


def test_get_app(client, db_session, test_data):
    test_app, _ = test_data
    response = client.get(f"/apps/{test_app.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test App"
    assert data["id"] == str(test_app.id)


def test_update_app(client, db_session, test_data):
    test_app, _ = test_data
    response = client.put(
        f"/apps/{test_app.id}",
        json={"name": "Updated App", "metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated App"
    assert data["metadata"] == {"new_key": "new_value"}


def test_create_user(client, db_session, test_data):
    test_app, _ = test_data
    response = client.post(
        f"/apps/{test_app.id}/users",
        json={"name": "New User", "metadata": {"user_key": "user_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "New User"
    assert data["metadata"] == {"user_key": "user_value"}
    assert "id" in data


def test_get_user(client, db_session, test_data):
    test_app, test_user = test_data
    response = client.get(f"/apps/{test_app.id}/users/{test_user.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test User"
    assert data["id"] == str(test_user.id)


def test_create_session(client, db_session, test_data):
    test_app, test_user = test_data
    response = client.post(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions",
        json={
            "location_id": "test_location",
            "metadata": {"session_key": "session_value"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["location_id"] == "test_location"
    assert data["metadata"] == {"session_key": "session_value"}
    assert "id" in data


def test_get_sessions(client, db_session, test_data):
    test_app, test_user = test_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.id, location_id="test_location", metadata={}
    )
    db_session.add(test_session)
    db_session.commit()

    response = client.get(f"/apps/{test_app.id}/users/{test_user.id}/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["location_id"] == "test_location"


def test_create_message(client, db_session, test_data):
    test_app, test_user = test_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.id, location_id="test_location", metadata={}
    )
    db_session.add(test_session)
    db_session.commit()

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
    assert data["is_user"] == True
    assert data["metadata"] == {"message_key": "message_value"}
    assert "id" in data


def test_get_messages(client, db_session, test_data):
    test_app, test_user = test_data
    # Create a test session and message
    test_session = models.Session(
        user_id=test_user.id, location_id="test_location", metadata={}
    )
    db_session.add(test_session)
    db_session.commit()
    test_message = models.Message(
        session_id=test_session.id, content="Test message", is_user=True, metadata={}
    )
    db_session.add(test_message)
    db_session.commit()

    response = client.get(
        f"/apps/{test_app.id}/users/{test_user.id}/sessions/{test_session.id}/messages"
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["content"] == "Test message"


# Add more test cases for other endpoints as needed
