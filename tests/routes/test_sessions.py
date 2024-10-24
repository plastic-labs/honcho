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
    test_session = models.Session(user_id=test_user.public_id, metadata={})
    db_session.add(test_session)
    await db_session.commit()

    response = client.put(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}",
        json={},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_update_delete_metadata(client, db_session, sample_data):
    test_app, test_user = sample_data
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id, metadata={"default": "value"}
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
    test_session = models.Session(user_id=test_user.public_id, metadata={})
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
    test_session = models.Session(user_id=test_user.public_id, metadata={})
    db_session.add(test_session)
    await db_session.commit()
    response = client.delete(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}"
    )
    assert response.status_code == 200
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}"
    )
    data = response.json()
    assert data["is_active"] is False
