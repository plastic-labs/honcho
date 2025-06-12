import pytest
from nanoid import generate as generate_nanoid

from src import models  # Import your SQLAlchemy models


@pytest.mark.asyncio
async def test_create_message(client, db_session, sample_data):
    test_workspace, test_peer = sample_data
    
    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name,
        name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages",
        json={
            "messages": [{
                "content": "Test message",
                "peer_id": test_peer.name,
                "metadata": {"message_key": "message_value"},
            }]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    message = data[0]
    assert message["content"] == "Test message"
    assert message["peer_name"] == test_peer.name
    assert message["metadata"] == {"message_key": "message_value"}
    assert "id" in message


@pytest.mark.asyncio
async def test_get_messages(client, db_session, sample_data):
    test_workspace, test_peer = sample_data
    
    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name,
        name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()
    
    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["content"] == "Test message"
    assert data["items"][0]["peer_name"] == test_peer.name
    assert data["items"][0]["metadata"] == {}


@pytest.mark.asyncio
async def test_get_filtered_messages(client, db_session, sample_data):
    test_workspace, test_peer = sample_data
    
    # Create a test session and messages
    test_session = models.Session(
        workspace_name=test_workspace.name,
        name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()
    
    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"key": "value"},
    )
    test_message2 = models.Message(
        session_name=test_session.name,
        content="Test message 2",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
        h_metadata={"key": "value2"},
    )
    db_session.add(test_message)
    db_session.add(test_message2)
    await db_session.commit()

    response = client.post(
        f"/v1/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/list",
        json={"filter": {"key": "value2"}},
    )
    assert response.status_code == 200
    data = response.json()
    print(data)
    assert "items" in data
    assert len(data["items"]) > 0
    assert data["items"][0]["content"] == "Test message 2"
    assert data["items"][0]["peer_name"] == test_peer.name
    assert data["items"][0]["metadata"] == {"key": "value2"}


@pytest.mark.asyncio
async def test_update_message(client, db_session, sample_data):
    test_workspace, test_peer = sample_data
    
    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name,
        name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()
    
    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}",
        json={"metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"new_key": "new_value"}


@pytest.mark.asyncio
async def test_update_message_empty_metadata(client, db_session, sample_data):
    test_workspace, test_peer = sample_data
    
    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name,
        name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()
    
    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.put(
        f"/v1/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}",
        json={"metadata": None},
    )
    assert response.status_code == 422
    data = response.json()
    print(data)


@pytest.mark.asyncio
async def test_get_single_message(client, db_session, sample_data):
    test_workspace, test_peer = sample_data
    
    # Create a test session and message
    test_session = models.Session(
        workspace_name=test_workspace.name,
        name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()
    
    test_message = models.Message(
        session_name=test_session.name,
        content="Test message",
        workspace_name=test_workspace.name,
        peer_name=test_peer.name
    )
    db_session.add(test_message)
    await db_session.commit()

    response = client.get(
        f"/v1/workspaces/{test_workspace.name}/sessions/{test_session.name}/messages/{test_message.public_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Test message"
    assert data["peer_name"] == test_peer.name
    assert data["id"] == test_message.public_id