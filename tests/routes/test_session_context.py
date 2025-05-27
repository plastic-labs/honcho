import pytest

from src import models
from src.utils.history import SummaryType


@pytest.mark.asyncio
async def test_get_session_context_no_summary(client, db_session, sample_data):
    """Test getting session context when no summary exists"""
    test_app, test_user = sample_data
    
    test_session = models.Session(
        user_id=test_user.public_id, h_metadata={}, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.flush()
    
    messages = []
    for i in range(5):
        message = models.Message(
            session_id=test_session.public_id,
            content=f"Test message {i}",
            is_user=i % 2 == 0,  # Alternate between user and assistant
            h_metadata={},
            app_id=test_app.public_id,
            user_id=test_user.public_id,
        )
        db_session.add(message)
        messages.append(message)
    
    await db_session.commit()
    
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/context"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["summary"] is None
    assert len(data["messages"]) == 5
    
    for i, message in enumerate(data["messages"]):
        assert message["content"] == f"Test message {i}"
        assert message["is_user"] == (i % 2 == 0)


@pytest.mark.asyncio
async def test_get_session_context_with_summary(client, db_session, sample_data):
    """Test getting session context with an existing summary"""
    test_app, test_user = sample_data
    
    test_session = models.Session(
        user_id=test_user.public_id, h_metadata={}, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.flush()
    
    initial_messages = []
    for i in range(3):
        message = models.Message(
            session_id=test_session.public_id,
            content=f"Initial message {i}",
            is_user=i % 2 == 0,
            h_metadata={},
            app_id=test_app.public_id,
            user_id=test_user.public_id,
        )
        db_session.add(message)
        initial_messages.append(message)
    
    await db_session.flush()
    
    summary = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=initial_messages[-1].public_id,  # Link to the last message
        content="This is a test summary of the conversation",
        h_metadata={"message_count": 3, "summary_type": "SHORT"},
        label=SummaryType.SHORT.value,
        app_id=test_app.public_id,
    )
    db_session.add(summary)
    await db_session.flush()
    
    after_summary_messages = []
    for i in range(3):
        message = models.Message(
            session_id=test_session.public_id,
            content=f"After summary message {i}",
            is_user=i % 2 == 0,
            h_metadata={},
            app_id=test_app.public_id,
            user_id=test_user.public_id,
        )
        db_session.add(message)
        after_summary_messages.append(message)
    
    await db_session.commit()
    
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/context"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["summary"] is not None
    assert data["summary"]["content"] == "This is a test summary of the conversation"
    assert len(data["messages"]) == 3
    
    for i, message in enumerate(data["messages"]):
        assert message["content"] == f"After summary message {i}"


@pytest.mark.asyncio
async def test_get_session_context_with_count_param(client, db_session, sample_data):
    """Test getting session context with count parameter"""
    test_app, test_user = sample_data
    
    test_session = models.Session(
        user_id=test_user.public_id, h_metadata={}, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.flush()
    
    messages = []
    for i in range(10):
        message = models.Message(
            session_id=test_session.public_id,
            content=f"Test message {i}",
            is_user=i % 2 == 0,
            h_metadata={},
            app_id=test_app.public_id,
            user_id=test_user.public_id,
        )
        db_session.add(message)
        messages.append(message)
    
    await db_session.commit()
    
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/context?count=3"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["summary"] is None
    assert len(data["messages"]) == 3
    
    for i, message in enumerate(data["messages"]):
        assert message["content"] == f"Test message {i}"


@pytest.mark.asyncio
async def test_get_session_context_with_summary_type_param(client, db_session, sample_data):
    """Test getting session context with summary_type parameter"""
    test_app, test_user = sample_data
    
    test_session = models.Session(
        user_id=test_user.public_id, h_metadata={}, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.flush()
    
    initial_messages = []
    for i in range(3):
        message = models.Message(
            session_id=test_session.public_id,
            content=f"Initial message {i}",
            is_user=i % 2 == 0,
            h_metadata={},
            app_id=test_app.public_id,
            user_id=test_user.public_id,
        )
        db_session.add(message)
        initial_messages.append(message)
    
    await db_session.flush()
    
    short_summary = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=initial_messages[-1].public_id,
        content="This is a SHORT summary",
        h_metadata={"message_count": 3, "summary_type": "SHORT"},
        label=SummaryType.SHORT.value,
        app_id=test_app.public_id,
    )
    db_session.add(short_summary)
    
    long_summary = models.Metamessage(
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        message_id=initial_messages[-1].public_id,
        content="This is a LONG summary with more details",
        h_metadata={"message_count": 3, "summary_type": "LONG"},
        label=SummaryType.LONG.value,
        app_id=test_app.public_id,
    )
    db_session.add(long_summary)
    await db_session.flush()
    
    after_summary_messages = []
    for i in range(3):
        message = models.Message(
            session_id=test_session.public_id,
            content=f"After summary message {i}",
            is_user=i % 2 == 0,
            h_metadata={},
            app_id=test_app.public_id,
            user_id=test_user.public_id,
        )
        db_session.add(message)
        after_summary_messages.append(message)
    
    await db_session.commit()
    
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/context"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["content"] == "This is a SHORT summary"
    
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/context?summary_type=long"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["content"] == "This is a LONG summary with more details"


@pytest.mark.asyncio
async def test_get_session_context_max_messages(client, db_session, sample_data):
    """Test getting session context with more than 60 messages"""
    test_app, test_user = sample_data
    
    test_session = models.Session(
        user_id=test_user.public_id, h_metadata={}, app_id=test_app.public_id
    )
    db_session.add(test_session)
    await db_session.flush()
    
    messages = []
    for i in range(70):
        message = models.Message(
            session_id=test_session.public_id,
            content=f"Test message {i}",
            is_user=i % 2 == 0,
            h_metadata={},
            app_id=test_app.public_id,
            user_id=test_user.public_id,
        )
        db_session.add(message)
        messages.append(message)
    
    await db_session.commit()
    
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/{test_session.public_id}/context"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["summary"] is None
    assert len(data["messages"]) == 60  # Should be capped at 60
    
    for i, message in enumerate(data["messages"]):
        assert message["content"] == f"Test message {i}"


@pytest.mark.asyncio
async def test_get_session_context_not_found(client, db_session, sample_data):
    """Test getting session context for a non-existent session"""
    test_app, test_user = sample_data
    
    response = client.get(
        f"/v1/apps/{test_app.public_id}/users/{test_user.public_id}/sessions/non_existent_session/context"
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
