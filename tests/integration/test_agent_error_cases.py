import pytest
import respx
import json
from httpx import TimeoutException
from openai import APIConnectionError, APIError, APITimeoutError

from src import agent, schemas, models
from src.deriver.tom.single_prompt import TOM_SYSTEM_PROMPT


@pytest.mark.asyncio
@respx.mock
async def test_chat_api_error(db_session, sample_data, mock_model_client):
    """Test chat functionality when the model API returns an error"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()

    # Enable error mode in the mock
    mock_model_client.error_mode = True
    mock_model_client.update_routes()
    
    # Test that the error is properly handled
    with pytest.raises((APIError, APIConnectionError)) as exc_info:
        await agent.chat(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,
            queries="Hello, how are you?",
            db=db_session,
            stream=False
        )
    
    assert "Mock API error" in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
async def test_chat_timeout(db_session, sample_data, mock_model_client):
    """Test chat functionality when the model API times out"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()

    # Enable timeout mode in the mock
    mock_model_client.timeout_mode = True
    mock_model_client.update_routes()
    
    # Test that the timeout is properly handled
    with pytest.raises((TimeoutException, APITimeoutError)) as exc_info:
        await agent.chat(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,
            queries="Hello, how are you?",
            db=db_session,
            stream=False
        )
    
    error_msg = str(exc_info.value).lower()
    assert any(msg in error_msg for msg in ["timeout", "timed out"]), f"Expected timeout error, got: {error_msg}"


@pytest.mark.asyncio
@respx.mock
async def test_chat_with_custom_response(db_session, sample_data, mock_model_client):
    """Test chat functionality with custom mock responses"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()

    # Set custom responses
    mock_model_client.tom_response = "<prediction>Custom ToM response for testing</prediction>"
    mock_model_client.query_response = ["custom query 1", "custom query 2"]
    mock_model_client.default_response = "Custom default response"
    mock_model_client.update_routes()
    
    # Test that custom responses are used
    response = await agent.chat(
        app_id=test_app.public_id,
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        queries="Hello, how are you?",
        db=db_session,
        stream=False
    )

    assert isinstance(response, schemas.DialecticResponse)
    assert "Custom default response" in response.content


@pytest.mark.asyncio
@respx.mock
async def test_verify_request_content(db_session, sample_data, mock_model_client):
    """Test that the requests are properly formatted"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()

    mock_model_client.update_routes()

    # Make a request and verify its content
    await agent.chat(
        app_id=test_app.public_id,
        user_id=test_user.public_id,
        session_id=test_session.public_id,
        queries="Hello, how are you?",
        db=db_session,
        stream=False
    )
