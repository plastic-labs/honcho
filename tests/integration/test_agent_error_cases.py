import json

import pytest
import respx
from anthropic import APIError as AnthropicAPIError
from anthropic import RateLimitError as AnthropicRateLimitError
from httpx import HTTPStatusError, Response, TimeoutException
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from src import agent, models, schemas
from src.deriver.tom.embeddings import CollectionEmbeddingStore


@pytest.mark.asyncio(loop_scope="session")
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
    
    # Mock the embedding store to avoid DB dependency during error testing
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Test fact 1", "Test fact 2"]
    
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # Test that the error is properly handled and propagated
        with pytest.raises((APIError, APIConnectionError, AnthropicAPIError)) as exc_info:
            await agent.chat(
                app_id=test_app.public_id,
                user_id=test_user.public_id,
                session_id=test_session.public_id,
                queries="Hello, how are you?",
                db=db_session,
                stream=False
            )
        
        # Verify the error contains our mock error message
        assert "Mock API error" in str(exc_info.value) or "500" in str(exc_info.value)
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts


@pytest.mark.asyncio(loop_scope="session")
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
    
    # Mock the embedding store to avoid DB dependency during timeout testing
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Test fact 1", "Test fact 2"]
    
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # Test that the timeout is properly handled and propagated
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
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts


@pytest.mark.asyncio(loop_scope="session")
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
    
    # Mock the embedding store
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Custom fact 1", "Custom fact 2"]
    
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
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
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts


@pytest.mark.asyncio(loop_scope="session")
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

    # Mock the embedding store
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Verification fact 1", "Verification fact 2"]
    
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # Make a request and verify its content
        await agent.chat(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,
            queries="Hello, how are you?",
            db=db_session,
            stream=False
        )
        
        # Verify that requests were made to the expected endpoints
        assert len(respx.calls) > 0, "No HTTP requests were made"
        
        # Verify the request structure
        for call in respx.calls:
            request_body = json.loads(call.request.content)
            # Each request should have proper structure
            assert "messages" in request_body or "system" in request_body, "Request missing required fields"
            
        print(f"\nVerified {len(respx.calls)} HTTP requests with proper structure")
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts


@pytest.mark.asyncio(loop_scope="session")
@respx.mock
async def test_chat_rate_limit_error(db_session, sample_data, mock_model_client):
    """Test chat functionality when the model API returns rate limit errors"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()

    # Set up mock to return rate limit error
    def rate_limit_side_effect(request):
        return Response(429, json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}})
    
    mock_model_client.anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=rate_limit_side_effect
    )
    mock_model_client.groq_route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=rate_limit_side_effect
    )
    
    # Mock the embedding store
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Rate limit test fact"]
    
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # Test that rate limit errors are properly handled
        with pytest.raises((HTTPStatusError, RateLimitError, AnthropicRateLimitError)) as exc_info:
            await agent.chat(
                app_id=test_app.public_id,
                user_id=test_user.public_id,
                session_id=test_session.public_id,
                queries="Hello, how are you?",
                db=db_session,
                stream=False
            )
        
        # Verify it's a rate limit error
        error_msg = str(exc_info.value).lower()
        assert any(msg in error_msg for msg in ["rate limit", "429", "too many requests"]), f"Expected rate limit error, got: {error_msg}"
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts


@pytest.mark.asyncio(loop_scope="session")
@respx.mock
async def test_chat_partial_failure_resilience(db_session, sample_data, mock_model_client):
    """Test chat functionality when some LLM calls fail but others succeed"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()

    # Make query generation fail but other calls succeed
    call_count = [0]
    
    def selective_failure(request):
        call_count[0] += 1
        request_body = json.loads(request.content)
        
        # Check if this is a query generation request
        system_content = ""
        if "system" in request_body:
            system_content = request_body["system"]
        elif "messages" in request_body:
            for msg in request_body["messages"]:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                    break
        
        # Fail query generation requests to test fallback behavior
        if "generate 3 focused search queries" in system_content:
            return Response(500, json={"error": {"message": "Query generation failed"}})
        
        # Success for other requests
        if "anthropic.com" in str(request.url):
            return Response(200, json={
                "content": [{"type": "text", "text": mock_model_client.default_response}]
            })
        else:  # OpenAI-compatible
            return Response(200, json={
                "choices": [{"message": {"content": mock_model_client.tom_response}}]
            })
    
    mock_model_client.anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=selective_failure
    )
    mock_model_client.groq_route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=selective_failure
    )
    
    # Mock the embedding store
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Resilience test fact"]
    
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # This should fail because query generation is required
        with pytest.raises(Exception) as exc_info:
            await agent.chat(
                app_id=test_app.public_id,
                user_id=test_user.public_id,
                session_id=test_session.public_id,
                queries="Hello, how are you?",
                db=db_session,
                stream=False
            )
        
        # Verify that we got an error from the failing component
        assert "Query generation failed" in str(exc_info.value) or "500" in str(exc_info.value)
        print("\nPartial failure test passed - query generation failure was properly handled")
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts


@pytest.mark.asyncio(loop_scope="session") 
@respx.mock
async def test_chat_slow_response_handling(db_session, sample_data, mock_model_client):
    """Test chat functionality with slow API responses"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()

    import asyncio
    
    async def slow_response(request):
        # Simulate slow response (but not timeout)
        await asyncio.sleep(0.1)  # 100ms delay
        
        if "anthropic.com" in str(request.url):
            return Response(200, json={
                "content": [{"type": "text", "text": mock_model_client.default_response}]
            })
        else:  # OpenAI-compatible
            if "generate 3 focused search queries" in str(request.content):
                return Response(200, json={
                    "choices": [{"message": {"content": json.dumps(mock_model_client.query_response)}}]
                })
            else:
                return Response(200, json={
                    "choices": [{"message": {"content": mock_model_client.tom_response}}]
                })
    
    mock_model_client.anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=slow_response
    )
    mock_model_client.groq_route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=slow_response
    )
    
    # Mock the embedding store
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Slow response test fact"]
    
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # This should succeed despite slow responses
        response = await agent.chat(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,
            queries="Hello, how are you?",
            db=db_session,
            stream=False
        )
        
        assert isinstance(response, schemas.DialecticResponse)
        assert response.content  # Should have content
        print(f"\nSlow response test passed - handled {len(respx.calls)} slow API calls successfully")
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts
