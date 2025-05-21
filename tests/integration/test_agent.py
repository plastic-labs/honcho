import json

import pytest
import respx

from src import agent, models, schemas
from src.deriver.tom.single_prompt import TOM_SYSTEM_PROMPT


@pytest.mark.asyncio(loop_scope="session")
@respx.mock
async def test_chat_basic_functionality(db_session, sample_data, mock_model_client):
    """Test basic chat functionality with all APIs working correctly"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()
    
    # Set up mock responses
    mock_model_client.default_response = "I'm doing well, thank you for asking!"
    mock_model_client.update_routes()
    
    # Mock the embedding store
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["User likes programming", "User is friendly"]
    
    from src.deriver.tom.embeddings import CollectionEmbeddingStore
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # Test non-streaming chat
        response = await agent.chat(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,
            queries="Hello, how are you?",
            db=db_session,
            stream=False
        )

        # Validate response structure and content
        assert isinstance(response, schemas.DialecticResponse)
        assert "I'm doing well" in response.content
        
        # Verify the proper sequence of LLM calls was made
        assert len(respx.calls) >= 2, f"Expected at least 2 LLM calls, got {len(respx.calls)}"
        
        # Check that different providers were used for different purposes
        anthropic_calls = sum(1 for call in respx.calls if "anthropic.com" in str(call.request.url))
        groq_calls = sum(1 for call in respx.calls if "groq.com" in str(call.request.url))
        
        assert anthropic_calls > 0, "Expected Anthropic API calls for Dialectic"
        assert groq_calls > 0, "Expected Groq API calls for ToM/Query Generation"
        
        print(f"\nBasic functionality test: {anthropic_calls} Anthropic calls, {groq_calls} Groq calls")
        
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts


@pytest.mark.asyncio(loop_scope="session")
@respx.mock
async def test_chat_with_mock_responses(db_session, sample_data, mock_model_client):
    """Test chat functionality using the mock responses"""
    test_app, test_user = sample_data
    
    # Create a test session in the database
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()
    
    # Create some test messages
    test_messages = [
        models.Message(
            session_id=test_session.public_id,
            content="Hello!",
            is_user=True,
            app_id=test_app.public_id,
            user_id=test_user.public_id
        ),
        models.Message(
            session_id=test_session.public_id,
            content="Hi there!",
            is_user=False,
            app_id=test_app.public_id,
            user_id=test_user.public_id
        )
    ]
    for msg in test_messages:
        db_session.add(msg)
    await db_session.commit()
    
    # Set up mock responses
    mock_model_client.tom_response = "<prediction>The user appears to be initiating a friendly conversation with a polite greeting.</prediction>"
    mock_model_client.query_response = ["query about interests", "query about personality", "query about experiences"]
    mock_model_client.default_response = "I'm doing well, thank you for asking!"
    mock_model_client.update_routes()
    
    # Mock the embedding store's get_relevant_facts method
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Test fact 1", "Test fact 2"]
    
    from src.deriver.tom.embeddings import CollectionEmbeddingStore
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # Test non-streaming chat
        response = await agent.chat(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,  # Use the actual session ID
            queries="Hello, how are you?",
            db=db_session,
            stream=False
        )

        assert isinstance(response, schemas.DialecticResponse)
        print(f"\nTest response: {response.content}")
        print(f"Anthropic route called: {mock_model_client.anthropic_route.called}")
        print(f"OpenAI route called: {mock_model_client.openai_route.called}")
        print(f"Groq route called: {mock_model_client.groq_route.called}")
        assert "I'm doing well" in response.content

        # Verify that the appropriate API endpoints were called
        assert mock_model_client.anthropic_route.called or mock_model_client.openai_route.called

        # Get the last request to verify ToM system prompt was used
        last_request = respx.calls.last.request
        assert last_request is not None
        
        # Verify that the appropriate system prompts were used
        tom_request_made = False
        dialectic_request_made = False
        
        for call in respx.calls:
            request_body = json.loads(call.request.content)
            
            # Check for Anthropic-style system prompts
            if "system" in request_body and isinstance(request_body["system"], str):
                system_content = request_body["system"]
                if system_content == TOM_SYSTEM_PROMPT:
                    tom_request_made = True
                elif "context service" in system_content:  # Dialectic system prompt
                    dialectic_request_made = True
            
            # Check for OpenAI-style system prompts in messages
            elif "messages" in request_body:
                for message in request_body["messages"]:
                    if message.get("role") == "system":
                        system_content = message.get("content", "")
                        if system_content == TOM_SYSTEM_PROMPT:
                            tom_request_made = True
                        elif "context service" in system_content:
                            dialectic_request_made = True
        
        # Verify that at least ToM and Dialectic requests were made (core agent functionality)
        assert tom_request_made, "No request was made with the ToM system prompt"
        assert dialectic_request_made, "No request was made with the Dialectic system prompt"

    finally:
        # Restore the original method
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts
        
        # Print debugging info about requests made
        print(f"\nRequests made: {len(respx.calls)}")
        for i, call in enumerate(respx.calls):
            request_body = json.loads(call.request.content)
            system_prompt = ""
            if "system" in request_body:
                system_prompt = request_body["system"][:50] + "..." if len(str(request_body["system"])) > 50 else str(request_body["system"])
            elif "messages" in request_body:
                for msg in request_body["messages"]:
                    if msg.get("role") == "system":
                        system_prompt = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                        break
            print(f"Request {i+1}: {call.request.url} - System: {system_prompt}")


@pytest.mark.asyncio(loop_scope="session")
@respx.mock
async def test_chat_streaming_functionality(db_session, sample_data, mock_model_client):
    """Test streaming chat functionality"""
    test_app, test_user = sample_data
    
    # Create a test session
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={}
    )
    db_session.add(test_session)
    await db_session.commit()
    
    # Set up mock responses for streaming
    mock_model_client.default_response = "Streaming response content"
    mock_model_client.tom_response = "<prediction>User is testing streaming functionality</prediction>"
    mock_model_client.query_response = ["streaming query 1", "streaming query 2"]
    mock_model_client.update_routes()
    
    # Mock the embedding store
    async def mock_get_relevant_facts(*args, **kwargs):
        return ["Streaming test fact 1", "Streaming test fact 2"]
    
    from src.deriver.tom.embeddings import CollectionEmbeddingStore
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # Test streaming chat
        response_stream = await agent.chat(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,
            queries="Test streaming response",
            db=db_session,
            stream=True
        )

        # Verify the response is a stream manager (could be different types depending on Anthropic version)
        from anthropic import MessageStreamManager
        from anthropic.lib.streaming._messages import AsyncMessageStreamManager
        assert isinstance(response_stream, (MessageStreamManager, AsyncMessageStreamManager)), f"Expected MessageStreamManager or AsyncMessageStreamManager, got {type(response_stream)}"
        
        # Verify that LLM calls were made for the streaming setup
        assert len(respx.calls) >= 2, f"Expected at least 2 LLM calls for streaming setup, got {len(respx.calls)}"
        
        print(f"\nStreaming functionality test: Set up {len(respx.calls)} LLM calls successfully")
        
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts


@pytest.mark.asyncio(loop_scope="session")
@respx.mock
async def test_chat_workflow_end_to_end(db_session, sample_data, mock_model_client):
    """Test the complete agent workflow with detailed validation"""
    test_app, test_user = sample_data
    
    # Create a test session with some message history
    test_session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        h_metadata={"test": "workflow"}
    )
    db_session.add(test_session)
    await db_session.commit()
    
    # Add some message history to test ToM inference
    test_messages = [
        models.Message(
            session_id=test_session.public_id,
            content="Hi, I'm working on a Python project",
            is_user=True,
            app_id=test_app.public_id,
            user_id=test_user.public_id
        ),
        models.Message(
            session_id=test_session.public_id,
            content="That sounds interesting! What kind of project?",
            is_user=False,
            app_id=test_app.public_id,
            user_id=test_user.public_id
        ),
        models.Message(
            session_id=test_session.public_id,
            content="It's an AI agent for memory management",
            is_user=True,
            app_id=test_app.public_id,
            user_id=test_user.public_id
        )
    ]
    for msg in test_messages:
        db_session.add(msg)
    await db_session.commit()
    
    # Set up realistic mock responses
    mock_model_client.query_response = [
        "user's programming experience",
        "user's AI interests", 
        "user's project goals"
    ]
    mock_model_client.tom_response = "<prediction>CURRENT STATE: User is actively developing an AI project and seeking assistance. ACTIVE GOALS: Building memory management functionality. PRESENT MOOD: Engaged and focused on technical work.</prediction>"
    mock_model_client.default_response = "Based on your work with AI agents and memory management, I can help you think through the architecture. What specific aspect of memory management are you focusing on?"
    mock_model_client.update_routes()
    
    # Mock the embedding store with realistic facts
    async def mock_get_relevant_facts(self, query, **kwargs):
        # Return facts relevant to the query
        if "programming" in query.lower():
            return ["User is experienced with Python", "User has worked on AI projects before"]
        elif "ai" in query.lower():
            return ["User is interested in artificial intelligence", "User builds AI applications"]
        else:
            return ["User is a software developer", "User likes technical challenges"]
    
    from src.deriver.tom.embeddings import CollectionEmbeddingStore
    original_get_relevant_facts = CollectionEmbeddingStore.get_relevant_facts
    CollectionEmbeddingStore.get_relevant_facts = mock_get_relevant_facts
    
    try:
        # Execute the full workflow
        response = await agent.chat(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            session_id=test_session.public_id,
            queries="How should I structure the memory system?",
            db=db_session,
            stream=False
        )

        # Validate the complete workflow
        assert isinstance(response, schemas.DialecticResponse)
        assert "memory management" in response.content.lower(), "Response should be contextually relevant"
        
        # Verify the complete LLM call sequence
        assert len(respx.calls) == 3, f"Expected exactly 3 LLM calls (query gen, ToM, dialectic), got {len(respx.calls)}"
        
        # Verify each type of call was made
        query_gen_calls = 0
        tom_calls = 0 
        dialectic_calls = 0
        
        for call in respx.calls:
            request_body = json.loads(call.request.content)
            system_content = ""
            
            if "system" in request_body:
                system_content = request_body["system"]
            elif "messages" in request_body:
                for msg in request_body["messages"]:
                    if msg.get("role") == "system":
                        system_content = msg.get("content", "")
                        break
            
            if "generate 3 focused search queries" in system_content:
                query_gen_calls += 1
            elif "analyzing conversations to make evidence-based inferences" in system_content:
                tom_calls += 1
            elif "context service" in system_content:
                dialectic_calls += 1
        
        assert query_gen_calls == 1, f"Expected 1 query generation call, got {query_gen_calls}"
        assert tom_calls == 1, f"Expected 1 ToM inference call, got {tom_calls}"
        assert dialectic_calls == 1, f"Expected 1 dialectic call, got {dialectic_calls}"
        
        print("\nEnd-to-end workflow validation passed:")
        print(f"  - Query generation: {query_gen_calls} call")
        print(f"  - ToM inference: {tom_calls} call")
        print(f"  - Dialectic response: {dialectic_calls} call")
        print("  - Final response relevant to context: ✓")
        
    finally:
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts