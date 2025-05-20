import pytest
import respx
import json

from src import agent, schemas, models
from src.deriver.tom.single_prompt import TOM_SYSTEM_PROMPT
from src.agent import QUERY_GENERATION_SYSTEM


@pytest.mark.asyncio
@respx.mock
async def test_chat_basic_functionality(db_session, sample_data, mock_model_client):
    """Test basic chat functionality with all APIs working correctly"""
    test_app, test_user = sample_data
    
    # Set up mock responses
    mock_model_client.default_response = "I'm doing well, thank you for asking!"
    mock_model_client.update_routes()
    
    # Test non-streaming chat
    response = await agent.chat(
        app_id=test_app.public_id,
        user_id=test_user.public_id,
        session_id="test_session",
        queries="Hello, how are you?",
        db=db_session,
        stream=False
    )

    assert isinstance(response, schemas.DialecticResponse)
    assert "I'm doing well" in response.content


@pytest.mark.asyncio
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
        print(f"test 1 response: {response.content}")
        assert "I'm doing well" in response.content

        # Verify that the appropriate API endpoints were called
        assert mock_model_client.anthropic_route.called or mock_model_client.openai_route.called

        # Get the last request to verify ToM system prompt was used
        last_request = respx.calls.last.request
        assert last_request is not None
        
        # Verify at least one request contained the ToM system prompt
        tom_request_made = False
        for call in respx.calls:
            request_body = json.loads(call.request.content)
            if "messages" in request_body:
                for message in request_body["messages"]:
                    if message.get("role") == "system" and message.get("content") == TOM_SYSTEM_PROMPT:
                        tom_request_made = True
                        break
            if tom_request_made:
                break
        assert tom_request_made, "No request was made with the ToM system prompt"

    finally:
        # Restore the original method
        CollectionEmbeddingStore.get_relevant_facts = original_get_relevant_facts