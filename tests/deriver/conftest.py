"""Fixtures and test configuration for deriver tests."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from nanoid import generate as generate_nanoid

from src import models
from src.deriver.queue import QueueManager
from src.deriver.tom.embeddings import CollectionEmbeddingStore
from .test_config import test_config, conditional_mock_llm, with_retry_and_rate_limit

# Set fake API keys for testing if real ones aren't available
if not test_config.has_groq_api_key:
    os.environ["GROQ_API_KEY"] = "fake-groq-key-for-testing"
if not test_config.has_anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = "fake-anthropic-key-for-testing"


@pytest.fixture
def mock_llm_responses():
    """Provides structured mock responses for different LLM operations."""
    return {
        "fact_extraction": json.dumps({
            "facts": [
                "User is a software developer",
                "User works remotely", 
                "User prefers coffee over tea",
                "User uses Python and JavaScript"
            ]
        }),
        "fact_extraction_empty": json.dumps({"facts": []}),
        "fact_extraction_malformed": "This is not valid JSON",
        "tom_single_prompt": json.dumps({
            "confidence": 0.8,
            "user_representation": {
                "personality_traits": ["analytical", "detail-oriented", "collaborative"],
                "preferences": ["remote work", "technical discussions"],
                "communication_style": "direct and concise",
                "expertise_areas": ["software development", "Python programming"]
            }
        }),
        "tom_conversational": "Based on our conversation, I believe the user is a thoughtful software developer who values clear communication and technical excellence.",
        "summary_short": "User discussed Python development and remote work preferences.",
        "summary_long": "User is a software developer working remotely who has shown expertise in Python development and expressed preferences for asynchronous communication and detailed technical discussions."
    }


@pytest.fixture
def mock_embeddings():
    """Provides mock embedding vectors for testing."""
    return {
        "fact_embedding": [0.1, 0.2, 0.3] + [0.0] * 1533,  # 1536-dim vector
        "query_embedding": [0.15, 0.25, 0.35] + [0.0] * 1533,
        "duplicate_embedding": [0.1, 0.2, 0.3] + [0.0] * 1533,  # Identical to fact_embedding
        "different_embedding": [0.9, 0.8, 0.7] + [0.0] * 1533
    }


@pytest.fixture(autouse=True)
def mock_llm_calls(mock_llm_responses):
    """Mock LLM calls for all TOM methods - conditionally use real APIs if available."""
    
    # Only mock if not using real APIs
    if not test_config.use_real_apis:
        with (
            patch("src.deriver.tom.single_prompt.tom_inference") as mock_tom_inference,
            patch("src.deriver.tom.single_prompt.user_representation_inference") as mock_user_rep_inference,
            patch("src.deriver.consumer.extract_facts_long_term") as mock_extract_facts_consumer,
            patch("src.deriver.tom.long_term.extract_facts_long_term") as mock_extract_facts,
            patch("src.deriver.tom.long_term.get_user_representation_long_term") as mock_long_term_user_rep,
            patch("src.deriver.tom.conversational.anthropic") as mock_anthropic,
            # Mock HTTP clients to prevent real API calls
            patch("httpx.AsyncClient.send") as mock_httpx_send,
        ):
            # Mock Mirascope single prompt functions
            mock_tom_response = MagicMock()
            mock_tom_response.model_dump_json.return_value = mock_llm_responses['tom_single_prompt']
            mock_tom_inference.return_value = mock_tom_response

            mock_user_rep_response = MagicMock()
            mock_user_rep_response.model_dump_json.return_value = mock_llm_responses['tom_single_prompt']
            mock_user_rep_inference.return_value = mock_user_rep_response

            # Mock long term functions
            mock_fact_extraction_response = AsyncMock()
            mock_fact_extraction_response.facts = ["User is a software developer", "User works remotely", "User prefers coffee over tea", "User uses Python and JavaScript"]
            mock_extract_facts.return_value = mock_fact_extraction_response
            mock_extract_facts_consumer.return_value = mock_fact_extraction_response

            mock_long_term_response = MagicMock()
            mock_long_term_response.current_state = "Active: Working on project"
            mock_long_term_response.tentative_patterns = ["User is focused", "User is technical"]
            mock_long_term_response.knowledge_gaps = ["Personal background unclear"]
            mock_long_term_response.expectation_violations = []
            mock_long_term_response.updates = ["New: Focus on current project"]
            mock_long_term_user_rep.return_value = mock_long_term_response

            # Mock Anthropic client for conversational methods
            mock_message = MagicMock()
            mock_message.content = [MagicMock()]
            mock_message.content[0].text = mock_llm_responses['tom_conversational']
            mock_anthropic.messages.create.return_value = mock_message
            
            # Mock HTTP client to prevent real API calls with proper Groq format
            mock_http_response = MagicMock()
            mock_http_response.status_code = 200
            mock_http_response.headers = {"content-type": "application/json"}
            mock_http_response.text = json.dumps({
                "id": "test-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "llama-3.3-70b-versatile",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "current_state": "Active: Working on project",
                            "tentative_patterns": ["User is focused", "User is technical"],
                            "knowledge_gaps": ["Personal background unclear"],
                            "expectation_violations": [],
                            "updates": ["New: Focus on current project"]
                        })
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            })
            mock_http_response.json.return_value = json.loads(mock_http_response.text)
            mock_http_response.raise_for_status.return_value = None
            mock_httpx_send.return_value = mock_http_response
            

            yield {
                "tom_inference": mock_tom_inference,
                "user_rep_inference": mock_user_rep_inference,
                "extract_facts": mock_extract_facts,
                "long_term_user_rep": mock_long_term_user_rep,
                "anthropic": mock_anthropic,
                # Additional keys for tom_modules tests
                "single_prompt": MagicMock(generate=AsyncMock(return_value=mock_llm_responses['tom_single_prompt'])),
                "long_term": MagicMock(generate=AsyncMock(return_value=mock_llm_responses['tom_single_prompt'])),
            }
    else:
        # When using real APIs, return empty mocks so tests can run but use real functions
        yield {
            "tom_inference": None,
            "user_rep_inference": None,
            "extract_facts": None,
            "long_term_user_rep": None,
            "anthropic": None,
            "single_prompt": None,
            "long_term": None,
        }


@pytest.fixture(autouse=True)
def mock_vector_operations(mock_embeddings):
    """Mock vector similarity operations and database queries."""
    with (
        patch("src.crud.get_duplicate_documents") as mock_get_duplicates,
        patch("src.crud.create_collection") as mock_create_collection,
        patch("src.crud.get_collection_by_name") as mock_get_collection,
        patch("src.crud.get_documents") as mock_get_documents,
        patch("src.crud.query_documents") as mock_query_documents,
    ):
        # No duplicates by default
        mock_get_duplicates.return_value = []
        
        # Mock collection operations
        mock_collection = MagicMock()
        mock_collection.public_id = str(uuid4())
        mock_create_collection.return_value = mock_collection
        mock_get_collection.return_value = mock_collection
        
        # Mock document operations
        mock_get_documents.return_value = []
        mock_query_documents.return_value = []
        
        yield {
            "get_duplicates": mock_get_duplicates,
            "create_collection": mock_create_collection,
            "get_collection": mock_get_collection,
            "get_documents": mock_get_documents,
            "query_documents": mock_query_documents
        }


@pytest_asyncio.fixture
async def queue_manager():
    """Provides a QueueManager instance for testing."""
    # Mock the environment variable for test concurrency
    with patch("src.deriver.queue.os.getenv") as mock_getenv:
        mock_getenv.return_value = "2"  # 2 workers for testing
        manager = QueueManager()
        yield manager


@pytest_asyncio.fixture
async def embedding_store(sample_data):
    """Provides a CollectionEmbeddingStore for testing."""
    test_app, test_user = sample_data
    collection_id = str(uuid4())
    store = CollectionEmbeddingStore(test_app.public_id, test_user.public_id, collection_id)
    yield store


@pytest_asyncio.fixture
async def sample_messages(db_session, sample_data):
    """Creates sample messages for testing deriver processing."""
    test_app, test_user = sample_data
    
    # Create a test session
    session = models.Session(
        user_id=test_user.public_id,
        app_id=test_app.public_id,
        metadata={}
    )
    db_session.add(session)
    await db_session.flush()

    # Create sample messages
    messages = []
    
    # User message
    user_message = models.Message(
        session_id=session.public_id,
        is_user=True,
        content="I'm a Python developer working on AI projects. I prefer remote work and love debugging complex problems.",
        metadata={},
        user_id=test_user.public_id,
        app_id=test_app.public_id
    )
    db_session.add(user_message)
    messages.append(user_message)
    
    # AI message
    ai_message = models.Message(
        session_id=session.public_id,
        is_user=False,
        content="That's great! Python is excellent for AI development. What specific AI frameworks do you work with?",
        metadata={},
        user_id=test_user.public_id,
        app_id=test_app.public_id
    )
    db_session.add(ai_message)
    messages.append(ai_message)
    
    # Another user message
    user_message_2 = models.Message(
        session_id=session.public_id,
        is_user=True,
        content="I mainly use PyTorch and transformers. Currently building a chatbot with FastAPI.",
        metadata={},
        user_id=test_user.public_id,
        app_id=test_app.public_id
    )
    db_session.add(user_message_2)
    messages.append(user_message_2)
    
    await db_session.flush()
    
    yield session, messages


@pytest_asyncio.fixture
async def sample_queue_items(db_session, sample_messages):
    """Creates sample queue items for testing queue processing."""
    session, messages = sample_messages
    
    queue_items = []
    for message in messages:
        if message.is_user:  # Only user messages get queued for processing
            queue_item = models.QueueItem(
                session_id=session.id,  # Use integer ID, not public_id
                payload={"message_id": message.public_id},
                processed=False
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)
    
    await db_session.flush()
    yield session, messages, queue_items


@pytest_asyncio.fixture
async def sample_facts(db_session, sample_data):
    """Creates sample user facts stored in collections."""
    test_app, test_user = sample_data
    
    # Create user collection
    collection = models.Collection(
        app_id=test_app.public_id,
        user_id=test_user.public_id,
        name=f"user_{test_user.public_id}",
        metadata={"type": "user_facts"}
    )
    db_session.add(collection)
    await db_session.flush()
    
    # Create sample documents (facts)
    facts = [
        "User is a Python developer",
        "User works remotely",
        "User prefers PyTorch over TensorFlow",
        "User enjoys debugging complex problems"
    ]
    
    documents = []
    for fact in facts:
        doc = models.Document(
            collection_id=collection.public_id,
            content=fact,
            metadata={"extracted_at": "2024-01-01T00:00:00Z"},
            embedding=[0.1] * 1536  # Mock embedding
        )
        db_session.add(doc)
        documents.append(doc)
    
    await db_session.flush()
    yield collection, documents


@pytest.fixture
def mock_background_task():
    """Mock FastAPI BackgroundTasks for testing async task scheduling."""
    with patch("fastapi.BackgroundTasks") as mock_bg:
        mock_task = MagicMock()
        mock_bg.return_value = mock_task
        yield mock_task


@pytest.fixture
def mock_signal_handling():
    """Mock signal handling for testing graceful shutdown."""
    with (
        patch("signal.signal") as mock_signal,
        patch("signal.SIGTERM") as mock_sigterm,
        patch("signal.SIGINT") as mock_sigint,
    ):
        yield {
            "signal": mock_signal,
            "sigterm": mock_sigterm,
            "sigint": mock_sigint
        }


@pytest.fixture
def mock_semaphore():
    """Mock asyncio.Semaphore for testing concurrency control."""
    mock_sem = AsyncMock()
    mock_sem.__aenter__ = AsyncMock(return_value=mock_sem)
    mock_sem.__aexit__ = AsyncMock(return_value=None)
    
    with patch("asyncio.Semaphore") as mock_semaphore_class:
        mock_semaphore_class.return_value = mock_sem
        yield mock_sem


@pytest.fixture
def tom_method_config():
    """Fixture for testing different TOM method configurations."""
    return {
        "single_prompt": "SINGLE_PROMPT",
        "conversational": "CONVERSATIONAL", 
        "long_term": "LONG_TERM"
    }


@pytest_asyncio.fixture
async def mock_session_processing():
    """Mock session processing for integration tests."""
    with (
        patch("src.deriver.consumer.process_user_message") as mock_process_user,
        patch("src.deriver.consumer.process_ai_message") as mock_process_ai,
        patch("src.deriver.consumer.maybe_create_summary") as mock_create_summary,
    ):
        mock_process_user.return_value = None
        mock_process_ai.return_value = None
        mock_create_summary.return_value = None
        
        yield {
            "process_user": mock_process_user,
            "process_ai": mock_process_ai,
            "create_summary": mock_create_summary
        }


@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return {
        "max_workers": 4,
        "message_count": 100,
        "session_count": 10,
        "timeout_seconds": 30,
        "fact_extraction_time_limit": 5.0,  # seconds
        "tom_inference_time_limit": 3.0,    # seconds
        "queue_processing_time_limit": 1.0  # seconds per message
    }


@pytest.fixture
def error_scenarios():
    """Provides various error scenarios for testing error handling."""
    return {
        "llm_timeout": "LLM request timed out",
        "llm_api_error": "API rate limit exceeded",
        "database_connection_error": "Database connection failed",
        "invalid_json_response": "Malformed JSON in LLM response",
        "embedding_api_error": "Embedding service unavailable",
        "vector_similarity_error": "Vector similarity calculation failed"
    }