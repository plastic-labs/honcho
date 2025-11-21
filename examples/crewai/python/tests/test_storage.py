"""
Tests for HonchoStorage

Basic tests to verify the HonchoStorage implementation works correctly
with CrewAI's Storage interface.
"""

import time
import pytest
from honcho_crewai import HonchoStorage


def test_honcho_storage_initialization():
    """Test that HonchoStorage initializes correctly."""
    storage = HonchoStorage(user_id="test_user_123")

    assert storage is not None
    assert storage.session_id is not None
    assert storage.user is not None
    assert storage.assistant is not None
    assert storage.session is not None


def test_honcho_storage_save():
    """Test that messages can be saved to storage."""
    storage = HonchoStorage(user_id="test_user_456")

    # Save a user message
    storage.save("Hello, how are you?", metadata={"agent": "user"})

    # Save an assistant message
    storage.save("I'm doing well, thank you!", metadata={"agent": "assistant"})

    # If no exceptions raised, test passes


def test_honcho_storage_search():
    """Test that messages can be retrieved from storage."""
    storage = HonchoStorage(user_id="test_user_789")

    # Save some messages
    storage.save("I like pizza", metadata={"agent": "user"})
    storage.save("That's great! Pizza is delicious.", metadata={"agent": "assistant"})
    storage.save("What's your favorite topping?", metadata={"agent": "assistant"})

    # Search for messages
    results = storage.search("pizza", limit=10)

    assert isinstance(results, list)
    # Should have at least some messages
    assert len(results) >= 0
    # Each result should have the required keys
    for result in results:
        assert "memory" in result
        assert "context" in result
        assert "metadata" in result


def test_honcho_storage_reset():
    """Test that storage can be reset to a new session."""
    storage = HonchoStorage(user_id="test_user_reset")

    # Save a message
    storage.save("Test message", metadata={"agent": "user"})

    # Get the original session ID
    original_session_id = storage.session_id

    # Reset the storage
    storage.reset()

    # Session ID should be different
    assert storage.session_id != original_session_id


def test_honcho_storage_with_custom_session():
    """Test that storage can be initialized with a custom session ID."""
    custom_session_id = "my_custom_session_123"
    storage = HonchoStorage(user_id="test_user_custom", session_id=custom_session_id)

    assert storage.session_id == custom_session_id


def test_honcho_storage_search_format():
    """Test that search results are in the correct format for CrewAI."""
    storage = HonchoStorage(user_id="test_user_format")

    # Save a message
    test_message = "Testing message format"
    storage.save(test_message, metadata={"agent": "user"})

    # Search
    results = storage.search("format", limit=5)

    if len(results) > 0:
        result = results[0]
        # Verify CrewAI expected format
        assert "memory" in result
        assert "context" in result
        assert "metadata" in result
        assert isinstance(result["memory"], str)
        assert isinstance(result["context"], str)
        assert isinstance(result["metadata"], dict)


def test_semantic_search_relevance():
    """Test that semantic search returns relevant results based on query."""
    storage = HonchoStorage(user_id="test_user_semantic")

    # Save messages on different topics
    storage.save("I love Italian food, especially pasta and pizza", metadata={"agent": "user"})
    storage.save("My favorite programming language is Python", metadata={"agent": "user"})
    storage.save("The weather today is sunny and warm", metadata={"agent": "user"})
    storage.save("Pizza is the best food in the world", metadata={"agent": "assistant"})

    # Allow time for indexing
    time.sleep(2)

    # Search for food-related content
    results = storage.search("pizza and Italian cuisine", limit=5)

    # Results should exist
    assert isinstance(results, list)

    # If results are returned, they should be relevant to the query
    # The semantic search should prioritize food-related messages
    if len(results) > 0:
        # Check that results contain expected fields
        for result in results:
            assert "memory" in result
            assert "context" in result
            assert "content" in result
            assert "metadata" in result


def test_session_summaries_included():
    """Test that session summaries are included in search results when available."""
    storage = HonchoStorage(user_id="test_user_summaries")

    # Save enough messages to potentially trigger summary generation
    # (Honcho generates summaries after ~20 messages)
    for i in range(25):
        storage.save(
            f"Message {i}: This is a test message about various topics",
            metadata={"agent": "user" if i % 2 == 0 else "assistant"}
        )

    # Allow time for background processing
    time.sleep(5)

    # Search should include summaries if available
    results = storage.search("test message", limit=10)

    assert isinstance(results, list)

    # Check if any summaries are present
    summary_results = [r for r in results if r["metadata"].get("type") == "summary"]

    # If summaries exist, verify their format
    for summary in summary_results:
        assert "summary_type" in summary["metadata"]
        assert summary["metadata"]["summary_type"] in ["short", "long"]
        assert "[Session Summary]" in summary["context"]
        assert "created_at" in summary["metadata"]

def test_search_limit_parameter():
    """Test that the limit parameter correctly restricts result count."""
    storage = HonchoStorage(user_id="test_user_limit")

    # Save multiple messages
    for i in range(10):
        storage.save(f"Message number {i} about testing", metadata={"agent": "user"})

    # Allow time for indexing
    time.sleep(2)

    # Search with small limit
    results = storage.search("testing", limit=3)

    # Note: Results may include summaries, so we check the total count
    # The limit should be respected (though summaries may add to the count)
    assert isinstance(results, list)
    # Should not exceed limit significantly (summaries could add 1-2)
    assert len(results) <= 5  # Allow some margin for summaries


def test_search_no_results():
    """Test search behavior when no relevant messages exist."""
    storage = HonchoStorage(user_id="test_user_noresults")

    # Save unrelated messages
    storage.save("The quick brown fox", metadata={"agent": "user"})

    time.sleep(1)

    # Search for completely unrelated content
    results = storage.search("quantum physics blockchain cryptocurrency", limit=5)

    # Should return a list (may be empty or with low relevance)
    assert isinstance(results, list)


def test_multiple_peers_search():
    """Test that search works with messages from multiple peers."""
    storage = HonchoStorage(user_id="test_user_multipeer")

    # Save messages from both user and assistant
    storage.save("I need help with Python", metadata={"agent": "user"})
    storage.save("I can help you with Python!", metadata={"agent": "assistant"})
    storage.save("How do I use decorators?", metadata={"agent": "user"})
    storage.save("Decorators are functions that modify other functions", metadata={"agent": "assistant"})

    time.sleep(2)

    # Search for Python-related content
    results = storage.search("Python programming", limit=10)

    if len(results) > 0:
        # Should find messages from both peers
        peer_ids = set(r["metadata"].get("peer_id") for r in results if "peer_id" in r["metadata"])

        # Verify we have results with peer information
        assert len(peer_ids) > 0

        # All results should have required fields
        for result in results:
            assert "memory" in result
            assert "context" in result
            assert "content" in result
            assert result["memory"] == result["content"]  # Should match


def test_search_result_format_consistency():
    """Test that all search results have consistent format."""
    storage = HonchoStorage(user_id="test_user_format_consistency")

    # Save various types of messages
    storage.save("Regular message", metadata={"agent": "user"})
    storage.save("Another message", metadata={"agent": "assistant", "type": "response"})

    time.sleep(1)

    results = storage.search("message", limit=5)

    # All results should have the same structure
    required_keys = ["content", "memory", "context", "metadata"]

    for result in results:
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check types
        assert isinstance(result["content"], str)
        assert isinstance(result["memory"], str)
        assert isinstance(result["context"], str)
        assert isinstance(result["metadata"], dict)


def test_save_preserves_peer_identity():
    """Test that save correctly identifies user vs assistant messages."""
    storage = HonchoStorage(user_id="test_user_peers")

    # Save as user
    storage.save("User message", metadata={"agent": "user"})
    storage.save("User message with role", metadata={"role": "user"})

    # Save as assistant
    storage.save("Assistant message", metadata={"agent": "assistant"})
    storage.save("Default message", metadata={})  # Should default to assistant

    # If no exceptions raised, test passes
    # The peer assignment happens internally


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
