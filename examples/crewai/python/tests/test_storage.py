"""
Tests for HonchoStorage

Basic tests to verify the HonchoStorage implementation works correctly
with CrewAI's Storage interface.
"""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
