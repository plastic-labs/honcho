"""
Tests for HonchoStorage

Tests the CrewAI-Honcho integration layer, focusing on:
- CrewAI Storage interface compliance
- Metadata mapping (agent/role -> peer_id)
- Format conversion (Honcho -> CrewAI format)
"""

from honcho_crewai import HonchoStorage


class TestHonchoStorage:
    """Tests for HonchoStorage integration layer."""

    def test_initialization(self):
        """Test that HonchoStorage initializes with correct peers and session."""
        storage = HonchoStorage(user_id="test_user")

        assert storage is not None
        assert storage.session_id is not None
        assert storage.user is not None
        assert storage.assistant is not None
        assert storage.session is not None

    def test_initialization_with_custom_session(self):
        """Test that custom session_id is preserved."""
        custom_session_id = "my_custom_session"
        storage = HonchoStorage(user_id="test_user", session_id=custom_session_id)

        assert storage.session_id == custom_session_id

    def test_save_with_different_roles(self):
        """Test that save handles different agent/role metadata."""
        storage = HonchoStorage(user_id="test_user_roles")

        # Save with different metadata patterns
        storage.save("User via agent", metadata={"agent": "user"})
        storage.save("User via role", metadata={"role": "user"})
        storage.save("Assistant via agent", metadata={"agent": "assistant"})
        storage.save("Default (no metadata)", metadata={})

        # If no exceptions raised, metadata mapping works

    def test_search_returns_crewai_format(self):
        """Test that search returns results in CrewAI format."""
        storage = HonchoStorage(user_id="test_user_search")

        # Add a message
        storage.save("Test message", metadata={"agent": "user"})

        # Search
        results = storage.search("test", limit=10)

        # Verify CrewAI format
        assert isinstance(results, list)
        for result in results:
            # Required keys for CrewAI
            assert "memory" in result
            assert "context" in result
            assert "content" in result
            assert "metadata" in result

    def test_search_includes_all_required_fields(self):
        """Test that all search results have required CrewAI fields."""
        storage = HonchoStorage(user_id="test_user_format")

        # Add a message
        storage.save("Test message", metadata={"agent": "user"})

        # Search
        results = storage.search("test", limit=5)

        # Verify all results have required fields with correct types
        for result in results:
            assert isinstance(result["content"], str)
            assert isinstance(result["memory"], str)
            assert isinstance(result["context"], str)
            assert isinstance(result["metadata"], dict)

    def test_search_formats_summaries_correctly(self):
        """Test that session summaries are formatted with [Session Summary] prefix."""
        storage = HonchoStorage(user_id="test_user_summaries")

        # Add enough messages to potentially trigger summaries
        for i in range(25):
            storage.save(
                f"Message {i}",
                metadata={"agent": "user" if i % 2 == 0 else "assistant"},
            )

        # Search
        results = storage.search("message", limit=10)

        # Check summary formatting (if summaries exist)
        summary_results = [r for r in results if r["metadata"].get("type") == "summary"]

        for summary in summary_results:
            # Verify our formatting logic
            assert "summary_type" in summary["metadata"]
            assert "[Session Summary]" in summary["context"]  # Our formatting

    def test_reset_creates_new_session_id(self):
        """Test that reset() creates a new session with different ID."""
        storage = HonchoStorage(user_id="test_user_reset")

        original_session_id = storage.session_id

        # Reset
        storage.reset()

        # Verify new session ID was created
        assert storage.session_id != original_session_id
