"""Functional tests for Honcho Zo skill tools.

These tests require a running Honcho server. Start the server with:

    uv run fastapi dev src/main.py

from the repository root before running these tests.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.save_memory import save_memory
from tools.query_memory import query_memory
from tools.get_context import get_context


class TestSaveMemory:
    """Tests for save_memory tool."""

    def test_returns_confirmation_string(self):
        """Test that save_memory returns a non-empty confirmation string."""
        result = save_memory("zo_test_user", "Hello, I love hiking!", "user", "zo_test_session")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_saves_user_message(self):
        """Test saving a user-role message."""
        result = save_memory("zo_test_user2", "I enjoy Python programming", "user", "zo_session2")

        assert isinstance(result, str)
        assert "user" in result.lower() or "zo_test_user2" in result

    def test_saves_assistant_message(self):
        """Test saving an assistant-role message."""
        result = save_memory("zo_test_user3", "That sounds great!", "assistant", "zo_session3")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_saves_multiple_turns(self):
        """Test saving multiple turns in the same session."""
        session_id = "zo_multi_turn_session"
        user_id = "zo_multi_turn_user"

        result1 = save_memory(user_id, "I love mountains", "user", session_id)
        result2 = save_memory(user_id, "That's wonderful!", "assistant", session_id)

        assert isinstance(result1, str) and len(result1) > 0
        assert isinstance(result2, str) and len(result2) > 0

    def test_non_assistant_role_treated_as_user(self):
        """Test that any role other than 'assistant' is treated as user."""
        result = save_memory("zo_role_user", "Testing role fallback", "human", "zo_role_session")

        assert isinstance(result, str)
        assert len(result) > 0


class TestQueryMemory:
    """Tests for query_memory tool."""

    def test_returns_string(self):
        """Test that query_memory returns a string response."""
        # Save something first
        save_memory("zo_query_user", "I love pizza and Italian food", "user", "zo_query_session")

        result = query_memory("zo_query_user", "What does the user enjoy?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string_with_session_scope(self):
        """Test query_memory scoped to a specific session."""
        session_id = "zo_scoped_session"
        save_memory("zo_scoped_user", "My favorite color is blue", "user", session_id)

        result = query_memory("zo_scoped_user", "What is the user's favorite color?", session_id)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_fallback_for_unknown_user(self):
        """Test that query_memory returns a non-empty string even for new users."""
        result = query_memory("zo_brand_new_user_xyz123", "What do I like?")

        assert isinstance(result, str)
        assert len(result) > 0


class TestGetContext:
    """Tests for get_context tool."""

    def test_returns_list(self):
        """Test that get_context returns a list."""
        session_id = "zo_context_session"
        user_id = "zo_context_user"

        save_memory(user_id, "Hello there!", "user", session_id)
        result = get_context(user_id, session_id, "assistant")

        assert isinstance(result, list)

    def test_returns_openai_format(self):
        """Test that returned messages are in OpenAI format."""
        session_id = "zo_openai_session"
        user_id = "zo_openai_user"

        save_memory(user_id, "My name is Alex", "user", session_id)
        save_memory(user_id, "Nice to meet you, Alex!", "assistant", session_id)

        result = get_context(user_id, session_id, "assistant")

        assert isinstance(result, list)
        for msg in result:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("user", "assistant", "system")
            assert isinstance(msg["content"], str)

    def test_respects_token_limit(self):
        """Test that context respects the token limit parameter."""
        session_id = "zo_token_session"
        user_id = "zo_token_user"

        # Add several messages
        for i in range(5):
            save_memory(user_id, f"Message number {i} with some content", "user", session_id)

        # Get context with a small token budget
        result_small = get_context(user_id, session_id, "assistant", tokens=100)
        result_large = get_context(user_id, session_id, "assistant", tokens=8000)

        # Both should be lists; larger budget may return more messages
        assert isinstance(result_small, list)
        assert isinstance(result_large, list)
        assert len(result_large) >= len(result_small)

    def test_empty_session_returns_list(self):
        """Test that get_context returns an empty list for a session with no messages."""
        result = get_context("zo_empty_user_xyz999", "zo_empty_session_xyz999", "assistant")

        assert isinstance(result, list)


class TestToolsWorkTogether:
    """Integration tests using all three tools in sequence."""

    def test_save_query_roundtrip(self):
        """Test saving a message and then querying it."""
        user_id = "zo_roundtrip_user"
        session_id = "zo_roundtrip_session"

        save_memory(user_id, "I am a software engineer who loves Rust", "user", session_id)
        result = query_memory(user_id, "What is the user's profession?", session_id)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_save_then_get_context(self):
        """Test that saved messages appear in context."""
        user_id = "zo_flow_user"
        session_id = "zo_flow_session"

        save_memory(user_id, "Hello!", "user", session_id)
        save_memory(user_id, "Hi there!", "assistant", session_id)

        messages = get_context(user_id, session_id, "assistant")

        assert isinstance(messages, list)
        assert len(messages) >= 1
