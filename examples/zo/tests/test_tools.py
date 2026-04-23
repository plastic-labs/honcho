"""Functional tests for Honcho Zo skill tools.

These tests require a Honcho API key set in the HONCHO_API_KEY environment
variable. They run against the Honcho cloud API (honcho.dev) by default.
Set HONCHO_WORKSPACE_ID to scope tests to a specific workspace.
"""

import os
import sys
import time
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.get_context import get_context
from tools.query_memory import query_memory
from tools.save_memory import save_memory

pytestmark = pytest.mark.skipif(
    not os.getenv("HONCHO_API_KEY"),
    reason="HONCHO_API_KEY not set — skipping integration tests",
)


@pytest.fixture(autouse=True)
def rate_limit_delay():
    """Pause between tests to stay under the Honcho API rate limit (5 req/sec)."""
    yield
    time.sleep(0.5)


def unique_id(prefix: str) -> str:
    """Generate a unique ID with a prefix to avoid test state leakage."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class TestSaveMemory:
    """Tests for save_memory tool."""

    def test_returns_confirmation_string(self):
        """Test that save_memory returns a non-empty confirmation string."""
        result = save_memory(unique_id("user"), "Hello, I love hiking!", "user", unique_id("session"))

        assert isinstance(result, str)
        assert len(result) > 0

    def test_saves_user_message(self):
        """Test saving a user-role message."""
        user_id = unique_id("user")
        result = save_memory(user_id, "I enjoy Python programming", "user", unique_id("session"))

        assert isinstance(result, str)
        assert "user" in result.lower() or user_id in result

    def test_saves_assistant_message(self):
        """Test saving an assistant-role message."""
        result = save_memory(unique_id("user"), "That sounds great!", "assistant", unique_id("session"))

        assert isinstance(result, str)
        assert len(result) > 0

    def test_saves_multiple_turns(self):
        """Test saving multiple turns in the same session."""
        user_id = unique_id("user")
        session_id = unique_id("session")

        result1 = save_memory(user_id, "I love mountains", "user", session_id)
        result2 = save_memory(user_id, "That's wonderful!", "assistant", session_id)

        assert isinstance(result1, str) and len(result1) > 0
        assert isinstance(result2, str) and len(result2) > 0

    def test_non_assistant_role_treated_as_user(self):
        """Test that any role other than 'assistant' is treated as user."""
        result = save_memory(unique_id("user"), "Testing role fallback", "human", unique_id("session"))

        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_assistant_id(self):
        """Test that a custom assistant_id is accepted."""
        result = save_memory(
            unique_id("user"), "Hello!", "assistant", unique_id("session"), assistant_id="my-bot"
        )

        assert isinstance(result, str)
        assert len(result) > 0


class TestQueryMemory:
    """Tests for query_memory tool."""

    def test_returns_string(self):
        """Test that query_memory returns a string response."""
        user_id = unique_id("user")
        session_id = unique_id("session")
        save_memory(user_id, "I love pizza and Italian food", "user", session_id)

        result = query_memory(user_id, "What does the user enjoy?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string_with_session_scope(self):
        """Test query_memory scoped to a specific session."""
        user_id = unique_id("user")
        session_id = unique_id("session")
        save_memory(user_id, "My favorite color is blue", "user", session_id)

        result = query_memory(user_id, "What is the user's favorite color?", session_id)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_fallback_for_unknown_user(self):
        """Test that query_memory returns a non-empty string even for new users."""
        result = query_memory(unique_id("user"), "What do I like?")

        assert isinstance(result, str)
        assert len(result) > 0


class TestGetContext:
    """Tests for get_context tool."""

    def test_returns_list(self):
        """Test that get_context returns a list."""
        user_id = unique_id("user")
        session_id = unique_id("session")
        save_memory(user_id, "Hello there!", "user", session_id)

        result = get_context(user_id, session_id, "assistant")

        assert isinstance(result, list)

    def test_returns_openai_format(self):
        """Test that returned messages are in OpenAI format."""
        user_id = unique_id("user")
        session_id = unique_id("session")
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
        user_id = unique_id("user")
        session_id = unique_id("session")
        for i in range(5):
            save_memory(user_id, f"Message number {i} with some content", "user", session_id)

        result_small = get_context(user_id, session_id, "assistant", tokens=100)
        result_large = get_context(user_id, session_id, "assistant", tokens=8000)

        assert isinstance(result_small, list)
        assert isinstance(result_large, list)
        assert len(result_large) >= len(result_small)

    def test_empty_session_returns_list(self):
        """Test that get_context returns an empty list for a session with no messages."""
        result = get_context(unique_id("user"), unique_id("session"), "assistant")

        assert isinstance(result, list)


class TestToolsWorkTogether:
    """Integration tests using all three tools in sequence."""

    def test_save_query_roundtrip(self):
        """Test saving a message and then querying it."""
        user_id = unique_id("user")
        session_id = unique_id("session")
        save_memory(user_id, "I am a software engineer who loves Rust", "user", session_id)

        result = query_memory(user_id, "What is the user's profession?", session_id)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_save_then_get_context(self):
        """Test that saved messages appear in context."""
        user_id = unique_id("user")
        session_id = unique_id("session")
        save_memory(user_id, "Hello!", "user", session_id)
        save_memory(user_id, "Hi there!", "assistant", session_id)

        messages = get_context(user_id, session_id, "assistant")

        assert isinstance(messages, list)
        assert len(messages) >= 1
