"""
Tests for Honcho Agno Tools

Tests the Agno-Honcho tool integration layer using real Honcho SDK.
Focuses on tool interface compliance and result formatting.
"""

import uuid

import pytest

from honcho_agno import HonchoTools


class TestHonchoToolsInitialization:
    """Tests for HonchoTools initialization."""

    def test_default_initialization(self):
        """Test that toolkit initializes with default parameters."""
        tools = HonchoTools()

        assert tools is not None
        assert tools.name == "honcho"
        assert tools.honcho is not None
        assert tools.session_id is not None
        assert tools.user_id == "default"
        assert tools.app_id == "default"

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_session = str(uuid.uuid4())
        tools = HonchoTools(
            app_id="test-app",
            user_id="test-user",
            session_id=custom_session,
        )

        assert tools.user_id == "test-user"
        assert tools.app_id == "test-app"
        assert tools.session_id == custom_session

    def test_session_auto_generation(self):
        """Test that session_id is auto-generated if not provided."""
        tools = HonchoTools(
            app_id="test-app",
            user_id="test-user",
        )

        assert tools.session_id is not None
        # Should be a valid UUID format
        uuid.UUID(tools.session_id)


class TestAddMessage:
    """Tests for add_message tool."""

    def test_add_user_message(self):
        """Test adding a user message."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"add-msg-user-{uuid.uuid4()}",
            session_id=f"add-msg-session-{uuid.uuid4()}",
        )

        result = tools.add_message("Test message content", role="user")

        assert isinstance(result, str)
        assert "saved" in result.lower() or "success" in result.lower()

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"add-msg-user-{uuid.uuid4()}",
            session_id=f"add-msg-session-{uuid.uuid4()}",
        )

        result = tools.add_message("Assistant response", role="assistant")

        assert isinstance(result, str)
        assert "saved" in result.lower() or "success" in result.lower()

    def test_add_multiple_messages(self):
        """Test adding multiple messages in sequence."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"multi-msg-user-{uuid.uuid4()}",
            session_id=f"multi-msg-session-{uuid.uuid4()}",
        )

        messages = [
            ("Hello, I need help", "user"),
            ("Of course! How can I assist?", "assistant"),
            ("I want to learn Python", "user"),
        ]

        for content, role in messages:
            result = tools.add_message(content, role=role)
            assert isinstance(result, str)
            assert "error" not in result.lower()


class TestGetContext:
    """Tests for get_context tool."""

    def test_get_empty_context(self):
        """Test getting context from empty session."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"context-user-{uuid.uuid4()}",
            session_id=f"empty-context-{uuid.uuid4()}",
        )

        result = tools.get_context()

        assert isinstance(result, str)

    def test_get_context_with_messages(self):
        """Test getting context after adding messages."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"context-user-{uuid.uuid4()}",
            session_id=f"context-session-{uuid.uuid4()}",
        )

        # Add messages first
        tools.add_message("I like pizza", role="user")
        tools.add_message("Great choice!", role="assistant")

        result = tools.get_context()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_context_with_token_limit(self):
        """Test getting context with token limit."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"token-user-{uuid.uuid4()}",
            session_id=f"token-session-{uuid.uuid4()}",
        )

        tools.add_message("This is a test message", role="user")

        result = tools.get_context(tokens=1000)

        assert isinstance(result, str)

    def test_get_context_without_summary(self):
        """Test getting context without summary."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"nosummary-user-{uuid.uuid4()}",
            session_id=f"nosummary-session-{uuid.uuid4()}",
        )

        tools.add_message("Test message", role="user")

        result = tools.get_context(include_summary=False)

        assert isinstance(result, str)


class TestSearchMessages:
    """Tests for search_messages tool."""

    def test_search_returns_formatted_string(self):
        """Test that search returns formatted results."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"search-user-{uuid.uuid4()}",
            session_id=f"search-session-{uuid.uuid4()}",
        )

        # Add searchable content
        tools.add_message("I enjoy Python programming and data science", role="user")

        result = tools.search_messages("programming", limit=5)

        assert isinstance(result, str)
        assert "Search Results" in result or "No messages found" in result

    def test_search_with_limit(self):
        """Test search with custom limit."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"search-limit-user-{uuid.uuid4()}",
            session_id=f"search-limit-session-{uuid.uuid4()}",
        )

        # Add multiple messages
        for i in range(5):
            tools.add_message(f"Test message number {i} about coding", role="user")

        result = tools.search_messages("coding", limit=3)

        assert isinstance(result, str)

    def test_search_no_results(self):
        """Test search with no matching results."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"search-empty-user-{uuid.uuid4()}",
            session_id=f"search-empty-session-{uuid.uuid4()}",
        )

        result = tools.search_messages("xyznonexistent123abcdef", limit=5)

        assert isinstance(result, str)
        # Should indicate no results found
        assert "No messages found" in result or "0 found" in result.lower()


class TestQueryUser:
    """Tests for query_user tool."""

    def test_query_returns_response(self):
        """Test that query returns a response string."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"query-user-{uuid.uuid4()}",
            session_id=f"query-session-{uuid.uuid4()}",
        )

        # Add context first
        tools.add_message("I love hiking and outdoor activities", role="user")
        tools.add_message("I also enjoy photography", role="user")

        result = tools.query_user("What does the user enjoy?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_without_context(self):
        """Test query on user with minimal context."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"query-empty-user-{uuid.uuid4()}",
            session_id=f"query-empty-session-{uuid.uuid4()}",
        )

        result = tools.query_user("What are the user's preferences?")

        assert isinstance(result, str)


class TestResetSession:
    """Tests for reset_session tool."""

    def test_reset_creates_new_session(self):
        """Test that reset creates a new session."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"reset-user-{uuid.uuid4()}",
            session_id=f"original-session-{uuid.uuid4()}",
        )

        original_session = tools.session_id

        result = tools.reset_session()

        assert isinstance(result, str)
        assert tools.session_id != original_session
        assert "reset" in result.lower() or "new" in result.lower()


class TestToolsIntegration:
    """Integration tests for all tools working together."""

    def test_all_tools_in_sequence(self):
        """Test using all tools in a realistic sequence."""
        tools = HonchoTools(
            app_id="test-app",
            user_id=f"integration-user-{uuid.uuid4()}",
            session_id=f"integration-session-{uuid.uuid4()}",
        )

        # Add messages
        add_result = tools.add_message(
            "I'm interested in AI and machine learning", role="user"
        )
        assert isinstance(add_result, str)
        assert "error" not in add_result.lower()

        # Get context
        context_result = tools.get_context()
        assert isinstance(context_result, str)

        # Search messages
        search_result = tools.search_messages("AI", limit=10)
        assert isinstance(search_result, str)

        # Query user
        query_result = tools.query_user("What topics interest the user?")
        assert isinstance(query_result, str)

    def test_multiple_sessions_same_user(self):
        """Test using multiple sessions for the same user."""
        user_id = f"multi-session-user-{uuid.uuid4()}"

        # First session
        tools1 = HonchoTools(
            app_id="test-app",
            user_id=user_id,
            session_id=f"session-1-{uuid.uuid4()}",
        )
        tools1.add_message("I like Python", role="user")

        # Second session
        tools2 = HonchoTools(
            app_id="test-app",
            user_id=user_id,
            session_id=f"session-2-{uuid.uuid4()}",
        )
        tools2.add_message("I also like JavaScript", role="user")

        # Both sessions should work independently
        context1 = tools1.get_context()
        context2 = tools2.get_context()

        assert isinstance(context1, str)
        assert isinstance(context2, str)
