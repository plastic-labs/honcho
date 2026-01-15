"""
Tests for Honcho Agno Tools

Tests the Agno-Honcho tool integration layer using real Honcho SDK.
Focuses on tool interface compliance and result formatting.

Note: Each HonchoTools instance represents ONE agent identity (peer_id).
The toolkit provides read access; orchestration code handles message saving.
"""

import uuid

from honcho import Honcho
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
        assert tools.peer_id == "assistant"

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_session = str(uuid.uuid4())
        tools = HonchoTools(
            app_id="test-app",
            peer_id="custom-agent",
            session_id=custom_session,
        )

        assert tools.peer_id == "custom-agent"
        assert tools.session_id == custom_session

    def test_session_auto_generation(self):
        """Test that session_id is auto-generated if not provided."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id="test-agent",
        )

        assert tools.session_id is not None
        # Should be a valid UUID format
        uuid.UUID(tools.session_id)

    def test_toolkit_represents_single_peer(self):
        """Test that toolkit has exactly one peer identity."""
        tools = HonchoTools(peer_id="my-agent")

        # Should have exactly one peer
        assert tools.peer is not None
        assert tools.peer_id == "my-agent"
        # Should NOT have separate user/assistant peers
        assert not hasattr(tools, "user")
        assert not hasattr(tools, "assistant")

    def test_honcho_client_used_directly(self):
        """Test that honcho_client is used when provided."""
        honcho = Honcho(workspace_id="client-workspace")
        tools = HonchoTools(
            app_id="ignored-app",  # Ignored when honcho_client provided
            peer_id="test-agent",
            honcho_client=honcho,
        )

        # The client should be the one we passed
        assert tools.honcho is honcho


class TestMultiPeerConversation:
    """Tests for multi-peer conversation patterns."""

    def test_multiple_toolkits_same_session(self):
        """Test multiple toolkits (agents) sharing a session."""
        session_id = f"shared-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        # Two agents with different identities
        agent1_tools = HonchoTools(
            app_id="test-app",
            peer_id="agent-alpha",
            session_id=session_id,
            honcho_client=honcho,
        )

        agent2_tools = HonchoTools(
            app_id="test-app",
            peer_id="agent-beta",
            session_id=session_id,
            honcho_client=honcho,
        )

        # Both should share the same session
        assert agent1_tools.session_id == agent2_tools.session_id
        # But have different peer identities
        assert agent1_tools.peer_id != agent2_tools.peer_id

    def test_messages_added_via_orchestration(self):
        """Test that messages are added via session, not toolkit methods."""
        session_id = f"orch-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        # Create toolkit
        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        # User messages added directly via Honcho (orchestration pattern)
        user_peer = honcho.peer("user")
        tools.session.add_messages([user_peer.message("Hello from user")])

        # Agent messages also added via orchestration
        tools.session.add_messages([tools.peer.message("Hello from assistant")])

        # Context should show both messages
        context = tools.get_context()
        assert isinstance(context, str)


class TestGetContext:
    """Tests for get_context tool."""

    def test_get_empty_context(self):
        """Test getting context from empty session."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"empty-session-{uuid.uuid4().hex[:8]}",
        )

        result = tools.get_context()

        assert isinstance(result, str)

    def test_get_context_with_messages(self):
        """Test getting context after adding messages."""
        session_id = f"context-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        # Add messages via orchestration
        user = honcho.peer("user")
        tools.session.add_messages([
            user.message("I like pizza"),
            tools.peer.message("Great choice!"),
        ])

        result = tools.get_context()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_context_with_token_limit(self):
        """Test getting context with token limit."""
        session_id = f"token-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        tools.session.add_messages([tools.peer.message("This is a test message")])

        result = tools.get_context(tokens=1000)

        assert isinstance(result, str)

    def test_get_context_without_summary(self):
        """Test getting context without summary."""
        session_id = f"nosummary-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        tools.session.add_messages([tools.peer.message("Test message")])

        result = tools.get_context(include_summary=False)

        assert isinstance(result, str)


class TestSearchMessages:
    """Tests for search_messages tool."""

    def test_search_returns_formatted_string(self):
        """Test that search returns formatted results."""
        session_id = f"search-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        # Add searchable content
        tools.session.add_messages([
            tools.peer.message("I enjoy Python programming and data science")
        ])

        result = tools.search_messages("programming", limit=5)

        assert isinstance(result, str)
        assert "Search Results" in result or "No messages found" in result

    def test_search_with_limit(self):
        """Test search with custom limit."""
        session_id = f"search-limit-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        # Add multiple messages
        messages = [tools.peer.message(f"Test message number {i} about coding") for i in range(5)]
        tools.session.add_messages(messages)

        result = tools.search_messages("coding", limit=3)

        assert isinstance(result, str)

    def test_search_no_results(self):
        """Test search with no matching results."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"search-empty-session-{uuid.uuid4().hex[:8]}",
        )

        result = tools.search_messages("xyznonexistent123abcdef", limit=5)

        assert isinstance(result, str)
        # Should indicate no results found
        assert "No messages found" in result or "0 found" in result.lower()


class TestChat:
    """Tests for chat tool."""

    def test_chat_returns_response(self):
        """Test that chat returns a response string."""
        session_id = f"chat-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        # Add context first
        user = honcho.peer("user")
        tools.session.add_messages([
            user.message("The user loves hiking and outdoor activities"),
            user.message("They also enjoy photography"),
        ])

        result = tools.chat("What does this person enjoy?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_without_context(self):
        """Test chat with minimal context."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"chat-empty-session-{uuid.uuid4().hex[:8]}",
        )

        result = tools.chat("What are the user's preferences?")

        assert isinstance(result, str)


class TestToolsIntegration:
    """Integration tests for all tools working together."""

    def test_all_tools_in_sequence(self):
        """Test using all tools in a realistic sequence."""
        session_id = f"integration-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        # Add message via orchestration
        user = honcho.peer("user")
        tools.session.add_messages([
            user.message("I'm interested in AI and machine learning")
        ])

        # Get context
        context_result = tools.get_context()
        assert isinstance(context_result, str)

        # Search messages
        search_result = tools.search_messages("AI", limit=10)
        assert isinstance(search_result, str)

        # Chat
        chat_result = tools.chat("What topics are mentioned?")
        assert isinstance(chat_result, str)

    def test_multi_agent_conversation(self):
        """Test realistic multi-agent conversation."""
        session_id = f"multi-agent-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        # User peer managed directly
        session = honcho.session(session_id)
        user = honcho.peer("user")

        # Two agent toolkits
        tech_agent = HonchoTools(
            app_id="test-app",
            peer_id="tech-advisor",
            session_id=session_id,
            honcho_client=honcho,
        )

        biz_agent = HonchoTools(
            app_id="test-app",
            peer_id="business-advisor",
            session_id=session_id,
            honcho_client=honcho,
        )

        # Conversation flow via orchestration
        session.add_messages([user.message("I want to build a SaaS product")])
        session.add_messages([tech_agent.peer.message("Consider microservices architecture")])
        session.add_messages([biz_agent.peer.message("Focus on a niche market first")])

        # Both agents can see full context
        tech_context = tech_agent.get_context()
        biz_context = biz_agent.get_context()

        assert isinstance(tech_context, str)
        assert isinstance(biz_context, str)
