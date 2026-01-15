"""
Tests for Honcho Agno Tools

Tests the Agno-Honcho tool integration layer using real Honcho SDK.
Focuses on tool interface compliance and result formatting.

Note: Each HonchoTools instance represents ONE agent identity (peer_id).
Messages added via add_message() are attributed to that peer.
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
        assert tools.app_id == "default"

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_session = str(uuid.uuid4())
        tools = HonchoTools(
            app_id="test-app",
            peer_id="custom-agent",
            session_id=custom_session,
        )

        assert tools.peer_id == "custom-agent"
        assert tools.app_id == "test-app"
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


class TestAddMessage:
    """Tests for add_message tool."""

    def test_add_message_as_peer(self):
        """Test adding a message attributed to the toolkit's peer."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"session-{uuid.uuid4().hex[:8]}",
        )

        result = tools.add_message("Test message content")

        assert isinstance(result, str)
        assert "saved" in result.lower() or "success" in result.lower()
        assert tools.peer_id in result  # Should mention the peer

    def test_add_multiple_messages(self):
        """Test adding multiple messages in sequence."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"session-{uuid.uuid4().hex[:8]}",
        )

        messages = [
            "First message from agent",
            "Second message from agent",
            "Third message from agent",
        ]

        for content in messages:
            result = tools.add_message(content)
            assert isinstance(result, str)
            assert "error" not in result.lower()


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

        # Both add messages to the same session
        result1 = agent1_tools.add_message("Message from Alpha")
        result2 = agent2_tools.add_message("Message from Beta")

        assert "agent-alpha" in result1
        assert "agent-beta" in result2
        assert agent1_tools.session_id == agent2_tools.session_id

    def test_user_messages_via_honcho_directly(self):
        """Test adding user messages via Honcho while agent uses toolkit."""
        session_id = f"mixed-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        # User messages added directly via Honcho
        session = honcho.session(session_id)
        user_peer = honcho.peer("user")
        session.add_messages([user_peer.message("Hello from user")])

        # Agent uses toolkit
        agent_tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        result = agent_tools.add_message("Hello from assistant")
        assert "assistant" in result

        # Both should be in context
        context = agent_tools.get_context()
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
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"context-session-{uuid.uuid4().hex[:8]}",
        )

        # Add messages first
        tools.add_message("I like pizza")
        tools.add_message("Great choice!")

        result = tools.get_context()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_context_with_token_limit(self):
        """Test getting context with token limit."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"token-session-{uuid.uuid4().hex[:8]}",
        )

        tools.add_message("This is a test message")

        result = tools.get_context(tokens=1000)

        assert isinstance(result, str)

    def test_get_context_without_summary(self):
        """Test getting context without summary."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"nosummary-session-{uuid.uuid4().hex[:8]}",
        )

        tools.add_message("Test message")

        result = tools.get_context(include_summary=False)

        assert isinstance(result, str)


class TestSearchMessages:
    """Tests for search_messages tool."""

    def test_search_returns_formatted_string(self):
        """Test that search returns formatted results."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"search-session-{uuid.uuid4().hex[:8]}",
        )

        # Add searchable content
        tools.add_message("I enjoy Python programming and data science")

        result = tools.search_messages("programming", limit=5)

        assert isinstance(result, str)
        assert "Search Results" in result or "No messages found" in result

    def test_search_with_limit(self):
        """Test search with custom limit."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"search-limit-session-{uuid.uuid4().hex[:8]}",
        )

        # Add multiple messages
        for i in range(5):
            tools.add_message(f"Test message number {i} about coding")

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


class TestQueryPeer:
    """Tests for query_peer tool."""

    def test_query_returns_response(self):
        """Test that query returns a response string."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"query-session-{uuid.uuid4().hex[:8]}",
        )

        # Add context first
        tools.add_message("The user loves hiking and outdoor activities")
        tools.add_message("They also enjoy photography")

        result = tools.query_peer("What does this person enjoy?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_without_context(self):
        """Test query with minimal context."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"query-empty-session-{uuid.uuid4().hex[:8]}",
        )

        result = tools.query_peer("What are the user's preferences?")

        assert isinstance(result, str)

    def test_query_specific_peer(self):
        """Test querying about a specific peer by ID."""
        session_id = f"query-peer-session-{uuid.uuid4().hex[:8]}"
        honcho = Honcho(workspace_id="test-app")

        # Add user messages directly
        session = honcho.session(session_id)
        user_peer = honcho.peer("user")
        session.add_messages([
            user_peer.message("I love hiking"),
            user_peer.message("Photography is my hobby"),
        ])

        # Agent queries about the user
        tools = HonchoTools(
            app_id="test-app",
            peer_id="assistant",
            session_id=session_id,
            honcho_client=honcho,
        )

        result = tools.query_peer("What are their interests?", target_peer_id="user")

        assert isinstance(result, str)


class TestResetSession:
    """Tests for reset_session tool."""

    def test_reset_creates_new_session(self):
        """Test that reset creates a new session."""
        tools = HonchoTools(
            app_id="test-app",
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"original-session-{uuid.uuid4().hex[:8]}",
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
            peer_id=f"agent-{uuid.uuid4().hex[:8]}",
            session_id=f"integration-session-{uuid.uuid4().hex[:8]}",
        )

        # Add message
        add_result = tools.add_message("I'm interested in AI and machine learning")
        assert isinstance(add_result, str)
        assert "error" not in add_result.lower()

        # Get context
        context_result = tools.get_context()
        assert isinstance(context_result, str)

        # Search messages
        search_result = tools.search_messages("AI", limit=10)
        assert isinstance(search_result, str)

        # Query peer
        query_result = tools.query_peer("What topics are mentioned?")
        assert isinstance(query_result, str)

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

        # Conversation flow
        session.add_messages([user.message("I want to build a SaaS product")])
        tech_agent.add_message("Consider microservices architecture")
        biz_agent.add_message("Focus on a niche market first")

        # Both agents can see full context
        tech_context = tech_agent.get_context()
        biz_context = biz_agent.get_context()

        assert isinstance(tech_context, str)
        assert isinstance(biz_context, str)
