"""
Tests for Honcho CrewAI Tools

Tests the CrewAI-Honcho tool integration layer using real Honcho SDK.
Focuses on tool interface compliance and result formatting.
"""

from honcho import Honcho
from honcho_crewai import (
    HonchoDialecticTool,
    HonchoGetContextTool,
    HonchoSearchTool,
    create_dialectic_tool,
    create_get_context_tool,
    create_search_tool,
)


class TestGetContextTool:
    """Tests for HonchoGetContextTool."""

    def test_initialization(self):
        """Test that tool initializes with correct attributes."""
        honcho = Honcho()
        tool = HonchoGetContextTool(
            honcho=honcho, session_id="test_session", peer_id="test_peer"
        )

        assert tool is not None
        assert tool.name == "get_session_context"
        assert tool.description is not None
        assert tool.args_schema is not None

    def test_factory_function(self):
        """Test that factory function creates tool correctly."""
        honcho = Honcho()
        tool = create_get_context_tool(
            honcho=honcho, session_id="test_session", peer_id="test_peer"
        )

        assert isinstance(tool, HonchoGetContextTool)
        assert tool.name == "get_session_context"

    def test_returns_formatted_context(self):
        """Test that tool returns formatted context string."""
        honcho = Honcho()
        peer = honcho.peer("context_test_user")
        session_id = "context_test_session"
        session = honcho.session(session_id)

        # Add test message
        session.add_messages([peer.message("Test message for context")])

        # Create and execute tool
        tool = create_get_context_tool(
            honcho=honcho, session_id=session_id, peer_id="context_test_user"
        )
        result = tool._run()

        # Verify result is a formatted string
        assert isinstance(result, str)
        assert len(result) > 0


class TestDialecticTool:
    """Tests for HonchoDialecticTool."""

    def test_initialization(self):
        """Test that tool initializes with correct attributes."""
        honcho = Honcho()
        tool = HonchoDialecticTool(
            honcho=honcho, session_id="test_session", peer_id="test_peer"
        )

        assert tool is not None
        assert tool.name == "query_peer_knowledge"
        assert tool.description is not None

    def test_factory_function(self):
        """Test that factory function creates tool correctly."""
        honcho = Honcho()
        tool = create_dialectic_tool(
            honcho=honcho, session_id="test_session", peer_id="test_peer"
        )

        assert isinstance(tool, HonchoDialecticTool)
        assert tool.name == "query_peer_knowledge"

    def test_returns_response(self):
        """Test that tool returns a response string."""
        honcho = Honcho()
        peer = honcho.peer("dialectic_test_user")
        session_id = "dialectic_test_session"
        session = honcho.session(session_id)

        # Add test messages
        session.add_messages([peer.message("I love pizza and Italian food")])

        # Create and execute tool
        tool = create_dialectic_tool(
            honcho=honcho, session_id=session_id, peer_id="dialectic_test_user"
        )
        result = tool._run(query="What does the user like?")

        # Verify result is a string
        assert isinstance(result, str)
        assert len(result) > 0


class TestSearchTool:
    """Tests for HonchoSearchTool."""

    def test_initialization(self):
        """Test that tool initializes with correct attributes."""
        honcho = Honcho()
        tool = HonchoSearchTool(honcho=honcho, session_id="test_session")

        assert tool is not None
        assert tool.name == "search_session_messages"
        assert tool.description is not None

    def test_factory_function(self):
        """Test that factory function creates tool correctly."""
        honcho = Honcho()
        tool = create_search_tool(honcho=honcho, session_id="test_session")

        assert isinstance(tool, HonchoSearchTool)
        assert tool.name == "search_session_messages"

    def test_returns_formatted_results(self):
        """Test that tool returns formatted search results."""
        honcho = Honcho()
        peer = honcho.peer("search_test_user")
        session_id = "search_test_session"
        session = honcho.session(session_id)

        # Add test messages
        session.add_messages([peer.message("I love pizza and pasta")])

        # Create and execute tool
        tool = create_search_tool(honcho=honcho, session_id=session_id)
        result = tool._run(query="food", limit=5)

        # Verify result is a formatted string
        assert isinstance(result, str)
        assert len(result) > 0
        # Should have either results or "No messages found"
        assert "Search Results" in result or "No messages found" in result

    def test_search_with_filters(self):
        """Test that search tool accepts and uses filters parameter."""
        honcho = Honcho()
        peer = honcho.peer("search_filter_test_user")
        session_id = "search_filter_test_session"
        session = honcho.session(session_id)

        # Add test messages
        session.add_messages([peer.message("Important message about Python")])

        # Create and execute tool with filters
        tool = create_search_tool(honcho=honcho, session_id=session_id)
        result = tool._run(
            query="Python",
            limit=5,
            filters={"peer_id": peer.id}
        )

        # Verify result is a formatted string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_with_metadata_filters(self):
        """Test that search tool works with metadata filters."""
        honcho = Honcho()
        peer = honcho.peer("search_metadata_filter_user")
        session_id = "search_metadata_filter_session"
        session = honcho.session(session_id)

        # Add test messages with metadata
        session.add_messages([peer.message("High priority task", metadata={"priority": "high"})])

        # Create and execute tool with metadata filter
        tool = create_search_tool(honcho=honcho, session_id=session_id)
        result = tool._run(
            query="task",
            limit=5,
            filters={"metadata": {"priority": "high"}}
        )

        # Verify result is a formatted string
        assert isinstance(result, str)
        assert len(result) > 0


class TestToolsWorkTogether:
    """Test that all tools can work together."""

    def test_all_tools_in_same_session(self):
        """Test that all three tools can be used in the same session."""
        honcho = Honcho()
        peer = honcho.peer("combo_test_user")
        session_id = "combo_test_session"
        session = honcho.session(session_id)

        # Add messages
        session.add_messages([peer.message("I enjoy coding in Python")])

        # Create all tools
        context_tool = create_get_context_tool(
            honcho=honcho, session_id=session_id, peer_id="combo_test_user"
        )
        dialectic_tool = create_dialectic_tool(
            honcho=honcho, session_id=session_id, peer_id="combo_test_user"
        )
        search_tool = create_search_tool(honcho=honcho, session_id=session_id)

        # Execute all tools
        context_result = context_tool._run()
        dialectic_result = dialectic_tool._run(query="What does the user like?")
        search_result = search_tool._run(query="coding", limit=5)

        # Verify all return valid strings
        assert isinstance(context_result, str) and len(context_result) > 0
        assert isinstance(dialectic_result, str) and len(dialectic_result) > 0
        assert isinstance(search_result, str) and len(search_result) > 0
