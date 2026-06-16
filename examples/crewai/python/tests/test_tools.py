"""
Tests for Honcho CrewAI tools.
"""

from honcho_crewai import (
    HonchoDialecticTool,
    HonchoGetContextTool,
    HonchoSearchTool,
)
from test_storage import FakeHoncho


class TestGetContextTool:
    def test_initialization_is_lazy(self):
        honcho = FakeHoncho()

        tool = HonchoGetContextTool(
            honcho=honcho,
            session_id="session-1",
            peer_id="user-1",
        )

        assert tool.name == "get_session_context"
        assert honcho.session_calls == []

    def test_returns_formatted_context(self):
        honcho = FakeHoncho()
        peer = honcho.peer("user-1")
        session = honcho.session("session-1")
        session.add_messages([peer.message("Test message for context")])

        honcho.session_calls.clear()
        tool = HonchoGetContextTool(
            honcho=honcho,
            session_id="session-1",
            peer_id="user-1",
        )
        result = tool._run()

        assert "Messages (1)" in result
        assert "user-1: Test message for context" in result
        assert honcho.session_calls == ["session-1"]


class TestDialecticTool:
    def test_initialization_is_lazy(self):
        honcho = FakeHoncho()

        tool = HonchoDialecticTool(
            honcho=honcho,
            session_id="session-1",
            peer_id="user-1",
        )

        assert tool.name == "query_peer_knowledge"
        assert honcho.peer_calls == []

    def test_returns_response(self):
        honcho = FakeHoncho()
        tool = HonchoDialecticTool(
            honcho=honcho,
            session_id="session-1",
            peer_id="user-1",
        )

        result = tool._run(query="What does the user like?")

        assert "answer: What does the user like?" in result
        assert honcho.peer_calls == ["user-1"]


class TestSearchTool:
    def test_initialization_is_lazy(self):
        honcho = FakeHoncho()

        tool = HonchoSearchTool(honcho=honcho, session_id="session-1")

        assert tool.name == "search_session_messages"
        assert honcho.session_calls == []

    def test_returns_formatted_results(self):
        honcho = FakeHoncho()
        peer = honcho.peer("user-1")
        session = honcho.session("session-1")
        session.add_messages([peer.message("I love pizza and pasta")])

        honcho.session_calls.clear()
        tool = HonchoSearchTool(honcho=honcho, session_id="session-1")
        result = tool._run(query="food", limit=5)

        assert "Search Results" in result
        assert "[user-1] I love pizza and pasta" in result
        assert honcho.session_calls == ["session-1"]


class TestToolsWorkTogether:
    def test_all_tools_in_same_session(self):
        honcho = FakeHoncho()
        peer = honcho.peer("user-1")
        session = honcho.session("session-1")
        session.add_messages([peer.message("I enjoy coding in Python")])

        context_tool = HonchoGetContextTool(
            honcho=honcho,
            session_id="session-1",
            peer_id="user-1",
        )
        dialectic_tool = HonchoDialecticTool(
            honcho=honcho,
            session_id="session-1",
            peer_id="user-1",
        )
        search_tool = HonchoSearchTool(honcho=honcho, session_id="session-1")

        assert context_tool._run()
        assert dialectic_tool._run(query="What does the user like?")
        assert search_tool._run(query="coding", limit=5)
