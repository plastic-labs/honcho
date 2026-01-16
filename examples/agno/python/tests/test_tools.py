"""
Tests for Honcho Agno Tools

Simple tests verifying tool structure, registration, and initialization.
These tests use minimal mocking - just enough to avoid network calls.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from honcho_agno import HonchoTools


@pytest.fixture
def mock_client():
    """Create a minimal mock Honcho client."""
    client = MagicMock()
    client.peer.return_value = MagicMock()
    client.session.return_value = MagicMock()
    return client


class TestInitialization:
    """Tests for HonchoTools initialization."""

    def test_initializes_with_client(self, mock_client):
        """Test initialization with a provided client."""
        tools = HonchoTools(
            peer_id="assistant",
            session_id="test-session",
            honcho_client=mock_client,
        )

        assert tools.honcho is mock_client
        assert tools.peer_id == "assistant"
        assert tools.session_id == "test-session"

    def test_default_peer_id(self, mock_client):
        """Test default peer_id is 'assistant'."""
        tools = HonchoTools(honcho_client=mock_client)
        assert tools.peer_id == "assistant"

    def test_auto_generates_session_id(self, mock_client):
        """Test session_id is auto-generated if not provided."""
        tools = HonchoTools(honcho_client=mock_client)

        assert tools.session_id is not None
        # Should be valid UUID
        uuid.UUID(tools.session_id)

    def test_toolkit_name(self, mock_client):
        """Test toolkit has correct name."""
        tools = HonchoTools(honcho_client=mock_client)
        assert tools.name == "honcho"


class TestToolRegistration:
    """Tests verifying tools are properly registered."""

    def test_all_tools_registered(self, mock_client):
        """Test all expected tools are registered."""
        tools = HonchoTools(honcho_client=mock_client)

        registered = [func.name for func in tools.functions.values()]

        assert "honcho_get_context" in registered
        assert "honcho_search_messages" in registered
        assert "honcho_chat" in registered
        assert len(registered) == 3

    def test_tools_are_callable(self, mock_client):
        """Test tool methods exist and are callable."""
        tools = HonchoTools(honcho_client=mock_client)

        assert callable(tools.honcho_get_context)
        assert callable(tools.honcho_search_messages)
        assert callable(tools.honcho_chat)


class TestMultiPeerPattern:
    """Tests for multi-peer conversation patterns."""

    def test_multiple_toolkits_share_session(self, mock_client):
        """Test multiple toolkits can share a session ID."""
        session_id = "shared-session"

        agent1 = HonchoTools(
            peer_id="agent-alpha",
            session_id=session_id,
            honcho_client=mock_client,
        )

        agent2 = HonchoTools(
            peer_id="agent-beta",
            session_id=session_id,
            honcho_client=mock_client,
        )

        assert agent1.session_id == agent2.session_id
        assert agent1.peer_id != agent2.peer_id

    def test_each_toolkit_has_one_peer(self, mock_client):
        """Test each toolkit represents exactly one peer."""
        tools = HonchoTools(
            peer_id="my-agent",
            honcho_client=mock_client,
        )

        assert tools.peer is not None
        assert tools.peer_id == "my-agent"
        # Should not have multiple peer attributes
        assert not hasattr(tools, "user")
        assert not hasattr(tools, "assistant_peer")
