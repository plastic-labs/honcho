"""Basic import and structure tests for the OpenAI Agents SDK integration.

These tests validate package structure and tool signatures without requiring
a running Honcho server or any API keys.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_save_memory_import():
    """Test that save_memory can be imported."""
    from tools.save_memory import save_memory

    assert callable(save_memory)


def test_query_memory_import():
    """Test that query_memory can be imported."""
    from tools.query_memory import query_memory

    assert callable(query_memory)


def test_get_context_import():
    """Test that get_context can be imported."""
    from tools.get_context import get_context

    assert callable(get_context)


def test_tools_package_exports():
    """Test that the tools package exports all three public symbols."""
    import tools

    assert hasattr(tools, "save_memory")
    assert hasattr(tools, "query_memory")
    assert hasattr(tools, "get_context")


def test_tools_all_list():
    """Test that __all__ contains the expected exports."""
    import tools

    assert hasattr(tools, "__all__")
    for name in ("get_context", "query_memory", "save_memory"):
        assert name in tools.__all__, f"{name!r} missing from __all__"


def test_honcho_context_dataclass():
    """Test that HonchoContext can be instantiated with required fields."""
    from tools.client import HonchoContext

    ctx = HonchoContext(user_id="alice", session_id="session-1")
    assert ctx.user_id == "alice"
    assert ctx.session_id == "session-1"
    assert ctx.assistant_id == "assistant"


def test_honcho_context_custom_assistant_id():
    """Test that HonchoContext accepts a custom assistant_id."""
    from tools.client import HonchoContext

    ctx = HonchoContext(user_id="alice", session_id="s1", assistant_id="my-bot")
    assert ctx.assistant_id == "my-bot"


def test_save_memory_raises_on_empty_content():
    """Test that save_memory raises ValueError for empty content."""
    from tools.save_memory import save_memory

    with pytest.raises(ValueError, match="content must not be empty"):
        save_memory("user1", "", "user", "session1")


def test_query_memory_is_function_tool():
    """Test that query_memory is decorated as an OpenAI Agents function tool."""
    from agents import FunctionTool

    from tools.query_memory import query_memory

    assert isinstance(query_memory, FunctionTool)


def test_main_module_imports():
    """Test that main.py can be imported and exposes the agent and chat function."""
    import main

    assert hasattr(main, "honcho_agent")
    assert hasattr(main, "chat")
    assert callable(main.chat)
