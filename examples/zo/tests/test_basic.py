"""Basic import and structure tests for honcho-zo-skill.

These tests validate package structure and imports without requiring
a running Honcho server.
"""

import sys
import os

# Add parent directory to path so tools/ can be imported
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


def test_tools_package_import():
    """Test that the tools package exports all three functions."""
    import tools

    assert hasattr(tools, "save_memory")
    assert hasattr(tools, "query_memory")
    assert hasattr(tools, "get_context")


def test_tools_all_exports():
    """Test that __all__ contains expected exports."""
    import tools

    assert hasattr(tools, "__all__")
    expected = ["save_memory", "query_memory", "get_context"]
    for name in expected:
        assert name in tools.__all__, f"{name} not in __all__"


def test_save_memory_raises_on_empty_content():
    """Test that save_memory raises ValueError for empty content."""
    import pytest
    from tools.save_memory import save_memory

    with pytest.raises(ValueError, match="content must not be empty"):
        save_memory("user1", "", "user", "session1")


def test_query_memory_raises_on_empty_query():
    """Test that query_memory raises ValueError for empty query."""
    import pytest
    from tools.query_memory import query_memory

    with pytest.raises(ValueError, match="query must not be empty"):
        query_memory("user1", "")
