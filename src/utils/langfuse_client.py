"""
Centralized Langfuse client management.

This module provides a singleton Langfuse client to avoid multiple initialization
errors when modules import get_client() at the module level.
"""

from typing import Any

from langfuse import get_client

_langfuse_client: Any = None


def get_langfuse_client() -> Any:
    """
    Get the singleton Langfuse client instance.

    This function ensures that get_client() is only called once, regardless of
    how many modules import this function. This prevents multiple authentication
    error messages when LANGFUSE_PUBLIC_KEY is not configured.

    Returns:
        Any: The singleton Langfuse client instance
    """
    global _langfuse_client

    if _langfuse_client is None:
        _langfuse_client = get_client()

    return _langfuse_client


# For backward compatibility, provide the client as a module-level variable
# but only initialize it when first accessed
def __getattr__(name: str):
    """Lazy initialization of module-level 'lf' attribute."""
    if name == "lf":
        return get_langfuse_client()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
