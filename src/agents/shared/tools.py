"""
Shared tool definitions and utilities for Honcho agents.

This module provides common tool-related functionality that can be
used across multiple agents.
"""

from typing import Any


def create_tool_definition(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized tool definition for LLM tool calling.

    Args:
        name: Tool name (snake_case)
        description: Clear description of what the tool does
        parameters: Dictionary defining the parameters and their types
        required: List of required parameter names

    Returns:
        Tool definition in the format expected by LLM APIs

    Example:
        >>> create_tool_definition(
        ...     name="search_premises",
        ...     description="Search for premises matching a query",
        ...     parameters={
        ...         "query": {"type": "string", "description": "Search query"},
        ...         "limit": {"type": "integer", "description": "Max results"},
        ...     },
        ...     required=["query"],
        ... )
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required or [],
            },
        },
    }


def validate_tool_call(
    tool_call: dict[str, Any],
    expected_tools: list[str],
) -> bool:
    """
    Validate that a tool call is well-formed and uses a known tool.

    Args:
        tool_call: The tool call to validate
        expected_tools: List of valid tool names

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(tool_call, dict):
        return False

    if "function" not in tool_call:
        return False

    function = tool_call["function"]
    if not isinstance(function, dict):
        return False

    if "name" not in function:
        return False

    return function["name"] in expected_tools


def extract_tool_arguments(
    tool_call: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract arguments from a tool call.

    Args:
        tool_call: Tool call dictionary

    Returns:
        Dictionary of arguments

    Raises:
        ValueError: If tool call is malformed
    """
    try:
        return tool_call["function"]["arguments"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Malformed tool call: {e}")


def format_tool_result(
    tool_name: str,
    result: Any,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Format a tool execution result.

    Args:
        tool_name: Name of the tool that was called
        result: The result from the tool execution
        error: Optional error message if execution failed

    Returns:
        Formatted tool result
    """
    if error:
        return {
            "tool": tool_name,
            "success": False,
            "error": error,
        }

    return {
        "tool": tool_name,
        "success": True,
        "result": result,
    }
