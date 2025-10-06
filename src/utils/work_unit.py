"""Work unit utility functions for generating and parsing work unit keys."""

import tiktoken
from typing_extensions import Any, TypedDict

tokenizer = tiktoken.get_encoding("cl100k_base")


class ParsedWorkUnit(TypedDict):
    """Parsed work unit components."""

    task_type: str
    workspace_name: str
    session_name: str | None
    observer: str | None
    observed: str | None


def get_work_unit_key(payload: dict[str, Any] | ParsedWorkUnit) -> str:
    """
    Generate a work unit key for a given task type, workspace name, and event type.

    Args:
        payload: Dictionary containing work unit information

    Returns:
        Formatted work unit key string

    Raises:
        ValueError: If required fields are missing or task type is invalid
    """
    workspace_name = payload.get("workspace_name")
    task_type = payload.get("task_type")
    if not workspace_name or not task_type:
        raise ValueError(
            "workspace_name and task_type are required to generate a work_unit_key"
        )

    if task_type in ["representation", "summary", "dream"]:
        observer = payload.get("observer", "None")
        observed = payload.get("observed", "None")
        session_name = payload.get("session_name", "None")
        if task_type == "dream":
            return f"{task_type}:{workspace_name}:{observer}:{observed}"
        return f"{task_type}:{workspace_name}:{session_name}:{observer}:{observed}"

    if task_type == "webhook":
        return f"webhook:{workspace_name}"

    raise ValueError(f"Invalid task type: {task_type}")


def parse_work_unit_key(work_unit_key: str) -> ParsedWorkUnit:
    """
    Parse a work unit key to extract its components.

    Args:
        work_unit_key: The work unit key string to parse

    Returns:
        ParsedWorkUnit dictionary with extracted components

    Raises:
        ValueError: If the work unit key format is invalid
    """
    parts = work_unit_key.split(":")
    task_type = parts[0]

    if task_type in ["representation", "summary"]:
        if len(parts) != 5:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return {
            "task_type": task_type,
            "workspace_name": parts[1],
            "session_name": parts[2],
            "observer": parts[3],
            "observed": parts[4],
        }

    if task_type == "dream":
        if len(parts) != 4:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return {
            "task_type": task_type,
            "workspace_name": parts[1],
            "session_name": None,
            "observer": parts[2],
            "observed": parts[3],
        }

    if task_type == "webhook":
        if len(parts) != 2:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return {
            "task_type": task_type,
            "workspace_name": parts[1],
            "session_name": None,
            "observer": None,
            "observed": None,
        }

    raise ValueError(f"Invalid task type in work_unit_key: {task_type}")


def estimate_tokens(text: str | list[str] | None) -> int:
    """Estimate token count using tiktoken for text or list of strings."""
    if not text:
        return 0
    if isinstance(text, list):
        text = "\n".join(text)
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text) // 4
