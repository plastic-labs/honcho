from typing_extensions import Any, TypedDict


class ParsedWorkUnit(TypedDict):
    task_type: str
    workspace_name: str
    session_name: str | None
    sender_name: str | None
    target_name: str | None


def get_work_unit_key(task_type: str, payload: dict[str, Any]) -> str:
    """
    Generate a work unit key for a given task type, workspace name, and event type.
    """
    workspace_name = payload.get("workspace_name")
    if not workspace_name:
        raise ValueError("workspace_name is required to generate a work_unit_key")

    if task_type in ["representation", "summary"]:
        sender_name = payload.get("sender_name", "None")
        target_name = payload.get("target_name", "None")
        session_name = payload.get("session_name", "None")
        return (
            f"{task_type}:{workspace_name}:{session_name}:{sender_name}:{target_name}"
        )

    if task_type == "webhook":
        return f"webhook:{workspace_name}"

    raise ValueError(f"Invalid task type: {task_type}")


def parse_work_unit_key(work_unit_key: str) -> ParsedWorkUnit:
    """
    Parse a work unit key to extract its components.
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
            "sender_name": parts[3],
            "target_name": parts[4],
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
            "sender_name": None,
            "target_name": None,
        }

    raise ValueError(f"Invalid task type in work_unit_key: {task_type}")
