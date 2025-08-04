from typing import Any


def get_work_unit_key(task_type: str, payload: dict[str, Any]) -> str:
    """
    Generate a work unit key for a given task type, workspace name, and event type.
    """
    if task_type == "representation" or task_type == "summary":
        print(f"Payload: {payload}")
        sender_name = payload["sender_name"] or "None"
        target_name = payload["target_name"] or "None"
        session_id = payload["session_name"] or "None"
        return f"{task_type}:{session_id}:{sender_name}:{target_name}"
    elif task_type == "webhook":
        workspace_name = payload["workspace_name"]
        if not workspace_name:
            raise ValueError("Workspace name is required for webhook tasks")
        return f"webhook:{workspace_name}"
    else:
        raise ValueError(f"Invalid task type: {task_type}")
