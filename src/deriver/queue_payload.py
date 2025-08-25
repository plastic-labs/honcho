from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class BasePayload(BaseModel):
    """Base payload with common fields."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]


class RepresentationPayload(BasePayload):
    """Payload for representation tasks."""

    task_type: Literal["representation"] = "representation"
    workspace_name: str
    session_name: str
    message_id: int
    content: str
    sender_name: str
    target_name: str
    created_at: datetime


class SummaryPayload(BasePayload):
    """Payload for summary tasks."""

    task_type: Literal["summary"] = "summary"
    workspace_name: str
    session_name: str
    message_id: int
    message_seq_in_session: int


class WebhookPayload(BasePayload):
    """Payload for webhook delivery tasks."""

    task_type: Literal["webhook"] = "webhook"
    workspace_name: str
    event_type: str
    data: dict[str, Any]


def create_webhook_payload(
    workspace_name: str,
    event_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    return WebhookPayload(
        workspace_name=workspace_name, event_type=event_type, data=data
    ).model_dump(mode="json")


def create_payload(
    message: dict[str, Any],
    task_type: Literal["representation", "summary"],
    sender_name: str | None = None,
    target_name: str | None = None,
    message_seq_in_session: int | None = None,
) -> dict[str, Any]:
    """
    Create a processed payload from a message for queue processing.

    Args:
        message: The original message dictionary
        task_type: Type of task ('representation' or 'summary')
        sender_name: Name of the message sender (required for representation tasks)
        target_name: Name of the observer peer (required for representation tasks)
        message_seq_in_session: Required for summary tasks, must be None for representation

    Returns:
        Processed payload dictionary ready for queue processing

    Raises:
        ValueError: If the payload doesn't match the expected schema
    """
    workspace_name = message.get("workspace_name")
    session_name = message.get("session_name")
    message_id = message.get("message_id")

    if not isinstance(workspace_name, str):
        raise TypeError("Workspace name must be a string")

    if not isinstance(session_name, str):
        raise TypeError("Session name must be a string")

    if not isinstance(message_id, int):
        raise TypeError("Message ID must be an integer")

    # Create the appropriate payload type based on task_type
    try:
        if task_type == "representation":
            content = message.get("content")
            created_at = message.get("created_at")

            if not isinstance(content, str):
                raise TypeError("Message content must be a string")

            if not isinstance(created_at, datetime):
                raise TypeError("created_at must be a datetime object")

            if sender_name is None:
                raise ValueError("sender_name is required for representation tasks")

            if target_name is None:
                raise ValueError("target_name is required for representation tasks")

            validated_payload = RepresentationPayload(
                content=content,
                workspace_name=workspace_name,
                sender_name=sender_name,
                target_name=target_name,
                session_name=session_name,
                message_id=message_id,
                created_at=created_at,
            )
        elif task_type == "summary":
            if message_seq_in_session is None:
                raise ValueError("message_seq_in_session is required for summary tasks")

            validated_payload = SummaryPayload(
                workspace_name=workspace_name,
                session_name=session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
            )

        # Convert back to dict for compatibility with JSON serialization
        # mode='json' ensures datetime is converted to ISO string
        payload = validated_payload.model_dump(mode="json")

    except Exception as e:
        raise ValueError(f"Failed to create valid payload: {str(e)}") from e

    return payload
