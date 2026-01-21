from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.schemas import DreamType, ReconcilerType, ResolvedConfiguration


class BasePayload(BaseModel):
    """Base payload with common fields."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]


class RepresentationPayload(BasePayload):
    """Payload for representation tasks."""

    task_type: Literal["representation"] = "representation"
    session_name: str
    content: str
    observers: list[str]
    observed: str
    created_at: datetime
    configuration: ResolvedConfiguration


class RepresentationPayloads(BasePayload):
    """Payload for a batch of representation tasks."""

    payloads: list[RepresentationPayload]


class SummaryPayload(BasePayload):
    """Payload for summary tasks."""

    task_type: Literal["summary"] = "summary"
    session_name: str
    message_seq_in_session: int
    configuration: ResolvedConfiguration
    # Optional for backward compatibility with older queue items
    message_public_id: str | None = None


class WebhookPayload(BasePayload):
    """Payload for webhook delivery tasks."""

    task_type: Literal["webhook"] = "webhook"
    event_type: str
    data: dict[str, Any]


class DreamPayload(BasePayload):
    """Payload for dream tasks."""

    task_type: Literal["dream"] = "dream"
    dream_type: DreamType
    observer: str
    observed: str
    session_name: str


class DeletionPayload(BasePayload):
    """Payload for deletion tasks."""

    task_type: Literal["deletion"] = "deletion"
    deletion_type: Literal["session", "observation"]
    resource_id: str


class ReconcilerPayload(BasePayload):
    """Payload for reconciler tasks (vector sync, queue cleanup, self-healing)."""

    task_type: Literal["reconciler"] = "reconciler"
    reconciler_type: ReconcilerType


def create_webhook_payload(
    event_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Create a webhook payload."""
    return WebhookPayload(event_type=event_type, data=data).model_dump(
        mode="json", exclude_none=True
    )


def create_dream_payload(
    dream_type: DreamType,
    *,
    observer: str,
    observed: str,
    session_name: str,
) -> dict[str, Any]:
    """Create a dream payload."""
    return DreamPayload(
        dream_type=dream_type,
        observer=observer,
        observed=observed,
        session_name=session_name,
    ).model_dump(mode="json", exclude_none=True)


def create_deletion_payload(
    deletion_type: Literal["session", "observation"],
    resource_id: str,
) -> dict[str, Any]:
    """Create a deletion payload."""
    return DeletionPayload(
        deletion_type=deletion_type,
        resource_id=resource_id,
    ).model_dump(mode="json", exclude_none=True)


def create_payload(
    message: dict[str, Any],
    configuration: ResolvedConfiguration,
    task_type: Literal["representation", "summary"],
    message_seq_in_session: int | None = None,
    *,
    observers: list[str] | None = None,
    observed: str | None = None,
) -> dict[str, Any]:
    """
    Create a processed payload from a message for queue processing.

    Note: workspace_name and message_id are no longer included in the returned payload
    as they are now stored in dedicated columns on the queue table. The caller is
    responsible for extracting and passing these values separately.

    Args:
        message: The original message dictionary
        task_type: Type of task ('representation' or 'summary')
        message_seq_in_session: Required for summary tasks, must be None for representation
        observers: List of observer peer names (required for representation tasks)
        observed: Name of the observed peer (*always* the peer who sent the message) (required for representation tasks)


    Returns:
        Processed payload dictionary ready for queue processing (without workspace_name and message_id)

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

            if observers is None or len(observers) == 0:
                raise ValueError("observers is required for representation tasks")

            if observed is None:
                raise ValueError("observed is required for representation tasks")

            validated_payload = RepresentationPayload(
                content=content,
                session_name=session_name,
                created_at=created_at,
                observers=observers,
                observed=observed,
                configuration=configuration,
            )
        elif task_type == "summary":
            if message_seq_in_session is None:
                raise ValueError("message_seq_in_session is required for summary tasks")
            message_public_id = message.get("message_public_id")
            if message_public_id is not None and (
                not isinstance(message_public_id, str) or not message_public_id.strip()
            ):
                raise ValueError(
                    "message_public_id must be a non-empty string if provided"
                )

            validated_payload = SummaryPayload(
                session_name=session_name,
                message_seq_in_session=message_seq_in_session,
                configuration=configuration,
                message_public_id=message_public_id,
            )

        # Convert back to dict for compatibility with JSON serialization
        # mode='json' ensures datetime is converted to ISO string
        payload = validated_payload.model_dump(mode="json", exclude_none=True)

    except Exception as e:
        raise ValueError(f"Failed to create valid payload: {str(e)}") from e

    return payload
