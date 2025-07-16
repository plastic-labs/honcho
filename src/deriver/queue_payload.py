from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class DeriverQueuePayload(BaseModel):
    """
    Schema for validating queue payload data.

    sender_name: the peer who sent the message
    target_name: the peer who is observing the message -- if this is the same as the sender,
                   this is a global ("honcho-level") representation task
    """

    content: str
    workspace_name: str
    sender_name: str
    target_name: str
    session_name: str | None
    message_id: int
    created_at: datetime
    task_type: Literal["representation", "summary"]

    model_config = ConfigDict(extra="forbid")  # pyright: ignore

    @classmethod
    def create_payload(
        cls,
        message: dict[str, Any],
        sender_name: str,
        target_name: str,
        task_type: Literal["representation", "summary"],
    ) -> dict[str, Any]:
        """
        Create a processed payload from a message for queue processing.

        Args:
            message: The original message dictionary
            sender_name: Name of the message sender
            target_name: Name of the observer peer
            task_type: Type of task ('representation' or 'summary')

        Returns:
            Processed payload dictionary ready for queue processing

        Raises:
            ValueError: If the payload doesn't match the expected schema
        """
        # Validate required fields and types
        if not isinstance(message.get("content"), str):
            raise TypeError("Message content must be a string")

        if not isinstance(message.get("workspace_name"), str):
            raise TypeError("Workspace name must be a string")

        # Ensure message_id is an integer
        message_id = message.get("message_id")
        if not isinstance(message_id, int):
            raise TypeError("Message ID must be an integer")
        
        # Ensure created_at exists and is a datetime
        if "created_at" not in message:
            raise TypeError("created_at is required")
        if not isinstance(message["created_at"], datetime):
            raise TypeError("created_at must be a datetime object")

        # Create the processed payload with properly typed fields
        content: str = message["content"]
        workspace_name: str = message["workspace_name"]
        session_name: str | None = message.get("session_name")
        created_at: datetime = message["created_at"]

        # Create and validate the payload using the schema
        try:
            validated_payload = DeriverQueuePayload(
                content=content,
                workspace_name=workspace_name,
                sender_name=sender_name,
                target_name=target_name,
                session_name=session_name,
                message_id=message_id,
                created_at=created_at,
                task_type=task_type,
            )
            # Convert back to dict for compatibility with JSON serialization
            # mode='json' ensures datetime is converted to ISO string
            payload = validated_payload.model_dump(mode='json')
        except Exception as e:
            raise ValueError(f"Failed to create valid payload: {str(e)}") from e

        return payload
