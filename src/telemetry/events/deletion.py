"""
Deletion events for Honcho telemetry.

Deletion tasks handle async removal of resources like sessions and observations,
ensuring proper cleanup across all storage layers.
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class DeletionCompletedEvent(BaseEvent):
    """Emitted when a deletion task completes.

    Deletion tasks handle async removal of sessions and observations,
    ensuring proper cleanup across all storage layers.
    """

    _event_type: ClassVar[str] = "deletion.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "deletion"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Deletion details
    deletion_type: str = Field(
        ..., description="Type of deletion: 'session' or 'observation'"
    )
    resource_id: str = Field(..., description="ID of the deleted resource")

    # Outcome
    success: bool = Field(..., description="Whether deletion succeeded")

    # Optional error info
    error_message: str | None = Field(
        default=None, description="Error message if deletion failed"
    )

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, type, and resource for uniqueness."""
        return f"{self.workspace_id}:{self.deletion_type}:{self.resource_id}"


__all__ = ["DeletionCompletedEvent"]
