"""
Deletion events for Honcho telemetry.

Deletion tasks handle async removal of resources like workspaces, sessions,
and conclusions, ensuring proper cleanup across all storage layers.
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class DeletionCompletedEvent(BaseEvent):
    """Emitted when a deletion task completes.

    Deletion tasks handle async removal of workspaces, sessions, and conclusions,
    ensuring proper cleanup across all storage layers. For workspace deletions,
    cascade counts track how many child resources were also deleted.
    """

    _event_type: ClassVar[str] = "deletion.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "deletion"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Deletion details
    deletion_type: str = Field(
        ..., description="Type of deletion: 'workspace', 'session', or 'conclusions'"
    )
    resource_id: str = Field(..., description="ID of the deleted resource")

    # Outcome
    success: bool = Field(..., description="Whether deletion succeeded")

    # Cascade counts (populated for workspace deletion)
    peers_deleted: int = Field(
        default=0, description="Number of peers deleted (workspace deletion)"
    )
    sessions_deleted: int = Field(
        default=0, description="Number of sessions deleted (workspace deletion)"
    )
    messages_deleted: int = Field(
        default=0, description="Number of messages deleted (workspace deletion)"
    )
    conclusions_deleted: int = Field(
        default=0, description="Number of conclusions deleted (workspace deletion)"
    )

    # Optional error info
    error_message: str | None = Field(
        default=None, description="Error message if deletion failed"
    )

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, type, and resource for uniqueness."""
        return f"{self.workspace_id}:{self.deletion_type}:{self.resource_id}"


__all__ = ["DeletionCompletedEvent"]
