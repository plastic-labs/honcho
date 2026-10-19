"""
Reconciliation events for Honcho telemetry.

Reconciliation tasks handle maintenance operations:
- Sync vectors: Synchronize documents and message embeddings to external vector stores
- Cleanup stale items: Clean up soft-deleted records and expired queue items
- Backfill: Copy legacy data into a new shape
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class SyncVectorsCompletedEvent(BaseEvent):
    """Emitted when a vector sync cycle completes.

    Vector sync tasks synchronize documents and message embeddings to external
    vector stores. These run periodically and operate across all workspaces.

    Note: This event has no workspace context as it operates globally.
    """

    _event_type: ClassVar[str] = "reconciliation.sync_vectors.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "reconciliation"

    # Document metrics
    documents_synced: int = Field(
        default=0, description="Documents successfully synced to vector store"
    )
    documents_failed: int = Field(
        default=0, description="Documents that failed to sync"
    )
    documents_cleaned: int = Field(
        default=0, description="Soft-deleted documents cleaned up during sync"
    )

    # Message embedding metrics
    message_embeddings_synced: int = Field(
        default=0, description="Message embeddings successfully synced"
    )
    message_embeddings_failed: int = Field(
        default=0, description="Message embeddings that failed to sync"
    )

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    def get_resource_id(self) -> str:
        """Resource ID is fixed for this global operation."""
        return "sync_vectors"


class CleanupStaleItemsCompletedEvent(BaseEvent):
    """Emitted when a stale items cleanup cycle completes.

    Cleanup tasks remove soft-deleted documents and expired queue items.
    These run periodically and operate across all workspaces.

    Note: This event has no workspace context as it operates globally.
    """

    _event_type: ClassVar[str] = "reconciliation.cleanup_stale_items.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "reconciliation"

    # Cleanup metrics
    documents_cleaned: int = Field(
        default=0, description="Soft-deleted documents cleaned up"
    )
    queue_items_cleaned: int = Field(
        default=0, description="Expired queue items cleaned up"
    )

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    def get_resource_id(self) -> str:
        """Resource ID is fixed for this global operation."""
        return "cleanup_stale_items"


class BackfillCompletedEvent(BaseEvent):
    """Emitted when one backfill cycle completes.

    Note: This event has no workspace context as it operates globally.
    """

    _event_type: ClassVar[str] = "reconciliation.backfill.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "reconciliation"

    backfill_name: str = Field(..., description="Registered backfill name")
    rows_touched: int = Field(default=0, description="Rows moved to the new shape")
    batches: int = Field(default=0, description="Batches run in this cycle")
    still_pending: bool = Field(
        ..., description="Whether rows still match the pending predicate"
    )
    total_duration_ms: float = Field(..., description="Total processing time")

    def get_resource_id(self) -> str:
        """Resource ID is the backfill name."""
        return f"backfill:{self.backfill_name}"


__all__ = [
    "SyncVectorsCompletedEvent",
    "CleanupStaleItemsCompletedEvent",
    "BackfillCompletedEvent",
]
