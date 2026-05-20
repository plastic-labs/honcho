"""
Base event classes and idempotency key generation for CloudEvents telemetry.

This module provides:
- BaseEvent: Abstract base class for all Honcho telemetry events
- generate_event_id(): Deterministic event ID generation for idempotency
"""

import base64
import hashlib
from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel, Field


def generate_event_id(
    event_type: str,
    timestamp: datetime,
    resource_id: str,
    honcho_version: str | None = None,
) -> str:
    """Generate a deterministic event ID for idempotency.

    Same inputs always produce the same ID, so retries are automatically
    deduplicated on the receiving end. `honcho_version` is folded into the
    payload so two deploys emitting the same logical event produce distinct
    IDs — protects against silently merging events whose payload shape may
    have shifted between versions.

    Args:
        event_type: The CloudEvents type (e.g., "honcho.work.representation.completed")
        timestamp: When the event occurred
        resource_id: A unique identifier for the resource (can include workspace_id if relevant)
        honcho_version: The honcho package version emitting the event;
            included in the hash so cross-deploy events don't dedupe.

    Returns:
        A deterministic event ID in the format "evt_{base64_hash}"
    """
    version_segment = honcho_version or ""
    payload = f"{event_type}:{resource_id}:{timestamp.isoformat()}:{version_segment}"
    hash_bytes = hashlib.sha256(payload.encode()).digest()[:16]
    # Use URL-safe base64 encoding, strip padding
    encoded = base64.urlsafe_b64encode(hash_bytes).decode().rstrip("=")
    return f"evt_{encoded}"


class BaseEvent(BaseModel):
    """Base class for all Honcho telemetry events.

    All events inherit from this class and define:
    - _event_type: The CloudEvents type string
    - _schema_version: Integer version for schema evolution
    - _category: Event category (work, activity, resource)

    Subclasses must implement the abstract class methods by setting class variables
    and providing get_resource_id(). Subclasses define their own context fields
    (e.g., workspace_id, workspace_name) as needed - not all events have workspace context.
    """

    # Class variables for event metadata (set by subclasses)
    _event_type: ClassVar[str]
    _schema_version: ClassVar[int]
    _category: ClassVar[str]  # "work", "activity", or "resource"

    # Volume class for sampling decisions:
    # - "ground_truth": always emitted at rate 1.0 (aggregates, calibration keys)
    # - "high_volume": subject to TELEMETRY.HIGH_VOLUME_SAMPLE_RATE
    # Default is ground_truth so existing events keep firing unconditionally.
    _volume_class: ClassVar[str] = "ground_truth"

    # Common timestamp field present in all events
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the event occurred (UTC)",
    )

    @classmethod
    def event_type(cls) -> str:
        """Return the CloudEvents type string for this event."""
        return cls._event_type

    @classmethod
    def schema_version(cls) -> int:
        """Return the schema version for this event."""
        return cls._schema_version

    @classmethod
    def category(cls) -> str:
        """Return the event category (work, activity, or resource)."""
        return cls._category

    @classmethod
    def volume_class(cls) -> str:
        """Return the volume class for sampling: 'ground_truth' or 'high_volume'."""
        return cls._volume_class

    def get_resource_id(self) -> str:
        """Return the resource ID for idempotency key generation.

        This should be a unique identifier for the specific operation,
        such as a task_id or a hash of message IDs.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_resource_id()")

    def generate_id(self) -> str:
        """Generate a deterministic event ID for this event instance.

        Folds in the honcho package version so the same logical event from
        two different deploys produces distinct ids — downstream dedupe by
        id won't silently merge events whose body shape may have shifted.
        """
        from src._version import HONCHO_VERSION

        return generate_event_id(
            event_type=self.event_type(),
            timestamp=self.timestamp,
            resource_id=self.get_resource_id(),
            honcho_version=HONCHO_VERSION,
        )
