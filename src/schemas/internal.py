"""Internal schemas used by the deriver, dreamer, and other background systems.

These are not part of the public API contract and may change without notice.
"""

from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from src.schemas.api import MessageCreate
from src.schemas.configuration import SessionPeerConfig
from src.utils.formatting import parse_datetime_iso
from src.utils.types import DocumentLevel


class ReconcilerType(str, Enum):
    """Types of reconciler tasks that can be performed."""

    SYNC_VECTORS = "sync_vectors"
    CLEANUP_QUEUE = "cleanup_queue"


# ---------------------------------------------------------------------------
# Document / observation schemas (vector storage internals)
# ---------------------------------------------------------------------------


class DocumentBase(BaseModel):
    pass


class MemoryExpiry(BaseModel):
    type: Literal["none", "review", "date", "event"] = "none"
    review_at: str | None = None
    expires_at: str | None = None
    event_key: str | None = None

    @model_validator(mode="after")
    def validate_expiry_fields(self) -> Self:
        if self.type == "review" and not self.review_at:
            raise ValueError("review expiry requires review_at")
        if self.type == "date" and not self.expires_at:
            raise ValueError("date expiry requires expires_at")
        if self.type == "event" and not self.event_key:
            raise ValueError("event expiry requires event_key")
        return self


class MemoryLifecycle(BaseModel):
    review_due_at: str | None = None
    pending_event_key: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    superseded_by: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    supersedes: list[Annotated[str, Field(min_length=1, max_length=256)]] | None = None
    demote_after: str | None = None

    @field_validator(
        "review_due_at",
        "demote_after",
        mode="after",
    )
    @classmethod
    def validate_datetime_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_datetime_iso(value)
        return value

    @field_validator("supersedes", mode="after")
    @classmethod
    def validate_supersedes(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) == 0:
            raise ValueError("supersedes must contain at least one document id when provided")
        return value


class MemoryTaxonomy(BaseModel):
    domain: Annotated[str, Field(min_length=1, max_length=256)]
    horizon: Literal["short", "medium", "long"]
    thesis_kind: Literal["preference", "fact", "decision", "plan", "state", "rule"]
    expiry: MemoryExpiry = Field(default_factory=MemoryExpiry)
    confidence: Literal["low", "medium", "high"] | None = None
    lifecycle: MemoryLifecycle | None = None

    @field_validator("domain", mode="after")
    @classmethod
    def sanitize_domain(cls, v: str) -> str:
        return v.replace("\x00", "").strip()


class DocumentMetadata(BaseModel):
    message_ids: list[int] = Field(
        description="The ID range(s) of the messages that this document was derived from. Acts as a link to the primary source of the document. Note that as a document gets deduplicated, additional ranges will be added, because the same document could be derived from completely separate message ranges."
    )
    message_created_at: str = Field(
        description="The timestamp of the message that this document was derived from. Note that this is not the same as the created_at timestamp of the document. This timestamp is usually only saved with second-level precision."
    )
    source_ids: list[str] | None = Field(
        default=None,
        description="Document IDs of source documents for tree traversal -- required for deductive and inductive documents",
    )
    premises: list[str] | None = Field(
        default=None,
        description="Human-readable premise text for display -- only applicable for deductive documents",
    )
    sources: list[str] | None = Field(
        default=None,
        description="Human-readable source text for display -- only applicable for inductive documents",
    )
    pattern_type: str | None = Field(
        default=None,
        description="Type of pattern identified (preference, behavior, personality, tendency, correlation) -- only applicable for inductive documents",
    )
    confidence: str | None = Field(
        default=None,
        description="Confidence level (high, medium, low) -- only applicable for inductive documents",
    )
    memory: MemoryTaxonomy | None = Field(
        default=None,
        description="Optional taxonomy metadata for domain, retention horizon, expiry, and thesis type.",
    )


class DocumentCreate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    session_name: str | None = Field(
        default=None,
        description="The session from which the document was derived (NULL for global observations)",
    )
    level: DocumentLevel = Field(
        default="explicit",
        description="The level of the document (explicit, deductive, inductive, or contradiction)",
    )
    times_derived: int = Field(
        default=1,
        ge=1,
        description="The number of times that a semantic duplicate document to this one has been derived",
    )
    metadata: DocumentMetadata = Field()
    embedding: list[float] = Field()
    # Tree linkage field
    source_ids: list[str] | None = Field(
        default=None,
        description="Document IDs of source/premise documents -- for deductive and inductive documents",
    )


class ObservationInput(BaseModel):
    """Validated observation input from LLM tool calls."""

    content: Annotated[str, Field(min_length=1)]
    level: DocumentLevel = "explicit"
    source_ids: list[str] | None = None
    premises: list[str] | None = None
    sources: list[str] | None = None
    pattern_type: (
        Literal["preference", "behavior", "personality", "tendency", "correlation"]
        | None
    ) = None
    confidence: Literal["high", "medium", "low"] | None = None
    memory: MemoryTaxonomy | None = None

    @field_validator("content", mode="after")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return v.replace("\x00", "")

    @model_validator(mode="after")
    def validate_level_fields(self) -> Self:
        """Validate that level-specific fields are present when required."""
        if self.level == "deductive" and not self.source_ids:
            raise ValueError(
                "deductive observations require 'source_ids' field with document IDs of premises"
            )
        if self.level == "inductive" and not self.source_ids:
            raise ValueError(
                "inductive observations require 'source_ids' field with document IDs of sources"
            )
        if self.level == "contradiction" and (
            not self.source_ids or len(self.source_ids) < 2
        ):
            raise ValueError(
                "contradiction observations require 'source_ids' field with at least 2 IDs of contradicting observations"
            )
        return self


# ---------------------------------------------------------------------------
# Queue internals
# ---------------------------------------------------------------------------


class SessionCounts(BaseModel):
    """Counts for a specific session in queue processing."""

    completed: int
    in_progress: int
    pending: int


class QueueCounts(BaseModel):
    """Aggregated counts for queue processing status."""

    total: int
    completed: int
    in_progress: int
    pending: int
    sessions: dict[str, SessionCounts]


class QueueStatusRow(BaseModel):
    """Represents a row from the queue status SQL query result."""

    session_id: str | None
    total: int
    completed: int
    in_progress: int
    pending: int
    session_total: int
    session_completed: int
    session_in_progress: int
    session_pending: int


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------


class SessionPeerData(BaseModel):
    """Data for managing session peer relationships."""

    peer_names: dict[str, SessionPeerConfig]


class MessageBulkData(BaseModel):
    """Data for bulk message operations."""

    messages: list[MessageCreate]
    session_name: str
    workspace_name: str
