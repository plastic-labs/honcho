"""API types for Honcho SDK.

These types mirror the server's Pydantic schemas for API responses and requests.
"""

from __future__ import annotations

import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ==============================================================================
# Configuration Types
# ==============================================================================


class ReasoningConfiguration(BaseModel):
    """Configuration for reasoning functionality."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    enabled: bool | None = None
    custom_instructions: str | None = None


class PeerCardConfiguration(BaseModel):
    """Configuration for peer card functionality."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    use: bool | None = None
    create: bool | None = None


class SummaryConfiguration(BaseModel):
    """Configuration for summary functionality."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    enabled: bool | None = None
    messages_per_short_summary: int | None = None
    messages_per_long_summary: int | None = None


class DreamConfiguration(BaseModel):
    """Configuration for dream functionality."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    enabled: bool | None = None


class WorkspaceConfiguration(BaseModel):
    """Workspace-level configuration options."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    reasoning: ReasoningConfiguration | None = None
    peer_card: PeerCardConfiguration | None = None
    summary: SummaryConfiguration | None = None
    dream: DreamConfiguration | None = None


class SessionConfiguration(WorkspaceConfiguration):
    """Session-level configuration options."""

    pass


class WorkspaceConfigurationResponse(WorkspaceConfiguration):
    """Workspace configuration for response parsing — tolerates unknown fields from newer servers."""

    model_config = ConfigDict(extra="ignore")  # pyright: ignore[reportUnannotatedClassAttribute]


class SessionConfigurationResponse(SessionConfiguration):
    """Session configuration for response parsing — tolerates unknown fields from newer servers."""

    model_config = ConfigDict(extra="ignore")  # pyright: ignore[reportUnannotatedClassAttribute]


class MessageConfiguration(BaseModel):
    """Message-level configuration options."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    reasoning: ReasoningConfiguration | None = None


# ==============================================================================
# Peer Config Types
# ==============================================================================


class PeerConfig(BaseModel):
    """Configuration for peer-level settings."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    observe_me: bool | None = None
    """Whether Honcho will use reasoning to form a representation of this peer."""


class SessionPeerConfig(BaseModel):
    """Configuration for a peer within a session."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    observe_others: bool | None = Field(
        None,
        description="Whether this peer should form a session-level theory-of-mind representation of other peers in the session",
    )
    observe_me: bool | None = Field(
        None,
        description="Whether other peers in this session should try to form a session-level theory-of-mind representation of this peer",
    )


# ==============================================================================
# Workspace Types
# ==============================================================================


class WorkspaceResponse(BaseModel):
    """Workspace API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: WorkspaceConfigurationResponse = Field(
        default_factory=WorkspaceConfigurationResponse
    )
    created_at: datetime.datetime


class WorkspaceCreateParams(BaseModel):
    """Parameters for creating a workspace."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str = Field(min_length=1, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: WorkspaceConfiguration = Field(
        default_factory=WorkspaceConfiguration
    )


class WorkspaceUpdateParams(BaseModel):
    """Parameters for updating a workspace."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    metadata: dict[str, Any] | None = None
    configuration: WorkspaceConfiguration | None = None


class WorkspaceListParams(BaseModel):
    """Parameters for listing workspaces."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


# ==============================================================================
# Peer Types
# ==============================================================================


class PeerResponse(BaseModel):
    """Peer API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    workspace_id: str
    created_at: datetime.datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: PeerConfig = Field(default_factory=PeerConfig)


class PeerCreateParams(BaseModel):
    """Parameters for creating a peer."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str = Field(min_length=1, max_length=100)
    metadata: dict[str, Any] | None = None
    configuration: PeerConfig | None = None


class PeerUpdateParams(BaseModel):
    """Parameters for updating a peer."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    metadata: dict[str, Any] | None = None
    configuration: PeerConfig | None = None


class PeerListParams(BaseModel):
    """Parameters for listing peers."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


class PeerRepresentationParams(BaseModel):
    """Parameters for getting peer representation."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    session_id: str | None = None
    target: str | None = None
    search_query: str | None = None
    search_top_k: int | None = Field(default=None, ge=1, le=100)
    search_max_distance: float | None = Field(default=None, ge=0.0, le=1.0)
    include_most_frequent: bool | None = None
    max_conclusions: int | None = Field(default=25, ge=1, le=100)


class RepresentationResponse(BaseModel):
    """Representation API response."""

    representation: str


class PeerCardResponse(BaseModel):
    """Peer card API response."""

    peer_card: list[str] | None = None


class PeerContextResponse(BaseModel):
    """Peer context API response."""

    peer_id: str
    target_id: str
    representation: str | None = None
    peer_card: list[str] | None = None


# ==============================================================================
# Session Types
# ==============================================================================


class SessionResponse(BaseModel):
    """Session API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    is_active: bool
    workspace_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: SessionConfigurationResponse = Field(
        default_factory=SessionConfigurationResponse
    )
    created_at: datetime.datetime


class SessionCreateParams(BaseModel):
    """Parameters for creating a session."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str = Field(min_length=1, max_length=100)
    metadata: dict[str, Any] | None = None
    peers: dict[str, SessionPeerConfig] | None = None
    configuration: SessionConfiguration | None = None


class SessionUpdateParams(BaseModel):
    """Parameters for updating a session."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    metadata: dict[str, Any] | None = None
    configuration: SessionConfiguration | None = None


class SessionListParams(BaseModel):
    """Parameters for listing sessions."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


# ==============================================================================
# Summary Types
# ==============================================================================


class Summary(BaseModel):
    """Summary model."""

    content: str
    message_id: str
    summary_type: str
    created_at: str
    token_count: int


class SessionSummariesResponse(BaseModel):
    """Session summaries API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    short_summary: Summary | None = None
    long_summary: Summary | None = None


# ==============================================================================
# Session Context Types
# ==============================================================================


class SessionContextResponse(BaseModel):
    """Session context API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    messages: list["MessageResponse"]
    summary: Summary | None = None
    peer_representation: str | None = None
    peer_card: list[str] | None = None


# ==============================================================================
# Message Types
# ==============================================================================


class MessageResponse(BaseModel):
    """Message API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    content: str
    peer_id: str
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime
    workspace_id: str
    token_count: int


class MessageCreateParams(BaseModel):
    """Parameters for creating a message."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    content: str
    peer_id: str
    metadata: dict[str, Any] | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None


class MessageBatchCreateParams(BaseModel):
    """Parameters for batch message creation."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    messages: list[MessageCreateParams] = Field(min_length=1, max_length=100)


class MessageUpdateParams(BaseModel):
    """Parameters for updating a message."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    metadata: dict[str, Any] | None = None


class MessageListParams(BaseModel):
    """Parameters for listing messages."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


class MessageSearchParams(BaseModel):
    """Parameters for searching messages."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    query: str
    filters: dict[str, Any] | None = None
    limit: int = Field(default=10, ge=1, le=100)


# ==============================================================================
# Conclusion Types
# ==============================================================================


class ConclusionResponse(BaseModel):
    """Conclusion API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    content: str
    observer_id: str
    observed_id: str
    session_id: str | None = None
    created_at: datetime.datetime


class ConclusionCreateParams(BaseModel):
    """Parameters for creating a conclusion."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    content: str = Field(min_length=1, max_length=65535)
    observer_id: str
    observed_id: str
    session_id: str | None = None


class ConclusionBatchCreateParams(BaseModel):
    """Parameters for batch conclusion creation."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    conclusions: list[ConclusionCreateParams] = Field(min_length=1, max_length=100)


class ConclusionListParams(BaseModel):
    """Parameters for listing conclusions."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


class ConclusionQueryParams(BaseModel):
    """Parameters for querying conclusions."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    distance: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: dict[str, Any] | None = None


# ==============================================================================
# Queue Status Types
# ==============================================================================


class SessionQueueStatus(BaseModel):
    """Status for a specific session in the queue."""

    session_id: str | None = None
    total_work_units: int
    completed_work_units: int
    in_progress_work_units: int
    pending_work_units: int


class QueueStatusResponse(BaseModel):
    """Queue status API response."""

    total_work_units: int
    completed_work_units: int
    in_progress_work_units: int
    pending_work_units: int
    sessions: dict[str, SessionQueueStatus] | None = None


# ==============================================================================
# Dialectic (Chat) Types
# ==============================================================================


ReasoningLevel = Literal["minimal", "low", "medium", "high", "max"]


class DialecticParams(BaseModel):
    """Parameters for dialectic chat."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    session_id: str | None = None
    target: str | None = None
    query: str = Field(min_length=1, max_length=10000)
    stream: bool = False
    reasoning_level: ReasoningLevel = "low"


class DialecticResponse(BaseModel):
    """Dialectic chat API response."""

    content: str | None


class DialecticStreamDelta(BaseModel):
    """Delta for streaming dialectic responses."""

    content: str | None = None


class DialecticStreamChunk(BaseModel):
    """Chunk in a streaming dialectic response."""

    delta: DialecticStreamDelta
    done: bool = False


# ==============================================================================
# Pagination Types
# ==============================================================================


class PageResponse(BaseModel):
    """Generic paginated response."""

    items: list[Any]
    page: int
    size: int
    total: int
    pages: int


# ==============================================================================
# File Upload Types
# ==============================================================================


class MessageUploadParams(BaseModel):
    """Parameters for file upload message creation."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    peer_id: str
    metadata: dict[str, Any] | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None


# Update forward reference
SessionContextResponse.model_rebuild()
