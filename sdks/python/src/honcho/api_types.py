"""API types for the Honcho SDK.

These types replace the honcho_core imports with local definitions.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _Omit:
    """Sentinel class for omitting optional fields in API requests."""

    _instance: "_Omit | None" = None

    def __new__(cls) -> "_Omit":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "omit"

    def __bool__(self) -> bool:
        return False


# Singleton instance
omit = _Omit()


class Workspace(BaseModel):
    """Workspace response from the API."""

    model_config = ConfigDict(extra="allow")

    id: str
    metadata: dict[str, Any]
    configuration: dict[str, Any]
    created_at: str


class PeerCore(BaseModel):
    """Peer response from the API (raw API type)."""

    model_config = ConfigDict(extra="allow")

    id: str
    workspace_id: str
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    created_at: str


class SessionCore(BaseModel):
    """Session response from the API (raw API type)."""

    model_config = ConfigDict(extra="allow")

    id: str
    workspace_id: str
    is_active: bool
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    created_at: str


class Message(BaseModel):
    """Message from the API."""

    model_config = ConfigDict(extra="allow")

    id: str
    content: str
    peer_id: str
    session_id: str
    workspace_id: str
    token_count: int
    metadata: dict[str, Any] | None = None
    created_at: str


class MessageCreateParam(BaseModel):
    """Parameters for creating a message."""

    model_config = ConfigDict(extra="allow")

    peer_id: str
    content: str
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    created_at: str | None = None


class Configuration(BaseModel):
    """Message configuration."""

    model_config = ConfigDict(extra="allow")

    deriver: dict[str, Any] | None = None


class SessionDeriverStatus(BaseModel):
    """Deriver status for a specific session."""

    model_config = ConfigDict(extra="allow")

    session_id: str | None = None
    total_work_units: int
    completed_work_units: int
    in_progress_work_units: int
    pending_work_units: int


class DeriverStatus(BaseModel):
    """Deriver status response from the API."""

    model_config = ConfigDict(extra="allow")

    total_work_units: int
    completed_work_units: int
    in_progress_work_units: int
    pending_work_units: int
    sessions: dict[str, SessionDeriverStatus] | None = None


class PeerCardResponse(BaseModel):
    """Peer card response from the API."""

    model_config = ConfigDict(extra="allow")

    peer_card: list[str] | None = None


class PeerWorkingRepresentationResponse(BaseModel):
    """Peer working representation response from the API."""

    model_config = ConfigDict(extra="allow")

    explicit: list[str] | None = None
    deductive: list[str] | None = None


class DialecticResponse(BaseModel):
    """Dialectic response from the API (non-streaming)."""

    model_config = ConfigDict(extra="allow")

    content: str | None = None


class PageResponse(BaseModel):
    """Generic page response from the API."""

    model_config = ConfigDict(extra="allow")

    items: list[Any]
    total: int | None = None
    page: int
    size: int
    pages: int | None = None


class SummaryData(BaseModel):
    """Summary data from the API."""

    model_config = ConfigDict(extra="allow")

    content: str
    message_id: str
    summary_type: str
    created_at: str
    token_count: int


class SessionSummariesResponse(BaseModel):
    """Session summaries response from the API."""

    model_config = ConfigDict(extra="allow")

    id: str
    short_summary: SummaryData | None = None
    long_summary: SummaryData | None = None


class SessionContextResponse(BaseModel):
    """Session context response from the API."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    messages: list[Message]
    summary: SummaryData | None = None
    peer_representation: str | None = None
    peer_card: list[str] | None = None


class PeerContextResponse(BaseModel):
    """Peer context response from the API."""

    model_config = ConfigDict(extra="allow")

    peer_card: list[str] | None = None
    representation: str | None = None


class ObservationResponse(BaseModel):
    """Observation response from the API."""

    model_config = ConfigDict(extra="allow")

    id: str
    content: str
    observer_id: str
    observed_id: str
    session_id: str
    created_at: str
