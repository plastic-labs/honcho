"""Pydantic schemas for graph memory API (edges, recall, contexts, thread bindings)."""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.utils.types import EdgeType, AccessLogEventType


# ── Edge schemas ──────────────────────────────────────────────────────────

class EdgeCreate(BaseModel):
    """Request body for creating an edge."""
    collection_name: str = Field(..., description="Collection scoping the edge")
    source_obs_id: str = Field(..., description="Source observation ID")
    target_obs_id: str = Field(..., description="Target observation ID")
    edge_type: EdgeType = Field(..., description="Type of edge")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    @field_validator("source_obs_id", "target_obs_id")
    @classmethod
    def validate_obs_id(cls, v: str) -> str:
        if len(v) != 21:
            raise ValueError("Observation ID must be 21 characters (nanoid)")
        return v


class EdgeResponse(BaseModel):
    """Response body for an edge."""
    id: int
    workspace_name: str
    collection_name: str
    source_obs_id: str
    target_obs_id: str
    edge_type: str
    created_by: str
    created_at: datetime.datetime
    metadata: dict[str, Any]


class EdgeListFilter(BaseModel):
    """Filter options for listing edges."""
    source_obs_id: str | None = None
    target_obs_id: str | None = None
    edge_type: EdgeType | None = None
    collection_name: str | None = None


# ── Access log schemas ─────────────────────────────────────────────────────

class AccessLogEntryCreate(BaseModel):
    """Request body for creating an access log entry."""
    collection_name: str
    obs_id: str
    event_type: AccessLogEventType
    session_id: str | None = None


class AccessLogEntryResponse(BaseModel):
    """Response body for an access log entry."""
    id: int
    workspace_name: str
    collection_name: str
    obs_id: str
    event_type: str
    created_by: str
    session_id: str | None
    created_at: datetime.datetime


# ── Recall schemas ─────────────────────────────────────────────────────────

class RecallRequest(BaseModel):
    """Request body for spreading-activation recall."""
    query: str = Field(..., description="Natural language query")
    collection_name: str = Field(..., description="Collection to search")
    max_depth: int = Field(default=3, ge=1, le=10, description="Max BFS depth")
    frontier_cap: int = Field(default=10, ge=1, le=100, description="Max frontier per level")
    token_budget: int = Field(default=2000, ge=100, le=10000, description="Max results")
    context: str | None = Field(default=None, description="Active context to filter by")
    include_pinned: bool = Field(default=True, description="Include pinned observations")


class RecallResult(BaseModel):
    """A single recall result."""
    obs_id: str
    content: str
    score: float
    activation: float
    confidence: float
    is_pinned: bool
    is_verify_due: bool
    workstream: str | None


class RecallResponse(BaseModel):
    """Response body for recall."""
    results: list[RecallResult]
    total_visited: int
    query_time_ms: float


# ── Context schemas ───────────────────────────────────────────────────────

class ContextCreate(BaseModel):
    """Request body for creating a context."""
    context_name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class ContextMemberAdd(BaseModel):
    """Request body for adding an observation to a context."""
    obs_id: str
    thread_id: str | None = None


class ContextResponse(BaseModel):
    """Response body for a context."""
    id: int
    workspace_name: str
    context_name: str
    member_count: int
    created_at: datetime.datetime


# ── Thread binding schemas ────────────────────────────────────────────────

class ThreadBindingCreate(BaseModel):
    """Request body for binding a thread to a context."""
    thread_id: str = Field(..., pattern=r"^[0-9]{10,}\.[0-9]+$")
    context_name: str


class ThreadBindingResponse(BaseModel):
    """Response body for a thread binding."""
    id: int
    workspace_name: str
    thread_id: str
    context_name: str
    bound_by: str
    bound_at: datetime.datetime


# ── Pin / Verify schemas ──────────────────────────────────────────────────

class PinRequest(BaseModel):
    """Request body for pinning an observation."""
    verify_cadence_days: int | None = Field(
        default=None, ge=1, le=3650,
        description="Optional verify cadence in days. Null = no explicit cadence."
    )


class VerifyRequest(BaseModel):
    """Request body for verifying an observation."""
    pass  # No body needed — verification is just a timestamped event


class VerifyDueItem(BaseModel):
    """A single verify-due observation."""
    obs_id: str
    content: str
    reason: str
    is_pinned: bool
    confidence: float
    last_verified: datetime.datetime | None
