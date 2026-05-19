"""
API events for Honcho telemetry.

These events track user-facing API operations
"""

from typing import ClassVar, Literal

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class MessageCreatedEvent(BaseEvent):
    """Emitted when one or more messages are created.

    This is the canonical API event for counting created messages, including
    messages created from file uploads. The resource_id keys on
    `last_message_id` (the trailing message's public nanoid) so two batches
    of the same size in the same session+source produce distinct event ids.
    """

    _event_type: ClassVar[str] = "message.created"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "api"

    workspace_name: str = Field(..., description="Workspace name")
    session_name: str = Field(..., description="Session name")
    message_count: int = Field(..., description="Number of messages created")
    total_tokens: int = Field(..., description="Total tokens across created messages")
    source: Literal["api", "file_upload"] = Field(
        default="api", description="Source of the created messages"
    )
    last_message_id: str = Field(
        ...,
        description="public_id (nanoid) of the trailing message in the batch — used as the stable unique key for this emission",
    )

    def get_resource_id(self) -> str:
        """Resource ID keys on the trailing message's public_id so two batches
        of the same size in the same session+source produce distinct event ids.
        """
        return (
            f"{self.workspace_name}:{self.session_name}:"
            f"{self.source}:{self.last_message_id}"
        )


class FileUploadedEvent(BaseEvent):
    """Emitted when an uploaded file is converted into messages.

    This captures file-side metadata. Message creation counts should use
    MessageCreatedEvent to avoid double-counting file uploads.
    """

    _event_type: ClassVar[str] = "file.uploaded"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "api"

    workspace_name: str = Field(..., description="Workspace name")
    session_name: str = Field(..., description="Session name")
    peer_name: str = Field(..., description="Peer that uploaded the file")
    file_id: str = Field(..., description="Generated file identifier")
    filename: str | None = Field(default=None, description="Uploaded filename")
    content_type: str | None = Field(default=None, description="Uploaded content type")
    file_size_bytes: int | None = Field(
        default=None, description="Uploaded file size in bytes"
    )
    message_count: int = Field(
        ..., description="Number of messages created from the file"
    )
    total_tokens: int = Field(
        ..., description="Total tokens across messages created from the file"
    )

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, session, and generated file ID."""
        return f"{self.workspace_name}:{self.session_name}:{self.file_id}"


class GetContextEvent(BaseEvent):
    """Emitted when context is retrieved for a session or peer."""

    _event_type: ClassVar[str] = "context.retrieved"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "api"

    workspace_name: str = Field(..., description="Workspace name")
    context_scope: Literal["session", "peer"] = Field(
        ..., description="Context endpoint scope"
    )
    session_name: str | None = Field(
        default=None, description="Session name for session-scoped context"
    )
    peer_name: str | None = Field(default=None, description="Observer peer name")
    target_name: str | None = Field(default=None, description="Observed peer name")
    tokens_requested: int | None = Field(
        default=None,
        description="Caller-supplied tokens query parameter (None = endpoint default applied)",
    )
    message_count: int = Field(
        default=0, description="Number of messages returned in context"
    )
    has_summary: bool = Field(
        default=False, description="Whether a summary was returned"
    )
    has_representation: bool = Field(
        default=False, description="Whether a representation was returned"
    )
    has_peer_card: bool = Field(
        default=False, description="Whether a peer card was returned"
    )
    search_query_provided: bool = Field(
        default=False, description="Whether semantic search query text was provided"
    )
    search_top_k: int | None = Field(
        default=None,
        description="Caller-supplied search_top_k (None = endpoint default)",
    )
    search_max_distance: float | None = Field(
        default=None,
        description="Caller-supplied search_max_distance (None = endpoint default)",
    )
    include_most_frequent: bool | None = Field(
        default=None,
        description="Caller-supplied include_most_frequent (None = endpoint default; defaults differ between peer and session endpoints)",
    )
    max_conclusions: int | None = Field(
        default=None,
        description="Caller-supplied max_conclusions (None = endpoint default)",
    )
    include_summary: bool | None = Field(
        default=None,
        description="Whether summary inclusion was requested; None when unsupported by endpoint",
    )
    limit_to_session: bool = Field(
        default=False,
        description="Whether representation retrieval was session-limited",
    )
    peer_perspective_provided: bool = Field(
        default=False,
        description="Whether peer_perspective was supplied for session context",
    )
    total_duration_ms: float = Field(..., description="Total processing time")

    def get_resource_id(self) -> str:
        """Resource ID identifies the requested context scope.

        Uses empty string (illegal in peer names — nanoid-derived) as the
        absent-peer sentinel so that a peer literally named "none" does not
        collide with the absent-peer case.
        """
        peer_name = self.peer_name if self.peer_name is not None else ""
        target_name = self.target_name if self.target_name is not None else ""
        if self.context_scope == "session":
            return (
                f"{self.workspace_name}:session:{self.session_name}:"
                f"{peer_name}:{target_name}"
            )
        return f"{self.workspace_name}:peer:{peer_name}:{target_name}"


__all__ = [
    "FileUploadedEvent",
    "GetContextEvent",
    "MessageCreatedEvent",
]
