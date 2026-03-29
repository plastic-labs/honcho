"""Pydantic schemas for API request/response validation.

These schemas are consumed by the FastAPI routers and define the public
API contract.
"""

import datetime
import ipaddress
from typing import Annotated, Any, Self, cast
from urllib.parse import urlparse

import tiktoken
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from src.config import ReasoningLevel, settings
from src.schemas.configuration import (
    DreamType,
    MessageConfiguration,
    SessionConfiguration,
    SessionPeerConfig,
    WorkspaceConfiguration,
)

# ---------------------------------------------------------------------------
# Metadata validation helpers
# ---------------------------------------------------------------------------

RESOURCE_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"

_METADATA_MAX_KEYS = 100
_METADATA_MAX_DEPTH = 5


def strip_nul_bytes(value: Any) -> Any:
    """Strip NUL bytes from string inputs without touching other types."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def _sanitize_value(v: Any) -> Any:
    """Recursively strip NUL bytes from strings in nested data structures."""
    if isinstance(v, str):
        return strip_nul_bytes(v)
    if isinstance(v, dict):
        d = cast(dict[str, Any], v)
        return {_sanitize_value(k): _sanitize_value(val) for k, val in d.items()}
    if isinstance(v, list):
        lst = cast(list[Any], v)
        return [_sanitize_value(item) for item in lst]
    return v


def _check_metadata_limits(
    value: Any,
    *,
    _current_depth: int = 1,
) -> None:
    """Validate metadata doesn't exceed key count or nesting depth limits."""
    if _current_depth > _METADATA_MAX_DEPTH:
        raise ValueError(
            f"Metadata nesting exceeds maximum depth of {_METADATA_MAX_DEPTH}"
        )

    if isinstance(value, dict):
        data = cast(dict[str, Any], value)
        if _current_depth == 1 and len(data) > _METADATA_MAX_KEYS:
            raise ValueError(
                f"Metadata exceeds maximum of {_METADATA_MAX_KEYS} top-level keys"
            )
        for item in data.values():
            if isinstance(item, (dict, list)):
                _check_metadata_limits(item, _current_depth=_current_depth + 1)
        return

    if isinstance(value, list):
        items = cast(list[Any], value)
        for item in items:
            if isinstance(item, (dict, list)):
                _check_metadata_limits(item, _current_depth=_current_depth + 1)
        return

    if _current_depth == 1:
        raise ValueError("Metadata must be a dict")


def _validate_metadata(v: Any) -> Any:
    """Validate and sanitize a metadata dict: enforce limits and strip NUL bytes."""
    if not isinstance(v, dict):
        return v
    data = cast(dict[str, Any], v)
    _check_metadata_limits(data)
    return _sanitize_value(data)


_SanitizedMetadata = Annotated[dict[str, Any], BeforeValidator(_validate_metadata)]

# ---------------------------------------------------------------------------
# Workspace schemas
# ---------------------------------------------------------------------------


class WorkspaceBase(BaseModel):
    pass


class WorkspaceCreate(WorkspaceBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: _SanitizedMetadata = {}
    configuration: WorkspaceConfiguration = Field(
        default_factory=WorkspaceConfiguration
    )

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class WorkspaceGet(WorkspaceBase):
    filters: dict[str, Any] | None = None


class WorkspaceUpdate(WorkspaceBase):
    metadata: _SanitizedMetadata | None = None
    configuration: WorkspaceConfiguration | None = None


class Workspace(WorkspaceBase):
    name: str = Field(serialization_alias="id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


# ---------------------------------------------------------------------------
# Peer schemas
# ---------------------------------------------------------------------------


class PeerBase(BaseModel):
    pass


class PeerCreate(PeerBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: _SanitizedMetadata | None = None
    configuration: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class PeerGet(PeerBase):
    filters: dict[str, Any] | None = None


class PeerUpdate(PeerBase):
    metadata: _SanitizedMetadata | None = None
    configuration: dict[str, Any] | None = None


class Peer(PeerBase):
    name: str = Field(serialization_alias="id")
    workspace_name: str = Field(serialization_alias="workspace_id")
    created_at: datetime.datetime
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class PeerRepresentationGet(BaseModel):
    session_id: str | None = Field(
        None, description="Optional session ID within which to scope the representation"
    )
    target: str | None = Field(
        None,
        description="Optional peer ID to get the representation for, from the perspective of this peer",
    )
    search_query: str | None = Field(
        None,
        description="Optional input to curate the representation around semantic search results",
    )
    search_top_k: int | None = Field(
        None,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. Number of semantic-search-retrieved conclusions to include in the representation",
    )
    search_max_distance: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Only used if `search_query` is provided. Maximum distance to search for semantically relevant conclusions",
    )
    include_most_frequent: bool | None = Field(
        default=None,
        description="Only used if `search_query` is provided. Whether to include the most frequent conclusions in the representation",
    )
    max_conclusions: int | None = Field(
        default=25,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. Maximum number of conclusions to include in the representation",
    )


class RepresentationResponse(BaseModel):
    representation: str


class PeerCardResponse(BaseModel):
    peer_card: list[str] | None = Field(
        None, description="The peer card content, or None if not found"
    )


class PeerCardSet(BaseModel):
    peer_card: list[str] = Field(..., description="The peer card content to set")

    @field_validator("peer_card", mode="before")
    @classmethod
    def sanitize_peer_card(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [
                item.replace("\x00", "") if isinstance(item, str) else item
                for item in cast(list[Any], v)
            ]
        return v


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: Annotated[str, Field(min_length=0, max_length=settings.MAX_MESSAGE_SIZE)]
    peer_name: str = Field(alias="peer_id")
    metadata: _SanitizedMetadata | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None

    _encoded_message: list[int] = PrivateAttr(default=[])

    @field_validator("content", mode="after")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return v.replace("\x00", "")

    @property
    def encoded_message(self) -> list[int]:
        return self._encoded_message

    @model_validator(mode="after")
    def validate_and_set_token_count(self) -> Self:
        encoding = tiktoken.get_encoding("o200k_base")
        encoded_message = encoding.encode(self.content)

        self._encoded_message = encoded_message
        return self


class MessageGet(MessageBase):
    filters: dict[str, Any] | None = None


class MessageUpdate(MessageBase):
    metadata: _SanitizedMetadata | None = None


class Message(MessageBase):
    public_id: str = Field(serialization_alias="id")
    content: str
    peer_name: str = Field(serialization_alias="peer_id")
    session_name: str = Field(serialization_alias="session_id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    created_at: datetime.datetime
    workspace_name: str = Field(serialization_alias="workspace_id")
    token_count: int

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class MessageBatchCreate(BaseModel):
    """Schema for batch message creation with a max of 100 messages"""

    messages: list[MessageCreate] = Field(..., min_length=1, max_length=100)


class MessageUploadCreate(BaseModel):
    """Schema for message creation from file uploads"""

    peer_id: str = Field(..., description="ID of the peer creating the message")
    metadata: _SanitizedMetadata | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


# ---------------------------------------------------------------------------
# Session schemas
# ---------------------------------------------------------------------------


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: _SanitizedMetadata | None = None
    peer_names: dict[str, SessionPeerConfig] | None = Field(default=None, alias="peers")
    configuration: SessionConfiguration | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class SessionGet(SessionBase):
    filters: dict[str, Any] | None = None


class SessionUpdate(SessionBase):
    metadata: _SanitizedMetadata | None = None
    configuration: SessionConfiguration | None = None


class Session(SessionBase):
    name: str = Field(serialization_alias="id")
    is_active: bool
    workspace_name: str = Field(serialization_alias="workspace_id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class Summary(BaseModel):
    content: str = Field(description="The summary text")
    message_id: int = Field(
        description="The internal ID of the message that this summary covers up to",
        exclude=True,
    )
    message_public_id: str = Field(
        description="The public ID of the message that this summary covers up to",
        serialization_alias="message_id",
    )
    summary_type: str = Field(description="The type of summary (short or long)")
    created_at: str = Field(
        description="The timestamp of when the summary was created (ISO format)"
    )
    token_count: int = Field(description="The number of tokens in the summary text")


class SessionContext(SessionBase):
    name: str = Field(serialization_alias="id")
    messages: list[Message]
    summary: Summary | None = Field(
        default=None, description="The summary if available"
    )
    peer_representation: str | None = Field(
        default=None,
        description="A curated subset of a peer representation, if context is requested from a specific perspective",
    )
    peer_card: list[str] | None = Field(
        default=None,
        description="The peer card, if context is requested from a specific perspective",
    )

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class PeerContext(BaseModel):
    """Context for a peer, including representation and peer card."""

    peer_id: str = Field(description="The ID of the peer")
    target_id: str = Field(description="The ID of the target peer being observed")
    representation: str | None = Field(
        default=None,
        description="A curated subset of the representation of the target peer from the observer's perspective",
    )
    peer_card: list[str] | None = Field(
        default=None,
        description="The peer card for the target peer from the observer's perspective",
    )


class SessionSummaries(SessionBase):
    name: str = Field(serialization_alias="id")
    short_summary: Summary | None = Field(
        default=None, description="The short summary if available"
    )
    long_summary: Summary | None = Field(
        default=None, description="The long summary if available"
    )

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


# ---------------------------------------------------------------------------
# Conclusion schemas
# ---------------------------------------------------------------------------


class ConclusionGet(BaseModel):
    """Schema for listing conclusions with optional filters."""

    filters: dict[str, Any] | None = None


class Conclusion(BaseModel):
    """Conclusion response - external view of a document."""

    id: str
    content: str
    observer: str = Field(
        description="The peer who made the conclusion",
        serialization_alias="observer_id",
    )
    observed: str = Field(
        description="The peer the conclusion is about",
        serialization_alias="observed_id",
    )
    session_name: str | None = Field(default=None, serialization_alias="session_id")
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True,
        populate_by_name=True,
    )


class ConclusionQuery(BaseModel):
    """Query parameters for semantic search of conclusions."""

    query: str = Field(..., description="Semantic search query")
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    distance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum cosine distance threshold for results",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Additional filters to apply",
    )


class ConclusionCreate(BaseModel):
    """Schema for creating a single conclusion."""

    content: Annotated[str, Field(min_length=1, max_length=65535)]
    observer_id: str = Field(..., description="The peer making the conclusion")
    observed_id: str = Field(..., description="The peer the conclusion is about")
    session_id: str | None = Field(
        default=None,
        description="A session ID to store the conclusion in, if specified",
    )

    _token_count: int = PrivateAttr(default=0)

    @field_validator("content", mode="after")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        sanitized = cast(str, strip_nul_bytes(v))
        if not sanitized:
            raise PydanticCustomError(
                "string_too_short",
                "String should have at least 1 character",
            )
        return sanitized

    @model_validator(mode="after")
    def validate_token_count(self) -> Self:
        """Validate that content doesn't exceed embedding token limit."""
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(self.content)
        self._token_count = len(tokens)

        if self._token_count > settings.MAX_EMBEDDING_TOKENS:
            raise ValueError(
                f"Content exceeds maximum embedding token limit of {settings.MAX_EMBEDDING_TOKENS} "
                + f"(got {self._token_count} tokens)"
            )
        return self


class ConclusionBatchCreate(BaseModel):
    """Schema for batch conclusion creation with a max of 100 conclusions."""

    conclusions: list[ConclusionCreate] = Field(
        ...,
        min_length=1,
        max_length=100,
        validation_alias=AliasChoices("conclusions", "observations"),
    )


# ---------------------------------------------------------------------------
# Search schemas
# ---------------------------------------------------------------------------


class MessageSearchOptions(BaseModel):
    query: Annotated[str, Field(description="Search query")]
    filters: dict[str, Any] | None = Field(
        default=None, description="Filters to scope the search"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v

        sanitized = cast(str, strip_nul_bytes(v))
        if v != "" and sanitized == "":
            raise PydanticCustomError(
                "string_too_short",
                "String should have at least 1 character",
            )

        return sanitized


# ---------------------------------------------------------------------------
# Dialectic schemas
# ---------------------------------------------------------------------------


class DialecticOptions(BaseModel):
    session_id: str | None = Field(
        None, description="ID of the session to scope the representation to"
    )
    target: str | None = Field(
        None,
        description="Optional peer to get the representation for, from the perspective of this peer",
    )
    query: Annotated[
        str, Field(min_length=1, max_length=10000, description="Dialectic API Prompt")
    ]
    stream: bool = False
    reasoning_level: ReasoningLevel = Field(
        default="low",
        description="Level of reasoning to apply: minimal, low, medium, high, or max",
    )

    @field_validator("query", mode="after")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        sanitized = cast(str, strip_nul_bytes(v))
        if not sanitized:
            raise PydanticCustomError(
                "string_too_short",
                "String should have at least 1 character",
            )
        return sanitized


class DialecticResponse(BaseModel):
    content: str | None


class DialecticStreamDelta(BaseModel):
    """Delta object for streaming dialectic responses."""

    content: str | None = None
    # Future fields can be added here:
    # premises: str | None = None
    # tokens: int | None = None
    # analytics: dict[str, Any] | None = None


class DialecticStreamChunk(BaseModel):
    """Chunk in a streaming dialectic response."""

    delta: DialecticStreamDelta
    done: bool = False


# ---------------------------------------------------------------------------
# Queue status schemas
# ---------------------------------------------------------------------------


class SessionQueueStatus(BaseModel):
    """Status for a specific session within the processing queue."""

    session_id: str | None = Field(
        default=None,
        description="Session ID if filtered by session",
    )
    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(description="Completed work units")
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")


class QueueStatus(BaseModel):
    """Aggregated processing queue status.

    Tracks user-facing task types only: representation, summary, and dream.
    Internal infrastructure tasks (reconciler, webhook, deletion) are excluded.

    Note: completed_work_units reflects items since the last periodic queue
    cleanup, not lifetime totals.
    """

    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(
        description="Completed work units (since last periodic cleanup)"
    )
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")
    sessions: dict[str, SessionQueueStatus] | None = Field(
        default=None,
        description="Per-session status when not filtered by session",
    )


# ---------------------------------------------------------------------------
# Dream scheduling schemas
# ---------------------------------------------------------------------------


class ScheduleDreamRequest(BaseModel):
    observer: str = Field(..., description="Observer peer name")
    observed: str | None = Field(
        None, description="Observed peer name (defaults to observer if not specified)"
    )
    dream_type: DreamType = Field(..., description="Type of dream to schedule")
    session_id: str | None = Field(
        None, description="Session ID to scope the dream to if specified"
    )


# ---------------------------------------------------------------------------
# Webhook schemas
# ---------------------------------------------------------------------------


class WebhookEndpointBase(BaseModel):
    pass


class WebhookEndpointCreate(WebhookEndpointBase):
    url: str

    @field_validator("url")
    @classmethod
    def validate_webhook_url(cls, v: str) -> str:
        parsed = urlparse(v)

        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")

        # Only allow HTTP/HTTPS
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("Only HTTP and HTTPS URLs are allowed")

        # Block private/internal addresses
        if parsed.hostname:
            try:
                ip_address = ipaddress.ip_address(parsed.hostname)
                if ip_address.is_private:
                    raise ValueError("Private IP addresses are not allowed")
            except ValueError:  # Not an IP address, might be a hostname
                pass

        return v


class WebhookEndpoint(WebhookEndpointBase):
    id: str
    workspace_name: str | None = Field(serialization_alias="workspace_id")
    url: str
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore
