import datetime
import ipaddress
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

import tiktoken
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from src.config import settings
from src.utils.representation import Representation

RESOURCE_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"


class ResolvedSessionConfiguration(BaseModel):
    """
    Like a SessionConfiguration, but with all fields resolved.
    """

    deriver_enabled: bool
    peer_cards_enabled: bool
    summaries_enabled: bool
    dreams_enabled: bool
    messages_per_short_summary: int
    messages_per_long_summary: int


class SessionConfiguration(BaseModel):
    """
    The set of options that can be in a session DB-level configuration dictionary.

    All fields are optional. Session-level configuration overrides workspace-level configuration, which overrides global configuration.
    """

    model_config = ConfigDict(extra="allow")  # pyright: ignore

    deriver_enabled: bool | None = Field(
        default=None,
        description="Whether to enable deriver functionality.",
    )
    peer_cards_enabled: bool | None = Field(
        default=None,
        description="Whether to enable peer card functionality. If deriver is disabled, peer cards will also be disabled and this setting will be ignored.",
    )
    summaries_enabled: bool | None = Field(
        default=None,
        description="Whether to enable summary functionality.",
    )
    dreams_enabled: bool | None = Field(
        default=None,
        description="Whether to enable dream functionality. If deriver is disabled, dreams will also be disabled and this setting will be ignored.",
    )

    messages_per_short_summary: int | None = Field(
        default=None,
        ge=10,
        description="Number of messages per short summary. Must be positive, greater than or equal to 10, and less than messages_per_long_summary.",
    )
    messages_per_long_summary: int | None = Field(
        default=None,
        ge=20,
        description="Number of messages per long summary. Must be positive, greater than or equal to 20, and greater than messages_per_short_summary.",
    )

    @model_validator(mode="after")
    def validate_summary_thresholds(self) -> Self:
        """Validate that short summary threshold <= long summary threshold."""
        short = self.messages_per_short_summary
        long = self.messages_per_long_summary

        if short is not None and long is not None and short >= long:
            raise ValueError(
                "messages_per_short_summary must be less than messages_per_long_summary"
            )

        return self


class WorkspaceConfiguration(SessionConfiguration):
    """
    The set of options that can be in a workspace DB-level configuration dictionary.

    All fields are optional. Session-level configuration overrides workspace-level configuration, which overrides global configuration.
    """

    pass


class WorkspaceBase(BaseModel):
    pass


class WorkspaceCreate(WorkspaceBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: dict[str, Any] = {}
    configuration: WorkspaceConfiguration = Field(
        default_factory=WorkspaceConfiguration
    )

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class WorkspaceGet(WorkspaceBase):
    filters: dict[str, Any] | None = None


class WorkspaceUpdate(WorkspaceBase):
    metadata: dict[str, Any] | None = None
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


class PeerBase(BaseModel):
    pass


class PeerCreate(PeerBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class PeerGet(PeerBase):
    filters: dict[str, Any] | None = None


class PeerUpdate(PeerBase):
    metadata: dict[str, Any] | None = None
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
        None, description="Get the working representation within this session"
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
        description="Only used if `search_query` is provided. Number of semantic-search-retrieved observations to include in the representation",
    )
    search_max_distance: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Only used if `search_query` is provided. Maximum distance to search for semantically relevant observations",
    )
    include_most_derived: bool | None = Field(
        default=None,
        description="Only used if `search_query` is provided. Whether to include the most derived observations in the representation",
    )
    max_observations: int | None = Field(
        default=25,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. Maximum number of observations to include in the representation",
    )


class PeerCardResponse(BaseModel):
    peer_card: list[str] | None = Field(
        None, description="The peer card content, or None if not found"
    )


class PeerConfig(BaseModel):
    observe_me: bool = Field(
        default=True,
        description="Whether honcho should form a global theory-of-mind representation of this peer",
    )


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: Annotated[str, Field(min_length=0, max_length=settings.MAX_MESSAGE_SIZE)]
    peer_name: str = Field(alias="peer_id")
    metadata: dict[str, Any] | None = None
    created_at: datetime.datetime | None = None

    _encoded_message: list[int] = PrivateAttr(default=[])

    @property
    def encoded_message(self) -> list[int]:
        return self._encoded_message

    @model_validator(mode="after")
    def validate_and_set_token_count(self) -> Self:
        encoding = tiktoken.get_encoding("cl100k_base")
        encoded_message = encoding.encode(self.content)

        self._encoded_message = encoded_message
        return self


class MessageGet(MessageBase):
    filters: dict[str, Any] | None = None


class MessageUpdate(MessageBase):
    metadata: dict[str, Any] | None = None


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

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class SessionBase(BaseModel):
    pass


class SessionPeerConfig(BaseModel):
    observe_others: bool = Field(
        default=False,
        description="Whether this peer should form a session-level theory-of-mind representation of other peers in the session",
    )
    observe_me: bool | None = Field(
        default=None,
        description="Whether other peers in this session should try to form a session-level theory-of-mind representation of this peer",
    )


class SessionCreate(SessionBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: dict[str, Any] | None = None
    peer_names: dict[str, SessionPeerConfig] | None = Field(default=None, alias="peers")
    configuration: SessionConfiguration | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class SessionGet(SessionBase):
    filters: dict[str, Any] | None = None


class SessionUpdate(SessionBase):
    metadata: dict[str, Any] | None = None
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
    peer_representation: Representation | None = Field(
        default=None,
        description="The peer representation, if context is requested from a specific perspective",
    )
    peer_card: list[str] | None = Field(
        default=None,
        description="The peer card, if context is requested from a specific perspective",
    )

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
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


class DocumentBase(BaseModel):
    pass


class DocumentMetadata(BaseModel):
    message_ids: list[tuple[int, int]] = Field(
        description="The ID range(s) of the messages that this document was derived from. Acts as a link to the primary source of the document. Note that as a document gets deduplicated, additional ranges will be added, because the same document could be derived from completely separate message ranges."
    )
    message_created_at: str = Field(
        description="The timestamp of the message that this document was derived from. Note that this is not the same as the created_at timestamp of the document. This timestamp is usually only saved with second-level precision."
    )
    premises: list[str] | None = Field(
        default=None,
        description="The premises of the deduction -- only applicable for deductive observations",
    )


class DocumentCreate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    session_name: str = Field(
        description="The session from which the document was derived"
    )
    level: Literal["explicit", "deductive"] = Field(
        default="explicit",
        description="The level of the document (explicit or deductive)",
    )
    times_derived: int = Field(
        default=1,
        ge=1,
        description="The number of times that a semantic duplicate document to this one has been derived",
    )
    metadata: DocumentMetadata = Field()
    embedding: list[float] = Field()


class MessageSearchOptions(BaseModel):
    query: str = Field(..., description="Search query")
    filters: dict[str, Any] | None = Field(
        default=None, description="Filters to scope the search"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )


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


class DialecticResponse(BaseModel):
    content: str


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


class PeerConfigResult(BaseModel):
    """Result from querying peer configuration data."""

    peer_name: str
    peer_configuration: dict[str, Any]
    session_peer_configuration: dict[str, Any]


class SessionPeerData(BaseModel):
    """Data for managing session peer relationships."""

    peer_names: dict[str, SessionPeerConfig]


class MessageBulkData(BaseModel):
    """Data for bulk message operations."""

    messages: list[MessageCreate]
    session_name: str
    workspace_name: str


class SessionDeriverStatus(BaseModel):
    session_id: str | None = Field(
        default=None, description="Session ID if filtered by session"
    )
    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(description="Completed work units")
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")


class DeriverStatus(BaseModel):
    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(description="Completed work units")
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")
    sessions: dict[str, SessionDeriverStatus] | None = Field(
        default=None, description="Per-session status when not filtered by session"
    )


# Webhook endpoint schemas
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
