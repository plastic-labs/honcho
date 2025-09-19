# pyright: reportUnannotatedClassAttribute=false # pyright: ignore
import datetime
import ipaddress
from typing import Annotated, Any, Self
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

RESOURCE_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"


class WorkspaceBase(BaseModel):
    pass


class WorkspaceCreate(WorkspaceBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: dict[str, Any] = {}
    configuration: dict[str, Any] = {}

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class WorkspaceGet(WorkspaceBase):
    filters: dict[str, Any] | None = None


class WorkspaceUpdate(WorkspaceBase):
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None


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
    session_id: str = Field(
        ..., description="Get the working representation within this session"
    )
    target: str | None = Field(
        None,
        description="Optional peer ID to get the representation for, from the perspective of this peer",
    )


class PeerCardGet(BaseModel):
    target: str | None = Field(
        None,
        description="The peer whose card to retrieve. If not provided, gets the observer's own card",
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
    configuration: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class SessionGet(SessionBase):
    filters: dict[str, Any] | None = None


class SessionUpdate(SessionBase):
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None


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
        description="The ID of the message that this summary covers up to"
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


class DocumentCreate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    metadata: dict[str, Any] = {}


class DocumentUpdate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    metadata: dict[str, Any] | None = None


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
