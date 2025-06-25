# pyright: reportUnannotatedClassAttribute=false # pyright: ignore
import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    filter: dict[str, Any] | None = None


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
    filter: dict[str, Any] | None = None


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


class PeerConfig(BaseModel):
    observe_me: bool = Field(
        default=True,
        description="Whether honcho should form a global theory-of-mind representation of this peer",
    )


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: Annotated[str, Field(min_length=0, max_length=50000)]
    peer_name: str = Field(alias="peer_id")
    metadata: dict[str, Any] | None = None


class MessageGet(MessageBase):
    filter: dict[str, Any] | None = None


class MessageUpdate(MessageBase):
    metadata: dict[str, Any] | None = None


class Message(MessageBase):
    public_id: str = Field(serialization_alias="id")
    content: str
    peer_name: str = Field(serialization_alias="peer_id")
    session_name: str | None = Field(serialization_alias="session_id")
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
    filter: dict[str, Any] | None = None
    is_active: bool = False


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


class SessionContext(SessionBase):
    name: str = Field(serialization_alias="id")
    messages: list[Message]
    summary: str

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
    use_semantic_search: bool = Field(
        default=False,
        description="Whether to use semantic search to filter the results",
    )


class DialecticOptions(BaseModel):
    session_id: str | None = Field(
        None, description="ID of the session to scope the representation to"
    )
    target: str | None = Field(
        None,
        description="Optional peer to get the representation for, from the perspective of this peer",
    )
    queries: str | list[str]
    stream: bool = False

    @field_validator("queries")
    def validate_queries(cls, v: str | list[str]) -> str | list[str]:
        MAX_STRING_LENGTH = 10000
        MAX_LIST_LENGTH = 25
        if isinstance(v, str):
            if len(v) > MAX_STRING_LENGTH:
                raise ValueError("Query too long")
        else:
            if len(v) > MAX_LIST_LENGTH:
                raise ValueError("Too many queries")
            if any(len(q) > MAX_STRING_LENGTH for q in v):
                raise ValueError("One or more queries too long")
        return v


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


class DeriverStatus(BaseModel):
    peer_id: str | None = Field(
        default=None,
        description="ID of the peer (optional when filtering by session only)",
    )
    session_id: str | None = Field(
        default=None, description="Session ID if filtered by session"
    )
    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(description="Completed work units")
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")
    sessions: dict[str, "DeriverStatus"] | None = Field(
        default=None, description="Per-session status when not filtered by session"
    )
