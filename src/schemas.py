import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkspaceBase(BaseModel):
    pass


class WorkspaceCreate(WorkspaceBase):
    name: Annotated[str, Field(alias="id", min_length=1, max_length=100)]
    metadata: dict = {}
    feature_flags: dict = {}

    model_config = ConfigDict(populate_by_name=True)


class WorkspaceGet(WorkspaceBase):
    filter: dict | None = None


class WorkspaceUpdate(WorkspaceBase):
    metadata: dict | None = None
    feature_flags: dict | None = None


class Workspace(WorkspaceBase):
    name: str = Field(serialization_alias="id")
    h_metadata: dict = Field(default={}, serialization_alias="metadata")
    feature_flags: dict = Field(default={})
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class PeerBase(BaseModel):
    pass


class PeerCreate(PeerBase):
    name: Annotated[str, Field(alias="id", min_length=1, max_length=100)]
    metadata: dict = {}
    feature_flags: dict = {}

    model_config = ConfigDict(populate_by_name=True)


class PeerGet(PeerBase):
    filter: dict | None = None


class PeerUpdate(PeerBase):
    metadata: dict | None = None
    feature_flags: dict | None = None


class Peer(PeerBase):
    name: str = Field(serialization_alias="id")
    workspace_name: str
    created_at: datetime.datetime
    h_metadata: dict = Field(default={}, serialization_alias="metadata")
    feature_flags: dict = Field(default={})

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: Annotated[str, Field(min_length=0, max_length=50000)]
    peer_name: str = Field(alias="peer_id")
    metadata: dict | None = None


class MessageGet(MessageBase):
    filter: dict | None = None


class MessageUpdate(MessageBase):
    metadata: dict


class Message(MessageBase):
    public_id: str = Field(serialization_alias="id")
    content: str
    peer_id: str = Field(alias="peer_name")
    session_id: str | None = Field(alias="session_name")
    h_metadata: dict = Field(default={}, serialization_alias="metadata")
    created_at: datetime.datetime
    workspace_name: str
    token_count: int

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    name: Annotated[str, Field(alias="id", min_length=1, max_length=100)]
    metadata: dict = {}
    peer_names: set[str] | None = None
    feature_flags: dict = {}

    model_config = ConfigDict(populate_by_name=True)


class SessionGet(SessionBase):
    filter: dict | None = None
    is_active: bool = False


class SessionUpdate(SessionBase):
    metadata: dict
    feature_flags: dict | None = None


class Session(SessionBase):
    name: str = Field(serialization_alias="id")
    is_active: bool
    workspace_name: str
    h_metadata: dict = Field(default={}, serialization_alias="metadata")
    feature_flags: dict = Field(default={})
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class SessionContext(SessionBase):
    name: str = Field(serialization_alias="id")
    messages: list[Message]
    summary: str

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class DocumentBase(BaseModel):
    pass


class DocumentCreate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    metadata: dict = {}


class DialecticOptions(BaseModel):
    queries: str | list[str]
    stream: bool = False

    @field_validator("queries")
    def validate_queries(cls, v):
        MAX_STRING_LENGTH = 10000
        MAX_LIST_LENGTH = 25
        if isinstance(v, str):
            if len(v) > MAX_STRING_LENGTH:
                raise ValueError("Query too long")
        elif isinstance(v, list):
            if len(v) > MAX_LIST_LENGTH:
                raise ValueError("Too many queries")
            if any(len(q) > MAX_STRING_LENGTH for q in v):
                raise ValueError("One or more queries too long")
        return v


class DialecticResponse(BaseModel):
    content: str


class MessageBatchCreate(BaseModel):
    """Schema for batch message creation with a max of 100 messages"""

    messages: list[MessageCreate] = Field(..., max_length=100)
