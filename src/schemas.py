import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class WorkspaceBase(BaseModel):
    pass


class WorkspaceCreate(WorkspaceBase):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    metadata: dict = {}


class WorkspaceGet(WorkspaceBase):
    filter: dict | None = None


class WorkspaceUpdate(WorkspaceBase):
    name: str | None = None
    metadata: dict | None = None


class Workspace(WorkspaceBase):
    public_id: str = Field(serialization_alias='id')
    name: str
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    )


class PeerBase(BaseModel):
    pass


class PeerCreate(PeerBase):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    metadata: dict = {}


class PeerGet(PeerBase):
    filter: dict | None = None


class PeerUpdate(PeerBase):
    name: str | None = None
    metadata: dict | None = None


class Peer(PeerBase):
    public_id: str = Field(serialization_alias='id')
    name: str
    workspace_id: str
    created_at: datetime.datetime
    h_metadata: dict = Field(default={}, serialization_alias='metadata')

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    )


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: Annotated[str, Field(min_length=0, max_length=50000)]
    sender_id: str
    metadata: dict = {}


class MessageGet(MessageBase):
    filter: dict | None = None


class MessageUpdate(MessageBase):
    metadata: dict


class Message(MessageBase):
    public_id: str = Field(serialization_alias='id')
    content: str
    sender_id: str
    session_id: str | None
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime
    workspace_id: str

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    )


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    metadata: dict = {}


class SessionGet(SessionBase):
    filter: dict | None = None
    is_active: bool = False


class SessionUpdate(SessionBase):
    metadata: dict


class Session(SessionBase):
    public_id: str = Field(serialization_alias='id')
    is_active: bool
    workspace_id: str
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    )




class CollectionBase(BaseModel):
    pass


class CollectionCreate(CollectionBase):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    metadata: dict = {}

    @field_validator("name")
    def validate_name(cls, v):
        if v.lower() == "honcho":
            raise ValueError("Collection name cannot be 'honcho'")
        return v


class CollectionGet(CollectionBase):
    filter: dict | None = None


class CollectionUpdate(CollectionBase):
    name: str | None = None
    metadata: dict | None = None

    @field_validator("name")
    def validate_name(cls, v):
        if v is not None and v.lower() == "honcho":
            raise ValueError("Collection name cannot be 'honcho'")
        return v


class Collection(CollectionBase):
    public_id: str = Field(serialization_alias='id')
    name: str
    peer_id: str
    workspace_id: str
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    )


class DocumentBase(BaseModel):
    pass


class DocumentCreate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    metadata: dict = {}


class DocumentGet(DocumentBase):
    filter: dict | None = None


class DocumentQuery(DocumentBase):
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    filter: dict | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class DocumentUpdate(DocumentBase):
    metadata: dict | None = Field(None, max_length=10000)
    content: Annotated[str | None, Field(min_length=1, max_length=100000)] = None


class Document(DocumentBase):
    public_id: str = Field(serialization_alias='id')
    content: str
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime
    collection_id: str
    workspace_id: str
    peer_id: str

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    )


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
