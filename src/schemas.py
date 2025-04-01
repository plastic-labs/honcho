import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppBase(BaseModel):
    pass


class AppCreate(AppBase):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    metadata: dict = {}


class AppUpdate(AppBase):
    name: str | None = None
    metadata: dict | None = None


class App(AppBase):
    public_id: str = Field(exclude=True)
    id: str
    name: str
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    @field_validator("id", mode="before")
    def internal_to_public(cls, value, info):
        return info.data.get("public_id", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata", "public_id"]},
    )


class UserBase(BaseModel):
    pass


class UserCreate(UserBase):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    metadata: dict = {}


class UserGet(UserBase):
    filter: dict | None = None


class UserUpdate(UserBase):
    name: str | None = None
    metadata: dict | None = None  # Allow user to explicitly set metadata to empty


class User(UserBase):
    public_id: str = Field(exclude=True)
    id: str
    name: str
    app_id: str
    created_at: datetime.datetime
    h_metadata: dict = Field(exclude=True)
    metadata: dict

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    @field_validator("id", mode="before")
    def internal_to_public(cls, value, info):
        return info.data.get("public_id", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata", "public_id"]},
    )


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: Annotated[str, Field(min_length=0, max_length=50000)]
    is_user: bool
    metadata: dict = {}


class MessageGet(MessageBase):
    filter: dict | None = None


class MessageUpdate(MessageBase):
    metadata: dict


class Message(MessageBase):
    public_id: str = Field(exclude=True)
    id: str
    content: str
    is_user: bool
    session_id: str
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    @field_validator("id", mode="before")
    def internal_to_public(cls, value, info):
        return info.data.get("public_id", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata", "public_id"]},
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
    public_id: str = Field(exclude=True)
    id: str
    # messages: list[Message]
    is_active: bool
    user_id: str
    h_metadata: dict = Field(exclude=True)
    metadata: dict

    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    @field_validator("id", mode="before")
    def internal_to_public(cls, value, info):
        return info.data.get("public_id", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata", "public_id"]},
    )


class MetamessageBase(BaseModel):
    pass


class MetamessageCreate(MetamessageBase):
    metamessage_type: Annotated[str, Field(min_length=1, max_length=50)]
    content: Annotated[str, Field(min_length=0, max_length=50000)]
    user_id: str | None = None  # Will be set from URL parameter in endpoint
    session_id: str | None = None
    message_id: str | None = None
    metadata: dict = {}


class MetamessageGet(MetamessageBase):
    metamessage_type: str | None = None
    user_id: str | None = None  # Can be provided in URL or body
    session_id: str | None = None
    message_id: str | None = None
    filter: dict | None = None


class MetamessageUpdate(MetamessageBase):
    user_id: str | None = None  # Will be set from URL parameter in endpoint
    session_id: str | None = None 
    message_id: str | None = None
    metamessage_type: str | None = None
    metadata: dict | None = None


class Metamessage(MetamessageBase):
    public_id: str = Field(exclude=True)
    id: str
    metamessage_type: str
    content: str
    user_id: str
    session_id: str | None
    message_id: str | None
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    @field_validator("id", mode="before")
    def internal_to_public(cls, value, info):
        return info.data.get("public_id", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata", "public_id"]},
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
    public_id: str = Field(exclude=True)
    id: str
    name: str
    user_id: str
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    @field_validator("id", mode="before")
    def internal_to_public(cls, value, info):
        return info.data.get("public_id", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata", "public_id"]},
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
    public_id: str = Field(exclude=True)
    id: str
    content: str
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime
    collection_id: str

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    @field_validator("id", mode="before")
    def internal_to_public(cls, value, info):
        return info.data.get("public_id", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata", "public_id"]},
    )


class AgentQuery(BaseModel):
    queries: str | list[str]

    @field_validator('queries')
    def validate_queries(cls, v):
        MAX_STRING_LENGTH = 10000
        MAX_LIST_LENGTH = 25
        if isinstance(v, str):
            if len(v) > MAX_STRING_LENGTH:
                raise ValueError('Query too long')
        elif isinstance(v, list):
            if len(v) > MAX_LIST_LENGTH:
                raise ValueError('Too many queries')
            if any(len(q) > MAX_STRING_LENGTH for q in v):
                raise ValueError('One or more queries too long')
        return v

class AgentChat(BaseModel):
    content: str


class MessageBatchCreate(BaseModel):
    """Schema for batch message creation with a max of 100 messages"""
    messages: list[MessageCreate] = Field(..., max_length=100)
