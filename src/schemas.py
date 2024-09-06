import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppBase(BaseModel):
    pass


class AppCreate(AppBase):
    name: str
    metadata: dict | None = {}


class AppUpdate(AppBase):
    name: str | None = None
    metadata: dict | None = None


class App(AppBase):
    id: uuid.UUID
    name: str
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata"]},
    )


class UserBase(BaseModel):
    pass


class UserCreate(UserBase):
    name: str
    metadata: dict | None = {}


class UserUpdate(UserBase):
    name: str | None = None
    metadata: dict | None = None


class User(UserBase):
    id: uuid.UUID
    name: str
    app_id: uuid.UUID
    created_at: datetime.datetime
    h_metadata: dict = Field(exclude=True)
    metadata: dict

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata"]},
    )


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: str
    is_user: bool
    metadata: dict | None = {}


class MessageUpdate(MessageBase):
    metadata: dict | None = None


class Message(MessageBase):
    content: str
    is_user: bool
    session_id: uuid.UUID
    id: uuid.UUID
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata"]},
    )


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    metadata: dict | None = {}


class SessionUpdate(SessionBase):
    metadata: dict | None = None


class Session(SessionBase):
    id: uuid.UUID
    # messages: list[Message]
    is_active: bool
    user_id: uuid.UUID
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata"]},
    )


class MetamessageBase(BaseModel):
    pass


class MetamessageCreate(MetamessageBase):
    metamessage_type: str
    content: str
    message_id: uuid.UUID
    metadata: dict | None = {}


class MetamessageUpdate(MetamessageBase):
    message_id: uuid.UUID
    metamessage_type: str | None = None
    metadata: dict | None = None


class Metamessage(MetamessageBase):
    metamessage_type: str
    content: str
    id: uuid.UUID
    message_id: uuid.UUID
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata"]},
    )


class CollectionBase(BaseModel):
    pass


class CollectionCreate(CollectionBase):
    name: str
    metadata: dict | None = {}


class CollectionUpdate(CollectionBase):
    name: str | None = None
    metadata: dict | None = None


class Collection(CollectionBase):
    id: uuid.UUID
    name: str
    user_id: uuid.UUID
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata"]},
    )


class DocumentBase(BaseModel):
    pass


class DocumentCreate(DocumentBase):
    content: str
    metadata: dict | None = {}


class DocumentUpdate(DocumentBase):
    metadata: dict | None = None
    content: str | None = None


class Document(DocumentBase):
    id: uuid.UUID
    content: str
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime
    collection_id: uuid.UUID

    @field_validator("metadata", mode="before")
    def fetch_h_metadata(cls, value, info):
        return info.data.get("h_metadata", {})

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={"exclude": ["h_metadata"]},
    )


class AgentQuery(BaseModel):
    queries: str | list[str]
    collections: str | list[str] = "honcho"


class AgentChat(BaseModel):
    content: str
