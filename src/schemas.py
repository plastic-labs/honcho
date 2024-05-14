import datetime
import uuid

from pydantic import BaseModel, Field, validator


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

    @validator("metadata", pre=True, allow_reuse=True)
    def fetch_h_metadata(cls, value, values):
        if "h_metadata" in values:
            return values["h_metadata"]
        return {}

    class Config:
        from_attributes = True
        json_schema_extra = {"exclude": ["h_metadata"]}


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

    @validator("metadata", pre=True, allow_reuse=True)
    def fetch_h_metadata(cls, value, values):
        if "h_metadata" in values:
            return values["h_metadata"]
        return {}

    class Config:
        from_attributes = True
        json_schema_extra = {"exclude": ["h_metadata"]}


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

    @validator("metadata", pre=True, allow_reuse=True)
    def fetch_h_metadata(cls, value, values):
        if "h_metadata" in values:
            return values["h_metadata"]
        return {}

    class Config:
        from_attributes = True
        json_schema_extra = {"exclude": ["h_metadata"]}


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    location_id: str
    metadata: dict | None = {}


class SessionUpdate(SessionBase):
    metadata: dict | None = None


class Session(SessionBase):
    id: uuid.UUID
    # messages: list[Message]
    is_active: bool
    user_id: uuid.UUID
    location_id: str
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @validator("metadata", pre=True, allow_reuse=True)
    def fetch_h_metadata(cls, value, values):
        if "h_metadata" in values:
            return values["h_metadata"]
        return {}

    class Config:
        from_attributes = True
        json_schema_extra = {"exclude": ["h_metadata"]}


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

    @validator("metadata", pre=True, allow_reuse=True)
    def fetch_h_metadata(cls, value, values):
        if "h_metadata" in values:
            return values["h_metadata"]
        return {}

    class Config:
        from_attributes = True
        json_schema_extra = {"exclude": ["h_metadata"]}


class CollectionBase(BaseModel):
    pass


class CollectionCreate(CollectionBase):
    name: str
    metadata: dict | None = {}


class CollectionUpdate(CollectionBase):
    name: str
    metadata: dict | None = None


class Collection(CollectionBase):
    id: uuid.UUID
    name: str
    user_id: uuid.UUID
    h_metadata: dict = Field(exclude=True)
    metadata: dict
    created_at: datetime.datetime

    @validator("metadata", pre=True, allow_reuse=True)
    def fetch_h_metadata(cls, value, values):
        if "h_metadata" in values:
            return values["h_metadata"]
        return {}

    class Config:
        from_attributes = True
        json_schema_extra = {"exclude": ["h_metadata"]}


class DocumentBase(BaseModel):
    content: str


class DocumentCreate(DocumentBase):
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

    @validator("metadata", pre=True, allow_reuse=True)
    def fetch_h_metadata(cls, value, values):
        if "h_metadata" in values:
            return values["h_metadata"]
        return {}

    class Config:
        from_attributes = True
        json_schema_extra = {"exclude": ["h_metadata"]}


class AgentChat(BaseModel):
    content: str
