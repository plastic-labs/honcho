import datetime

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
    name: str
    metadata: dict | None = {}


class UserUpdate(UserBase):
    name: str | None = None
    metadata: dict | None = None


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
    content: str
    is_user: bool
    metadata: dict | None = {}


class MessageUpdate(MessageBase):
    metadata: dict | None = None


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
    metadata: dict | None = {}


class SessionUpdate(SessionBase):
    metadata: dict | None = None


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
    metamessage_type: str
    content: str
    message_id: str
    metadata: dict | None = {}


class MetamessageUpdate(MetamessageBase):
    message_id: str
    metamessage_type: str | None = None
    metadata: dict | None = None


class Metamessage(MetamessageBase):
    public_id: str = Field(exclude=True)
    id: str
    metamessage_type: str
    content: str
    message_id: str
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
    name: str
    metadata: dict | None = {}


class CollectionUpdate(CollectionBase):
    name: str | None = None
    metadata: dict | None = None


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
    content: str
    metadata: dict | None = {}


class DocumentUpdate(DocumentBase):
    metadata: dict | None = None
    content: str | None = None


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
    # collections: str | list[str] = "honcho"


class AgentChat(BaseModel):
    content: str
