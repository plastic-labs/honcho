import datetime

from pydantic import BaseModel, ConfigDict, Field


class AppBase(BaseModel):
    pass


class AppCreate(AppBase):
    name: str
    metadata: dict | None = {}


class AppUpdate(AppBase):
    name: str | None = None
    metadata: dict | None = None


class App(AppBase):
    id: str = Field(alias="public_id")
    name: str
    metadata: dict = Field(alias="h_metadata")
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
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
    id: str = Field(alias="public_id")
    name: str
    app_id: str
    created_at: datetime.datetime
    metadata: dict = Field(alias="h_metadata")

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
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
    id: str = Field(alias="public_id")
    content: str
    is_user: bool
    session_id: str
    metadata: dict = Field(alias="h_metadata")
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    metadata: dict | None = {}


class SessionUpdate(SessionBase):
    metadata: dict | None = None


class Session(SessionBase):
    id: str = Field(alias="public_id")
    # messages: list[Message]
    is_active: bool
    user_id: str
    metadata: dict = Field(alias="h_metadata")
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
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
    id: str = Field(alias="public_id")
    metamessage_type: str
    content: str
    message_id: str
    metadata: dict = Field(alias="h_metadata")
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
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
    id: str = Field(alias="public_id")
    name: str
    user_id: str
    metadata: dict = Field(alias="h_metadata")
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
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
    id: str = Field(alias="public_id")
    content: str
    metadata: dict = Field(alias="h_metadata")
    created_at: datetime.datetime
    collection_id: str

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class AgentQuery(BaseModel):
    queries: str | list[str]
    collections: str | list[str] = "honcho"


class AgentChat(BaseModel):
    content: str
