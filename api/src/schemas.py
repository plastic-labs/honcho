from pydantic import BaseModel
import datetime
import uuid


class MessageBase(BaseModel):
    content: str
    is_user: bool


class MessageCreate(MessageBase):
    pass


class Message(MessageBase):
    session_id: uuid.UUID
    id: uuid.UUID
    created_at: datetime.datetime

    class Config:
        orm_mode = True


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    location_id: str
    metadata: dict | None = None


class SessionUpdate(SessionBase):
    metadata: dict | None = None


class Session(SessionBase):
    id: uuid.UUID
    # messages: list[Message]
    is_active: bool
    user_id: str
    location_id: str
    app_id: str
    metadata: dict
    created_at: datetime.datetime

    class Config:
        orm_mode = True


class MetamessageBase(BaseModel):
    metamessage_type: str
    content: str


class MetamessageCreate(MetamessageBase):
    message_id: uuid.UUID


class Metamessage(MetamessageBase):
    id: uuid.UUID
    message_id: uuid.UUID
    created_at: datetime.datetime

    class Config:
        orm_mode = True

class CollectionBase(BaseModel):
    pass

class CollectionCreate(CollectionBase):
    name: str

class CollectionUpdate(CollectionBase):
    name: str

class Collection(CollectionBase):
    id: uuid.UUID
    name: str
    app_id: str
    user_id: str
    created_at: datetime.datetime

    class Config:
        orm_mode = True

class DocumentBase(BaseModel):
    content: str
    collection_id: uuid.UUID

class DocumentCreate(DocumentBase):
    metadata: dict | None = None

class DocumentUpdate(DocumentBase):
    metadata: dict | None = None
    content: str | None = None

class Document(DocumentBase):
    id: uuid.UUID
    content: str
    metadata: dict | None = None
    created_at: datetime.datetime

    class Config:
        orm_mode = True

