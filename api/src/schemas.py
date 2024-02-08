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
    session_data: dict | None = None


class SessionUpdate(SessionBase):
    session_data: dict | None = None


class Session(SessionBase):
    id: uuid.UUID
    # messages: list[Message]
    is_active: bool
    user_id: str
    location_id: str
    app_id: str
    session_data: str
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
