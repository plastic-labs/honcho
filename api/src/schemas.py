from pydantic import BaseModel


class MessageBase(BaseModel):
    content: str
    is_user: bool


class MessageCreate(MessageBase):
    pass


class Message(MessageBase):
    session_id: int
    id: int

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
    id: int
    messages: list[Message]
    is_active: bool
    user_id: str
    location_id: str
    session_data: str

    class Config:
        orm_mode = True


class MetamessagesBase(BaseModel):
    metamessage_type: str
    content: str


class MetamessagesCreate(MetamessagesBase):
    pass


class Metamessages(MetamessagesBase):
    id: int

    class Config:
        orm_mode = True
