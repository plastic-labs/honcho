import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class AppBase(BaseModel):
    pass


class AppCreate(AppBase):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    metadata: dict = {}


class AppGet(AppBase):
    filter: dict | None = None


class AppUpdate(AppBase):
    name: str | None = None
    metadata: dict | None = None


class App(AppBase):
    public_id: str = Field(serialization_alias='id')
    name: str
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
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
    metadata: dict | None = None


class User(UserBase):
    public_id: str = Field(serialization_alias='id')
    name: str
    app_id: str
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
    is_user: bool
    metadata: dict = {}


class MessageGet(MessageBase):
    filter: dict | None = None


class MessageUpdate(MessageBase):
    metadata: dict


class Message(MessageBase):
    public_id: str = Field(serialization_alias='id')
    content: str
    is_user: bool
    session_id: str
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime
    app_id: str
    user_id: str

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
    user_id: str
    app_id: str
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    )


class MetamessageBase(BaseModel):
    pass


class MetamessageCreate(MetamessageBase):
    label: Annotated[str, Field(min_length=1, max_length=50, alias='metamessage_type')]
    content: Annotated[str, Field(min_length=0, max_length=50000)]
    session_id: str | None = None
    message_id: str | None = None
    metadata: dict = {}

    model_config = ConfigDict(populate_by_name=True)


class MetamessageGet(MetamessageBase):
    label: str | None = Field(default=None, alias='metamessage_type')
    session_id: str | None = None
    message_id: str | None = None
    filter: dict | None = None

    model_config = ConfigDict(populate_by_name=True)


class MetamessageUpdate(MetamessageBase):
    session_id: str | None = None
    message_id: str | None = None
    label: str | None = Field(default=None, alias='metamessage_type')
    metadata: dict | None = None

    model_config = ConfigDict(populate_by_name=True)


class Metamessage(MetamessageBase):
    public_id: str = Field(serialization_alias='id')
    label: str
    content: str
    user_id: str
    app_id: str
    session_id: str | None
    message_id: str | None
    h_metadata: dict = Field(default={}, serialization_alias='metadata')
    created_at: datetime.datetime

    # Included for backwards compatibility with the old metamessage_type field
    @computed_field
    @property
    def metamessage_type(self) -> str:
        return self.label

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
    user_id: str
    app_id: str
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
    app_id: str
    user_id: str

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



class DeriverStatus(BaseModel):
    """Schema for deriver status response"""
    
    unprocessed_count: int = Field(description="Number of unprocessed messages in the queue")
    processed_count: int = Field(description="Number of messages already processed")
    total_count: int = Field(description="Total number of messages in the queue")
    
    @computed_field
    @property
    def is_complete(self) -> bool:
        """Whether all messages have been processed"""
        return self.unprocessed_count == 0
    
    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Processing progress as a percentage (0-100)"""
        if self.total_count == 0:
            return 100.0
        return round((self.processed_count / self.total_count) * 100, 2)
    
