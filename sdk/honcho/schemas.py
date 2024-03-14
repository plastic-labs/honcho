import uuid
import datetime


class Message:
    def __init__(
        self,
        session_id: uuid.UUID,
        id: uuid.UUID,
        is_user: bool,
        content: str,
        metadata: dict,
        created_at: datetime.datetime,
    ):
        """Constructor for Message"""
        self.session_id = session_id
        self.id = id
        self.is_user = is_user
        self.content = content
        self.metadata = metadata
        self.created_at = created_at

    def __str__(self):
        return f"Message(id={self.id}, is_user={self.is_user}, content={self.content})"


class Metamessage:
    def __init__(
        self,
        id: uuid.UUID,
        message_id: uuid.UUID,
        metamessage_type: str,
        content: str,
        metadata: dict,
        created_at: datetime.datetime,
    ):
        """Constructor for Metamessage"""
        self.id = id
        self.message_id = message_id
        self.metamessage_type = metamessage_type
        self.content = content
        self.metadata = metadata
        self.created_at = created_at

    def __str__(self):
        return f"Metamessage(id={self.id}, message_id={self.message_id}, metamessage_type={self.metamessage_type}, content={self.content})"


class Document:
    def __init__(
        self,
        id: uuid.UUID,
        collection_id: uuid.UUID,
        content: str,
        metadata: dict,
        created_at: datetime.datetime,
    ):
        """Constructor for Document"""
        self.collection_id = collection_id
        self.id = id
        self.content = content
        self.metadata = metadata
        self.created_at = created_at

    def __str__(self) -> str:
        return f"Document(id={self.id}, metadata={self.metadata}, content={self.content}, created_at={self.created_at})"
