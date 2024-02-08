import uuid
import datetime

class Message:
    def __init__(self, session_id: uuid.UUID, id: uuid.UUID, is_user: bool, content: str, created_at: datetime.datetime):
        """Constructor for Message"""
        self.session_id = session_id
        self.id = id
        self.is_user = is_user
        self.content = content
        self.created_at = created_at

    def __str__(self):
        return f"Message(id={self.id}, is_user={self.is_user}, content={self.content})"

class Metamessage:
    def __init__(self, id: uuid.UUID, message_id: uuid.UUID, metamessage_type: str, content: str, created_at: datetime.datetime):
        """Constructor for Metamessage"""
        self.id = id
        self.message_id = message_id
        self.metamessage_type = metamessage_type
        self.content = content
        self.created_at = created_at

    def __str__(self):
        return f"Metamessage(id={self.id}, message_id={self.message_id}, metamessage_type={self.metamessage_type}, content={self.content})"
