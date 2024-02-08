

class Message:
    def __init__(self, session_id: int, id: int, is_user: bool, content: str):
        """Constructor for Message"""
        self.session_id = session_id
        self.id = id
        self.is_user = is_user
        self.content = content

    def __str__(self):
        return f"Message(id={self.id}, is_user={self.is_user}, content={self.content})"
