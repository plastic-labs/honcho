"""Message class for Honcho SDK."""

from __future__ import annotations

import datetime
from typing import Any

from .api_types import MessageResponse


class Message:
    """
    A message in a Honcho session.

    Messages represent communication between peers within a session.
    This class wraps the API response with convenient attribute access.

    Attributes:
        id: Unique identifier for this message
        content: The message content
        peer_id: The peer ID who authored this message
        session_id: The session ID this message belongs to
        workspace_id: The workspace ID this message belongs to
        metadata: Metadata associated with this message
        created_at: Timestamp for when the message was created
        token_count: Number of tokens in this message
    """

    id: str
    content: str
    peer_id: str
    session_id: str
    workspace_id: str
    metadata: dict[str, Any]
    created_at: datetime.datetime
    token_count: int

    def __init__(
        self,
        id: str,
        content: str,
        peer_id: str,
        session_id: str,
        workspace_id: str,
        metadata: dict[str, Any],
        created_at: datetime.datetime,
        token_count: int,
    ) -> None:
        self.id = id
        self.content = content
        self.peer_id = peer_id
        self.session_id = session_id
        self.workspace_id = workspace_id
        self.metadata = metadata
        self.created_at = created_at
        self.token_count = token_count

    @classmethod
    def from_api_response(cls, data: MessageResponse) -> "Message":
        """Create a Message from an API response."""
        return cls(
            id=data.id,
            content=data.content,
            peer_id=data.peer_id,
            session_id=data.session_id,
            workspace_id=data.workspace_id,
            metadata=data.metadata,
            created_at=data.created_at,
            token_count=data.token_count,
        )

    def __repr__(self) -> str:
        truncated = (
            f"{self.content[:50]}..." if len(self.content) > 50 else self.content
        )
        return (
            f"Message(id='{self.id}', peer_id='{self.peer_id}', content='{truncated}')"
        )

    def __str__(self) -> str:
        return self.content
