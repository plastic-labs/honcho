from datetime import datetime
from typing import Any

from honcho_core import Honcho as HonchoCore
from honcho_core.types.workspaces.sessions.message import Message as MessageCore
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call


class Message(BaseModel):
    """
    Represents a message in Honcho.

    Messages are scoped to sessions and provide full message data plus SDK methods.

    Attributes:
        id: Unique identifier for this message
        content: The message content
        created_at: When the message was created
        peer_id: ID of the peer who created the message
        session_id: ID of the session this message belongs to
        token_count: Number of tokens in the message
        workspace_id: Workspace ID for scoping operations
        metadata: Optional metadata dictionary
        _client: Reference to the parent Honcho client instance
    """

    id: str = Field(..., min_length=1, description="Unique identifier for this message")
    content: str = Field(..., description="The message content")
    created_at: datetime = Field(..., description="When the message was created")
    peer_id: str = Field(
        ..., min_length=1, description="ID of the peer who created the message"
    )
    session_id: str = Field(
        ..., min_length=1, description="ID of the session this message belongs to"
    )
    token_count: int = Field(..., ge=0, description="Number of tokens in the message")
    workspace_id: str = Field(
        ..., min_length=1, description="Workspace ID for scoping operations"
    )
    metadata: dict[str, object] = Field(
        default_factory=dict, description="Optional metadata dictionary"
    )

    _client: HonchoCore = PrivateAttr()

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        message_id: str = Field(
            ..., min_length=1, description="Unique identifier for this message"
        ),
        content: str = Field(..., description="The message content"),
        created_at: datetime = Field(..., description="When the message was created"),
        peer_id: str = Field(
            ..., min_length=1, description="ID of the peer who created the message"
        ),
        session_id: str = Field(
            ..., min_length=1, description="ID of the session this message belongs to"
        ),
        token_count: int = Field(
            ..., ge=0, description="Number of tokens in the message"
        ),
        workspace_id: str = Field(
            ..., min_length=1, description="Workspace ID for scoping operations"
        ),
        client: Any = Field(
            ..., description="Reference to the parent Honcho client instance"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None, description="Optional metadata dictionary"
        ),
    ) -> None:
        """
        Initialize a Message.

        Args:
            message_id: Unique identifier for this message
            content: The message content
            created_at: When the message was created
            peer_id: ID of the peer who created the message
            session_id: ID of the session this message belongs to
            token_count: Number of tokens in the message
            workspace_id: Workspace ID for scoping operations
            client: Reference to the parent Honcho client instance
            metadata: Optional metadata dictionary
        """
        super().__init__(
            id=message_id,
            content=content,
            created_at=created_at,
            peer_id=peer_id,
            session_id=session_id,
            token_count=token_count,
            workspace_id=workspace_id,
            metadata=metadata or {},
        )
        self._client = client

    @classmethod
    def from_core(cls, core_message: MessageCore, client: HonchoCore) -> "Message":
        """
        Create SDK Message from core Message.

        This is the primary way Message objects are created from API responses.

        Args:
            core_message: Core message object from honcho_core
            client: Reference to the parent Honcho client instance

        Returns:
            SDK Message object with all data and methods available
        """
        return cls(
            message_id=core_message.id,
            content=core_message.content,
            created_at=core_message.created_at,
            peer_id=core_message.peer_id,
            session_id=core_message.session_id,
            token_count=core_message.token_count,
            workspace_id=core_message.workspace_id,
            client=client,
            metadata=core_message.metadata,
        )

    @validate_call
    def update(self, metadata: dict[str, Any]) -> Any:
        """
        Update metadata for this message.

        Makes an API call to update the metadata associated with this message.
        This will overwrite any existing metadata with the provided values.

        Args:
            metadata: A dictionary of metadata to associate with the message.
                Keys must be strings, values can be any JSON-serializable type

        Returns:
            The updated Message object (self)
        """
        self._client.workspaces.sessions.messages.update(
            message_id=self.id,
            session_id=self.session_id,
            workspace_id=self.workspace_id,
            metadata=metadata,
        )

        # Update local copy
        self.metadata = metadata
        return self

    def __repr__(self) -> str:
        """
        Return a string representation of the Message.

        Returns:
            A string representation suitable for debugging
        """
        return f"Message(id='{self.id}', session_id='{self.session_id}', workspace_id='{self.workspace_id}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Message.

        Returns:
            A string representation suitable for debugging
        """
        return f"Message(id='{self.id}', session_id='{self.session_id}', workspace_id='{self.workspace_id}')"
