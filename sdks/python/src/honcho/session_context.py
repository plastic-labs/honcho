from typing import TYPE_CHECKING, Union

from honcho_core.types.workspaces.sessions.message import Message
from pydantic import BaseModel, Field, validate_call

if TYPE_CHECKING:
    from .peer import Peer


class SessionContext(BaseModel):
    """
    Represents the context of a session containing a curated list of messages.

    The SessionContext provides methods to convert message history into formats
    compatible with different LLM providers while staying within token limits
    and providing optimal conversation context.

    Attributes:
        messages: List of Message objects representing the conversation context
    """

    session_id: str = Field(
        ..., description="ID of the session this context belongs to"
    )
    messages: list[Message] = Field(
        ..., description="List of Message objects to include in the context"
    )
    summary: str = Field(
        ..., description="Summary of the session history prior to the message cutoff"
    )

    @validate_call
    def __init__(
        self,
        session_id: str = Field(
            ..., description="ID of the session this context belongs to"
        ),
        messages: list[Message] = Field(
            ..., description="List of Message objects to include in the context"
        ),
        summary: str = Field(
            ...,
            description="Summary of the session history prior to the message cutoff",
        ),
    ) -> None:
        """
        Initialize a new SessionContext.

        Args:
            messages: List of Message objects to include in the context
        """
        super().__init__(
            session_id=session_id,
            messages=messages,
            summary=summary,
        )

    def to_openai(
        self,
        *,
        assistant: Union[str, "Peer"],
    ) -> list[dict[str, object]]:
        """
        Convert the context to OpenAI-compatible message format.

        Transforms the message history into the format expected by OpenAI's
        Chat Completions API, with proper role assignments based on the
        assistant's identity.

        Args:
            assistant: The assistant peer (Peer object or peer ID string) to use
                       for determining message roles. Messages from this peer will
                       be marked as "assistant", others as "user"

        Returns:
            A list of dictionaries in OpenAI format, where each dictionary contains
            "role" and "content" keys suitable for the OpenAI API

        Raises:
            ValidationError: If assistant parameter is invalid
        """
        assistant_id = assistant.id if isinstance(assistant, Peer) else assistant
        return [
            {
                "role": "assistant" if message.peer_id == assistant_id else "user",
                "content": message.content,
            }
            for message in self.messages
        ]

    def to_anthropic(
        self,
        *,
        assistant: Union[str, "Peer"],
    ) -> list[dict[str, object]]:
        """
        Convert the context to Anthropic-compatible message format.

        Transforms the message history into the format expected by Anthropic's
        Claude API. TODO: Anthropic requires messages to alternate between
        user and assistant roles, so this method may need to handle role
        consolidation or filtering in the future.

        Args:
            assistant: The assistant peer (Peer object or peer ID string) to use
                       for determining message roles. Messages from this peer will
                       be marked as "assistant", others as "user"

        Returns:
            A list of dictionaries in Anthropic format, where each dictionary contains
            "role" and "content" keys suitable for the Anthropic API

        Raises:
            ValidationError: If assistant parameter is invalid

        Note:
            Future versions may implement role alternation requirements for
            Anthropic's API compatibility
        """
        assistant_id = assistant.id if isinstance(assistant, Peer) else assistant
        return [
            {
                "role": "assistant" if message.peer_id == assistant_id else "user",
                "content": message.content,
            }
            for message in self.messages
        ]

    def __len__(self) -> int:
        """
        Return the number of messages in the context.

        Returns:
            The number of messages in this context
        """
        return len(self.messages)

    def __repr__(self) -> str:
        """
        Return a string representation of the SessionContext.

        Returns:
            A string representation suitable for debugging
        """
        return f"SessionContext(messages={len(self.messages)})"
