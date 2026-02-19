from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .message import Message

if TYPE_CHECKING:
    from .peer import Peer


class Summary(BaseModel):
    """Represents a summary of a session's conversation."""

    content: str = Field(..., description="The summary text")
    message_id: str = Field(
        ..., description="The ID of the message that this summary covers up to"
    )
    summary_type: str = Field(..., description="The type of summary (short or long)")
    created_at: str = Field(
        ..., description="The timestamp of when the summary was created (ISO format)"
    )
    token_count: int = Field(
        ..., description="The number of tokens in the summary text"
    )


class SessionSummaries(BaseModel):
    """Contains both short and long summaries for a session."""

    id: str = Field(..., description="The session ID")
    short_summary: Summary | None = Field(
        None, description="The short summary if available"
    )
    long_summary: Summary | None = Field(
        None, description="The long summary if available"
    )


class SessionContext(BaseModel):
    """
    Represents the context of a session containing a curated list of messages.

    The SessionContext provides methods to convert message history into formats
    compatible with different LLM providers while staying within token limits
    and providing optimal conversation context.

    Attributes:
        messages: List of Message objects representing the conversation context
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(
        ..., description="ID of the session this context belongs to"
    )
    messages: list[Message] = Field(
        ..., description="List of Message objects to include in the context"
    )
    summary: Summary | None = Field(
        None, description="Summary of the session history prior to the message cutoff"
    )
    peer_representation: str | None = Field(
        None,
        description="The peer representation, if context is requested from a specific perspective",
    )
    peer_card: list[str] | None = Field(
        None,
        description="The peer card, if context is requested from a specific perspective",
    )

    def to_openai(
        self,
        *,
        assistant: str | Peer,
    ) -> list[dict[str, str]]:
        """
        Convert the context to OpenAI-compatible message format.

        Transforms the message history and summary into the format expected by
        OpenAI's Chat Completions API, with proper role assignments based on the
        assistant's identity.

        Args:
            assistant: The assistant peer (Peer object or peer ID string) to use
            for determining message roles. Messages from this peer will be marked
            as "assistant", others as "user"

        Returns:
            A list of dictionaries in OpenAI format, where each dictionary contains
            "role" and "content" keys suitable for the OpenAI API
        """

        assistant_id = assistant if isinstance(assistant, str) else assistant.id
        messages = [
            {
                "role": "assistant" if message.peer_id == assistant_id else "user",
                "name": message.peer_id,
                "content": message.content,
            }
            for message in self.messages
        ]
        system_messages: list[dict[str, str]] = []

        if self.peer_representation:
            peer_representation_message = {
                "role": "system",
                "content": f"<peer_representation>{self.peer_representation}</peer_representation>",
            }
            system_messages.append(peer_representation_message)

        if self.peer_card:
            peer_card_message = {
                "role": "system",
                "content": f"<peer_card>{self.peer_card}</peer_card>",
            }
            system_messages.append(peer_card_message)

        if self.summary:
            summary_message = {
                "role": "system",
                "content": f"<summary>{self.summary.content}</summary>",
            }
            system_messages.append(summary_message)

        return system_messages + messages

    def to_anthropic(
        self,
        *,
        assistant: str | Peer,
    ) -> list[dict[str, str]]:
        """
        Convert the context to Anthropic-compatible message format.

        Transforms the message history into the format expected by Anthropic's
        Claude API, with proper role assignments based on the assistant's identity.

        Args:
            assistant: The assistant peer (Peer object or peer ID string) to use
                       for determining message roles. Messages from this peer will
                       be marked as "assistant", others as "user"

        Returns:
            A list of dictionaries in Anthropic format, where each dictionary contains
            "role" and "content" keys suitable for the Anthropic API

        Note:
            Future versions may implement role alternation requirements for
            Anthropic's API compatibility
        """

        assistant_id = assistant if isinstance(assistant, str) else assistant.id
        messages = [
            {
                "role": "assistant",
                "content": message.content,
            }
            if message.peer_id == assistant_id
            else {
                "role": "user",
                "content": f"{message.peer_id}: {message.content}",
            }
            for message in self.messages
        ]
        system_messages: list[dict[str, str]] = []

        if self.peer_representation:
            peer_representation_message = {
                "role": "user",
                "content": f"<peer_representation>{self.peer_representation}</peer_representation>",
            }
            system_messages.append(peer_representation_message)

        if self.peer_card:
            peer_card_message = {
                "role": "user",
                "content": f"<peer_card>{self.peer_card}</peer_card>",
            }
            system_messages.append(peer_card_message)

        if self.summary:
            summary_message = {
                "role": "user",
                "content": f"<summary>{self.summary.content}</summary>",
            }
            system_messages.append(summary_message)

        return system_messages + messages

    def __len__(self) -> int:
        """
        Return the number of messages in the context.

        Returns:
            The number of messages in this context
        """
        return len(self.messages) + (1 if self.summary else 0)

    def __repr__(self) -> str:
        """
        Return a string representation of the SessionContext.

        Returns:
            A string representation suitable for debugging
        """
        return f"SessionContext(messages={len(self.messages)}, summary={'present' if self.summary else 'None'})"
