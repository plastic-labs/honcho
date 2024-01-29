# TODO: Should conversations and messages also be used through via an adapter class like user model storage for more flexibility?

from __future__ import (
    annotations,
)  # required so that __getitem__ in ConversationHistory can return an instance of it's own class
from typing import TypedDict, Literal, Union

# Define a type for the role of the message sender.
Role = Literal["Assistant", "User"]


class Message(TypedDict):
    """
    Typed dictionary to represent a message.

    Attributes:
        role (Role): The role of the message sender, either 'AI' or 'User'.
        content (str): The text content of the message.
    """

    role: Role
    content: str


class ConversationHistory:
    """
    Represents a conversation consisting of a list of messages.

    Attributes:
        (Optional) messages (list[Message]): A list of messages that make up the conversation. Conversation will be initalized as empty if not provided
    """

    def __init__(self, messages: list[Message] = None):
        """
        Initializes a new Conversation instance.

        Args:
            messages (list[Message], optional): An initial list of messages. Defaults to None.
        """
        self.messages = messages if messages is not None else []

    @classmethod
    def from_honcho_dicts(cls, messages: list[dict[str, str]]):
        messages = [
            {
                "role": "User" if message["is_user"] else "Assistant",
                "content": message["content"],
            }
            for message in messages
        ]

        return cls(messages)

    def __getitem__(
        self, key: Union[slice, int]
    ) -> Union[ConversationHistory, Message]:
        """
        Allows indexing or slicing into the ConversationHistory to select a single message or a range of messages.
        If a single index is provided, a Message object is returned. If a slice is provided, a new ConversationHistory
        object containing the selected range of messages is returned.

        Args:
            key (Union[slice, int]): The index or slice object indicating the message(s) to retrieve.

        Returns:
            Union[ConversationHistory, Message]: A new ConversationHistory object containing the selected range of messages
            if a slice is provided, or a single Message object if an index is provided.
        """
        if isinstance(key, slice):
            selected_messages = self.messages[key]
            return ConversationHistory(messages=selected_messages)
        elif isinstance(key, int):
            # Return a single message at the provided index
            return self.messages[key]
        else:
            raise TypeError("Invalid argument type. Expected slice or int.")

    def __repr__(self) -> str:
        """
        Represents the conversation history as a string with each message on a new line.
        E.g. User: ..., AI: ...

        Returns:
            str: A string representation of the conversation history.
        """
        conversation_str = ""
        for message in self.messages:
            speaker = message["role"]
            conversation_str += f"{speaker}: {message['content']}\n"
        return conversation_str.strip()

    def add_ai_message(self, content: str) -> None:
        """
        Appends an AI message to the conversation.

        Args:
            content (str): The text content of the AI's message.
        """
        self.messages.append(Message(role="Assistant", content=content))

    def add_user_message(self, content: str) -> None:
        """
        Appends a user message to the conversation.

        Args:
            content (str): The text content of the user's message.
        """
        self.messages.append(Message(role="User", content=content))
