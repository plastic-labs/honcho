"""
Utilities to integrate Honcho with Langchain projects
"""

import functools
import importlib
from typing import List, Union

from honcho import AsyncSession, Session
from honcho.schemas import Message


def requires_langchain(func):
    """A utility to check if langchain is installed before running a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Check if langchain is installed before running a function"""

        if importlib.util.find_spec("langchain") is None:  # type: ignore
            raise ImportError("Langchain must be installed to use this feature")
            # raise RuntimeError("langchain is not installed")
        return func(*args, **kwargs)

    return wrapper


@requires_langchain
def _messages_to_langchain(messages: List[Message]):
    """Converts Honcho messages to Langchain messages

    Args:
        messages (List[Message]): The list of messages to convert

    Returns:
        List: The list of converted LangChain messages

    """
    from langchain_core.messages import AIMessage, HumanMessage  # type: ignore

    new_messages = []
    for message in messages:
        if message.is_user:
            new_messages.append(HumanMessage(content=message.content))
        else:
            new_messages.append(AIMessage(content=message.content))
    return new_messages


@requires_langchain
def _langchain_to_messages(
    messages, session: Union[Session, AsyncSession]
) -> List[Message]:
    """Converts Langchain messages to Honcho messages and adds to appropriate session

    Args:
        messages: The LangChain messages to convert
        session: The session to add the messages to

    Returns:
        List[Message]: The list of converted messages

    """
    from langchain_core.messages import HumanMessage  # type: ignore

    messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            message = session.create_message(
                is_user=True, content=message.content, metadata=message.metadata
            )
            messages.append(message)
        else:
            message = session.create_message(
                is_user=False, content=message.content, metadata=message.metadata
            )
            messages.append(message)
    return messages
