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
    """Converts Honcho messages to Langchain messages"""
    from langchain_core.messages import AIMessage, HumanMessage  # type: ignore

    new_messages = []
    for message in messages:
        if message.is_user:
            new_messages.append(HumanMessage(content=message.content))
        else:
            new_messages.append(AIMessage(content=message.content))
    return new_messages


@requires_langchain
def _langchain_to_messages(messages, session: Union[Session, AsyncSession]):
    """Converts Langchain messages to Langchain messages"""
    from langchain_core.messages import HumanMessage  # type: ignore

    for message in messages:
        if isinstance(message, HumanMessage):
            session.create_message(
                is_user=True, content=message.content, metadata=message.metadata
            )
        else:
            session.create_message(
                is_user=False, content=message.content, metadata=message.metadata
            )
