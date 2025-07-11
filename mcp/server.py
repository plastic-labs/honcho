import logging
import os
import uuid

from dotenv import load_dotenv
from fastmcp import FastMCP
from honcho import AsyncHoncho

load_dotenv()


# Basic configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Get a logger instance
logger = logging.getLogger(__name__)

mcp = FastMCP("Honcho MCP Server", dependencies=["honcho-ai"])

# Configuration
honcho = AsyncHoncho(environment="production")

ASSISTANT_NAME = os.environ.get("HONCHO_ASSISTANT_NAME", "Assistant")

USER_NAME = os.environ.get("HONCHO_USER_NAME", "User")


@mcp.tool
async def start_conversation() -> str:
    """
    Start a new conversation with a user. Call this when a user starts a new conversation.

    Returns:
        A session ID for the conversation.
    """

    # Get/create the assistant peer with observe_me=False
    assistant_peer = await honcho.peer(ASSISTANT_NAME, config={"observe_me": False})

    # Create a new session
    session_id = str(uuid.uuid4())
    session = await honcho.session(session_id)

    # Add the user and assistant peers to the session
    session.add_peers([USER_NAME, assistant_peer])

    return session_id


@mcp.tool
async def add_turn(session_id: str, messages: list[dict[str, str]]) -> None:
    """
    Add a turn to a conversation. Call this after a a user has sent a message and the assistant has responded.

    Args:
        session_id: The ID of the session to add the turn to.
        messages: A list of messages to add to the session. Each message is a dictionary with the following keys:
            - role: The role of the message author. Must be one of "user" or "assistant".
            - content: The content of the message.
            - metadata: *Optional* metadata about the message.

    Returns:
        None
    """
    session = await honcho.session(session_id)

    user_peer = await honcho.peer(USER_NAME)
    assistant_peer = await honcho.peer(ASSISTANT_NAME)

    session_messages = []

    for message in messages:
        if message["role"] == "user":
            session_messages.append(user_peer.message(message["content"]))
        elif message["role"] == "assistant":
            session_messages.append(assistant_peer.message(message["content"]))
        else:
            raise ValueError(f"Invalid role: {message['role']}")

    await session.add_messages(session_messages)


@mcp.tool
async def get_personalization_insights(query: str) -> str:
    """
    Get personalization insights about the user, based on the query and the accumulated knowledge of the user
    across all conversations.

    Args:
        session_id: The ID of the session to get personalization insights for.
        query: The question about the user's preferences, habits, etc.

    Returns:
        A string with the personalization insights.
    """
    user_peer = await honcho.peer(USER_NAME)

    # Get the personalization insights
    personalization_insights = await user_peer.chat(query)

    if personalization_insights is None:
        return "No personalization insights found."

    return personalization_insights


if __name__ == "__main__":
    mcp.run()
