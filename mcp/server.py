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

mcp = FastMCP("Honcho MCP Server")

# Configuration
honcho = AsyncHoncho()

ASSISTANT_NAME = os.environ.get("HONCHO_ASSISTANT_NAME", "Assistant")


@mcp.tool
async def start_conversation(user_name: str, initial_message: str | None = None) -> str:
    """
    Start a new conversation with a user. Call this when a user starts a new conversation.

    Args:
        user_name: The name of the user to start a conversation with.
        initial_message: An optional initial message from the user.

    Returns:
        A session ID for the conversation.
    """
    # Get/create the assistant peer with observe_me=False
    assistant_peer = await honcho.peer(ASSISTANT_NAME, config={"observe_me": False})

    # Create a new session
    session_id = str(uuid.uuid4())
    session = await honcho.session(session_id)

    # Add the user and assistant peers to the session
    session.add_peers([user_name, assistant_peer])

    # Add a message from the user
    if initial_message:
        user_peer = await honcho.peer(user_name)
        session.add_messages([user_peer.message(initial_message)])

    return session_id


if __name__ == "__main__":
    mcp.run()
