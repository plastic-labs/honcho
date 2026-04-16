"""OpenAI Agents SDK integration with Honcho persistent memory.

Demonstrates a conversational agent that remembers users across sessions.
Honcho stores every message and builds a long-term representation of the user;
the agent injects that context into its instructions on every turn and can
query memory on demand via the ``query_memory`` tool.

Usage:
    python main.py

Environment variables:
    HONCHO_API_KEY      Required. Your Honcho API key from honcho.dev.
    HONCHO_WORKSPACE_ID Optional. Workspace ID (default: "default").
    OPENAI_API_KEY      Required. Your OpenAI API key.
"""

import asyncio
import logging
import uuid

from agents import Agent, RunContextWrapper, Runner
from honcho.http.exceptions import HonchoError

from tools.client import HonchoContext, get_client
from tools.get_context import get_context
from tools.query_memory import query_memory
from tools.save_memory import save_memory

logger = logging.getLogger(__name__)


def setup_session(user_id: str, session_id: str, assistant_id: str = "assistant") -> None:
    """Register peers in the session once at startup.

    Should be called once before the conversation loop begins. Calling
    ``add_peers`` on every turn is redundant — this ensures peers are
    registered exactly once.

    Args:
        user_id: Unique identifier for the user peer.
        session_id: Identifier for the conversation session.
        assistant_id: Peer ID for the assistant. Defaults to ``"assistant"``.

    Raises:
        RuntimeError: If the Honcho API call fails.
    """
    try:
        honcho = get_client()
        user_peer = honcho.peer(user_id)
        assistant_peer = honcho.peer(assistant_id)
        session = honcho.session(session_id)
        session.add_peers([user_peer, assistant_peer])
    except HonchoError as exc:
        raise RuntimeError("Failed to initialize Honcho session peers") from exc


honcho_agent = Agent[HonchoContext](
    name="HonchoMemoryAgent",
    instructions=(
        "You are a helpful assistant with persistent memory powered by Honcho. "
        "You remember users across conversations. "
        "When a user asks what you remember about them, use the query_memory tool."
    ),
    tools=[query_memory],
    model="gpt-4.1-mini",
)


async def chat(user_id: str, message: str, session_id: str) -> str:
    """Run one conversation turn with persistent Honcho memory.

    Saves the user message to Honcho, retrieves structured session history
    from Honcho and passes it directly to the SDK as prior messages, then
    saves the assistant reply.

    Args:
        user_id: Unique identifier for the user.
        message: The user's input message.
        session_id: Identifier for the current conversation session.

    Returns:
        The agent's response as a string.
    """
    ctx = HonchoContext(user_id=user_id, session_id=session_id)

    # Persist user message before the agent runs so it's available in context
    try:
        save_memory(user_id, message, "user", session_id)
    except HonchoError as exc:
        logger.warning("Could not persist user message: %s", exc)

    # Pass structured OpenAI-format history directly — don't flatten to plain text
    try:
        history = get_context(ctx, tokens=2000)
    except HonchoError as exc:
        logger.warning("Could not load Honcho context; continuing without history: %s", exc)
        history = []
    input_messages = history + [{"role": "user", "content": message}]

    result = await Runner.run(honcho_agent, input_messages, context=ctx)
    response = str(result.final_output)

    # Persist assistant response after the run
    try:
        save_memory(user_id, response, "assistant", session_id)
    except HonchoError as exc:
        logger.warning("Could not persist assistant message: %s", exc)

    return response


if __name__ == "__main__":
    print("HonchoMemoryAgent — type 'quit' to exit\n")
    # Replace "demo-user" with a real user identifier in production.
    _user_id = "demo-user"
    # A fresh session ID per run prevents history from accumulating across runs.
    _session_id = str(uuid.uuid4())

    # Register peers once at session start — not on every turn.
    setup_session(_user_id, _session_id)

    while True:
        _user_input = input("You: ").strip()
        if not _user_input:
            continue
        if _user_input.lower() in ("quit", "exit"):
            break
        _response = asyncio.run(chat(_user_id, _user_input, _session_id))
        print(f"Agent: {_response}\n")
