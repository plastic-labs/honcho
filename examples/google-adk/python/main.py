"""Google ADK + Honcho persistent memory integration.

Demonstrates a conversational agent that remembers users across sessions.
Honcho stores every message and builds a long-term representation of the user;
the agent injects that context into its instructions on every turn and can
query memory on demand via the ``query_memory`` tool.

Usage:
    python main.py

Environment variables:
    HONCHO_API_KEY      Required. Your Honcho API key from honcho.dev.
    HONCHO_WORKSPACE_ID Optional. Workspace ID (default: "default").
    GOOGLE_API_KEY      Required. Your Google AI API key.
"""

import asyncio
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types

from tools.client import HonchoContext, get_client
from tools.get_context import get_context
from tools.query_memory import query_memory
from tools.save_memory import save_memory

APP_NAME = "honcho-memory-agent"
MODEL_ID = "gemini-2.5-flash"


def setup_session(user_id: str, session_id: str, assistant_id: str = "assistant") -> None:
    """Register peers in the session once at startup.

    Args:
        user_id: Unique identifier for the user peer.
        session_id: Identifier for the conversation session.
        assistant_id: Peer ID for the assistant. Defaults to ``"assistant"``.
    """
    honcho = get_client()
    user_peer = honcho.peer(user_id)
    assistant_peer = honcho.peer(assistant_id)
    session = honcho.session(session_id)
    session.add_peers([user_peer, assistant_peer])


def build_instruction(ctx: HonchoContext) -> str:
    """Build dynamic system instructions that include Honcho memory context.

    Args:
        ctx: HonchoContext holding user, session, and assistant IDs.

    Returns:
        System prompt string with injected conversation history.
    """
    base = (
        "You are a helpful assistant with persistent memory powered by Honcho. "
        "You remember users across conversations. "
        "When a user asks what you remember about them, use the query_memory tool."
    )

    history = get_context(ctx, tokens=2000)
    if not history:
        return base

    # Use content directly from structured messages — preserve role context
    formatted = "\n".join(
        f"{msg['role'].title()}: {msg['content']}" for msg in history
    )
    return f"{base}\n\n## Conversation History\n{formatted}"


def make_query_memory_fn(user_id: str):
    """Create a query_memory function bound to a specific user_id."""

    def _query_memory(query: str) -> str:
        """Query Honcho's Dialectic API to recall facts about the current user.

        Use this when the user asks what you remember about them or their
        past conversations.

        Args:
            query: Natural language question about the user.

        Returns:
            A natural language answer from Honcho's memory.
        """
        return query_memory(user_id=user_id, query=query)

    return _query_memory


async def chat(user_id: str, message: str, session_id: str) -> str:
    """Run one conversation turn with persistent Honcho memory.

    Saves the user message to Honcho before the agent runs, then saves the
    assistant reply afterwards. Dynamic instructions inject Honcho context
    for every turn automatically.

    Args:
        user_id: Unique identifier for the user.
        message: The user's input message.
        session_id: Identifier for the current conversation session.

    Returns:
        The agent's response as a string.
    """
    ctx = HonchoContext(user_id=user_id, session_id=session_id)

    # Build dynamic instructions with current Honcho context
    instruction = build_instruction(ctx)

    # Create a fresh agent with updated instructions each turn
    agent = LlmAgent(
        name="HonchoMemoryAgent",
        model=MODEL_ID,
        instruction=instruction,
        tools=[FunctionTool(make_query_memory_fn(user_id))],
    )

    session_service = InMemorySessionService()
    adk_session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Persist user message before the agent runs
    save_memory(user_id, message, "user", session_id)

    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=adk_session.id,
        new_message=user_content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text or ""

    # Persist assistant response after the run — only when the agent produced output
    if response_text:
        save_memory(user_id, response_text, "assistant", session_id)

    return response_text


async def main() -> None:
    print("Google ADK HonchoMemoryAgent — type 'quit' to exit\n")
    user_id = "demo-user"
    session_id = str(uuid.uuid4())

    # Register peers once at session start — not on every turn
    setup_session(user_id, session_id)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        response = await chat(user_id, user_input, session_id)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
