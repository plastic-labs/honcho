"""Agno + Honcho persistent memory integration.

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

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

from tools.client import HonchoContext, get_client
from tools.get_context import get_context
from tools.save_memory import save_memory


def make_query_memory_tool(user_id: str):
    """Create a query_memory tool bound to a specific user.

    Agno tools are plain functions decorated with ``@tool``. Because Agno
    does not pass a run-context to tools, we use a factory to close over
    the ``user_id`` at call time.

    Args:
        user_id: The user peer ID whose memory to query.

    Returns:
        An Agno-compatible tool function.
    """

    @tool
    def query_memory(query: str) -> str:
        """Query Honcho's Dialectic API to recall facts about the current user.

        Use this when the user asks what you remember about them or their
        past conversations.

        Args:
            query: Natural language question about the user.

        Returns:
            A natural language answer from Honcho's memory.
        """
        honcho = get_client()
        peer = honcho.peer(user_id)
        response = peer.chat(query=query)
        return str(response) if response else "No relevant information found in memory."

    return query_memory


def chat(user_id: str, message: str, session_id: str) -> str:
    """Run one conversation turn with persistent Honcho memory.

    Builds a fresh agent with dynamic instructions derived from Honcho
    context, saves the user message before the run, and persists the
    assistant reply after.

    Args:
        user_id: Unique identifier for the user.
        message: The user's input message.
        session_id: Identifier for the current conversation session.

    Returns:
        The agent's response as a string.
    """
    ctx = HonchoContext(user_id=user_id, session_id=session_id)

    base = (
        "You are a helpful assistant with persistent memory powered by Honcho. "
        "You remember users across conversations. "
        "When a user asks what you remember about them, use the query_memory tool."
    )
    history = get_context(ctx, tokens=2000)
    if history:
        formatted = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history)
        description = f"{base}\n\n## Conversation History\n{formatted}"
    else:
        description = base

    agent = Agent(
        model=OpenAIChat(id="gpt-4.1-mini"),
        description=description,
        tools=[make_query_memory_tool(user_id)],
        markdown=False,
    )

    save_memory(user_id, message, "user", session_id)
    run_response = agent.run(message)
    response = (
        run_response.content
        if hasattr(run_response, "content")
        else str(run_response)
    )
    save_memory(user_id, response, "assistant", session_id)

    return response


if __name__ == "__main__":
    print("Agno HonchoMemoryAgent — type 'quit' to exit\n")
    _user_id = "demo-user"
    _session_id = "demo-session"

    while True:
        _user_input = input("You: ").strip()
        if not _user_input:
            continue
        if _user_input.lower() in ("quit", "exit"):
            break
        _response = chat(_user_id, _user_input, _session_id)
        print(f"Agent: {_response}\n")
