"""Smolagents + Honcho persistent memory integration.

Demonstrates a conversational agent that remembers users across sessions.
Honcho stores every message and builds a long-term representation of the user;
the agent injects that context into its task string on every turn and can
query memory on demand via the ``query_memory`` tool.

Usage:
    python main.py

Environment variables:
    HONCHO_API_KEY      Required. Your Honcho API key from honcho.dev.
    HONCHO_WORKSPACE_ID Optional. Workspace ID (default: "default").
    OPENAI_API_KEY      Required. Your OpenAI API key (used via LiteLLM).
"""

from smolagents import LiteLLMModel, ToolCallingAgent

from tools.client import HonchoContext
from tools.get_context import get_context
from tools.query_memory import QueryMemoryTool
from tools.save_memory import save_memory

SYSTEM_PROMPT = (
    "You are a helpful assistant with persistent memory powered by Honcho. "
    "You remember users across conversations. "
    "When a user asks what you remember about them, use the query_memory tool."
)


def chat(user_id: str, message: str, session_id: str) -> str:
    """Run one conversation turn with persistent Honcho memory.

    Fetches Honcho context and prepends it to the message so the agent
    always has an up-to-date view of the session. Saves the user message
    before the run and the assistant reply after.

    Args:
        user_id: Unique identifier for the user.
        message: The user's input message.
        session_id: Identifier for the current conversation session.

    Returns:
        The agent's response as a string.
    """
    ctx = HonchoContext(user_id=user_id, session_id=session_id)

    # Build context prefix from Honcho history
    history = get_context(ctx, tokens=2000)
    if history:
        formatted = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history)
        context_prefix = f"## Conversation History\n{formatted}\n\n"
    else:
        context_prefix = ""

    model = LiteLLMModel(model_id="openai/gpt-4.1-mini")
    agent = ToolCallingAgent(
        tools=[QueryMemoryTool(user_id)],
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )

    save_memory(user_id, message, "user", session_id)

    # Prepend history so the agent has full context without relying on
    # built-in memory (Smolagents manages its own short-term memory separately)
    full_message = f"{context_prefix}User: {message}" if context_prefix else message
    response = str(agent.run(full_message))

    save_memory(user_id, response, "assistant", session_id)
    return response


if __name__ == "__main__":
    print("Smolagents HonchoMemoryAgent — type 'quit' to exit\n")
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
