"""LlamaIndex + Honcho persistent memory integration.

Demonstrates a conversational agent that remembers users across sessions.
Honcho stores every message and builds a long-term representation of the user;
the agent injects that context into its system prompt on every turn and can
query memory on demand via the ``query_memory`` tool.

Usage:
    python main.py

Environment variables:
    HONCHO_API_KEY      Required. Your Honcho API key from honcho.dev.
    HONCHO_WORKSPACE_ID Optional. Workspace ID (default: "default").
    OPENAI_API_KEY      Required. Your OpenAI API key.
"""

from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import OpenAI

from tools.client import HonchoContext
from tools.get_context import get_context
from tools.query_memory import make_query_memory_tool
from tools.save_memory import save_memory


def chat(user_id: str, message: str, session_id: str) -> str:
    """Run one conversation turn with persistent Honcho memory.

    Builds a ``ReActAgent`` with dynamic system-prompt messages derived from
    Honcho context, saves the user message before the run, and persists the
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
        system_content = f"{base}\n\n## Conversation History\n{formatted}"
    else:
        system_content = base

    llm = OpenAI(model="gpt-4.1-mini")
    agent = ReActAgent.from_tools(
        tools=[make_query_memory_tool(ctx)],
        llm=llm,
        verbose=False,
        prefix_messages=[
            ChatMessage(role=MessageRole.SYSTEM, content=system_content)
        ],
    )

    save_memory(user_id, message, "user", session_id)
    response = str(agent.chat(message))
    save_memory(user_id, response, "assistant", session_id)

    return response


if __name__ == "__main__":
    print("LlamaIndex HonchoMemoryAgent — type 'quit' to exit\n")
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
