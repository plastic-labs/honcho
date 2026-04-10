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

from agents import Agent, RunContextWrapper, Runner

from tools.client import HonchoContext, get_client
from tools.get_context import get_context
from tools.query_memory import query_memory
from tools.save_memory import save_memory


def honcho_instructions(ctx: RunContextWrapper[HonchoContext], agent: Agent) -> str:
    """Build dynamic system instructions that include Honcho memory context.

    Called by the OpenAI Agents SDK before every LLM request. Fetches recent
    conversation history from Honcho and prepends it so the model always has
    an up-to-date view of the session.

    Args:
        ctx: Run context wrapping the ``HonchoContext``.
        agent: The current agent instance (unused, required by the SDK).

    Returns:
        System prompt string with injected conversation history.
    """
    base = (
        "You are a helpful assistant with persistent memory powered by Honcho. "
        "You remember users across conversations. "
        "When a user asks what you remember about them, use the query_memory tool."
    )

    history = get_context(ctx.context, tokens=2000)
    if not history:
        return base

    formatted = "\n".join(
        f"{msg['role'].title()}: {msg['content']}" for msg in history
    )
    return f"{base}\n\n## Conversation History\n{formatted}"


honcho_agent = Agent[HonchoContext](
    name="HonchoMemoryAgent",
    instructions=honcho_instructions,
    tools=[query_memory],
    model="gpt-4.1-mini",
)


async def chat(user_id: str, message: str, session_id: str) -> str:
    """Run one conversation turn with persistent Honcho memory.

    Saves the user message to Honcho before the agent runs, then saves the
    assistant reply afterwards. The dynamic instructions callable injects
    the full Honcho context for every turn automatically.

    Args:
        user_id: Unique identifier for the user.
        message: The user's input message.
        session_id: Identifier for the current conversation session.

    Returns:
        The agent's response as a string.
    """
    ctx = HonchoContext(user_id=user_id, session_id=session_id)

    # Persist user message before the agent runs so it's available in context
    save_memory(user_id, message, "user", session_id)

    result = await Runner.run(honcho_agent, message, context=ctx)
    response = str(result.final_output)

    # Persist assistant response after the run
    save_memory(user_id, response, "assistant", session_id)

    return response


if __name__ == "__main__":
    print("HonchoMemoryAgent — type 'quit' to exit\n")
    _user_id = "demo-user"
    _session_id = "demo-session"

    while True:
        _user_input = input("You: ").strip()
        if not _user_input:
            continue
        if _user_input.lower() in ("quit", "exit"):
            break
        _response = asyncio.run(chat(_user_id, _user_input, _session_id))
        print(f"Agent: {_response}\n")
