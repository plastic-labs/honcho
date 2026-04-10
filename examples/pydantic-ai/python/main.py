"""Pydantic AI + Honcho persistent memory integration.

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

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

from tools.client import HonchoContext, get_client
from tools.get_context import get_context
from tools.save_memory import save_memory


@dataclass
class HonchoAgentDeps:
    """Dependencies injected into every Pydantic AI agent call.

    Attributes:
        ctx: Honcho identity for the current conversation turn.
    """

    ctx: HonchoContext


honcho_agent: Agent[HonchoAgentDeps, str] = Agent(
    "openai:gpt-4.1-mini",
    deps_type=HonchoAgentDeps,
    result_type=str,
    system_prompt=(
        "You are a helpful assistant with persistent memory powered by Honcho. "
        "You remember users across conversations. "
        "When a user asks what you remember about them, use the query_memory tool."
    ),
)


@honcho_agent.system_prompt
def honcho_system_prompt(run_ctx: RunContext[HonchoAgentDeps]) -> str:
    """Append Honcho conversation history to the system prompt.

    Called by Pydantic AI before every LLM request. Returns an additional
    system-prompt segment containing the recent session history fetched from
    Honcho. Returns an empty string when the session has no history yet.

    Args:
        run_ctx: The run context exposing ``HonchoAgentDeps``.

    Returns:
        A formatted history string, or ``""`` if no history exists.
    """
    history = get_context(run_ctx.deps.ctx, tokens=2000)
    if not history:
        return ""
    formatted = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history)
    return f"\n\n## Conversation History\n{formatted}"


@honcho_agent.tool
def query_memory(run_ctx: RunContext[HonchoAgentDeps], query: str) -> str:
    """Query Honcho's Dialectic API to recall facts about the current user.

    Use this when the user asks what you remember about them or their past
    conversations.

    Args:
        run_ctx: The run context exposing ``HonchoAgentDeps``.
        query: Natural language question about the user.

    Returns:
        A natural language answer from Honcho's memory.
    """
    ctx = run_ctx.deps.ctx
    honcho = get_client()
    peer = honcho.peer(ctx.user_id)
    response = peer.chat(query=query)
    return str(response) if response else "No relevant information found in memory."


async def chat(
    user_id: str,
    message: str,
    session_id: str,
    message_history: list[ModelMessage] | None = None,
) -> tuple[str, list[ModelMessage]]:
    """Run one conversation turn with persistent Honcho memory.

    Pydantic AI's ``message_history`` parameter lets the agent maintain
    in-session coherence across turns — it is separate from Honcho's
    long-term cross-session memory.

    Args:
        user_id: Unique identifier for the user.
        message: The user's input message.
        session_id: Identifier for the current conversation session.
        message_history: Prior messages for in-session coherence.

    Returns:
        Tuple of ``(response_text, updated_message_history)``.
    """
    ctx = HonchoContext(user_id=user_id, session_id=session_id)
    deps = HonchoAgentDeps(ctx=ctx)

    try:
        save_memory(user_id, message, "user", session_id)
    except Exception as exc:
        print(f"Warning: failed to save user message — {exc}")

    result = await honcho_agent.run(
        message,
        deps=deps,
        message_history=message_history or [],
    )
    response = str(result.output)

    try:
        save_memory(user_id, response, "assistant", session_id)
    except Exception as exc:
        print(f"Warning: failed to save assistant response — {exc}")

    return response, result.all_messages()


async def main() -> None:
    print("Pydantic AI HonchoMemoryAgent — type 'quit' to exit\n")
    user_id = "demo-user"
    session_id = "demo-session"
    message_history: list[ModelMessage] = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        response, message_history = await chat(
            user_id, user_input, session_id, message_history
        )
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
