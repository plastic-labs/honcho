"""
Simple Honcho + Agno Example

Demonstrates the RunContext integration:
- user_id and session_id are passed to agent.run()
- Tools automatically receive RunContext with these values
- Orchestration uses the honcho client directly

Environment Variables:
    LLM_OPENAI_API_KEY: OpenAI API key (matches honcho .env)
    HONCHO_API_KEY: Required for Honcho API access
"""

import os
import uuid

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from honcho import Honcho
from honcho_agno import HonchoTools

load_dotenv()

# Use LLM_OPENAI_API_KEY from honcho .env
if llm_key := os.getenv("LLM_OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = llm_key


def main():
    # Unique IDs for this run
    user_id = "user-python-learner"
    session_id = f"simple-{uuid.uuid4().hex[:8]}"

    # Initialize Honcho client
    honcho = Honcho(workspace_id="agno-demo")

    # Initialize HonchoTools with agent identity
    # user_id and session_id come from RunContext at runtime
    honcho_tools = HonchoTools(
        agent_id="assistant",
        honcho_client=honcho,
    )

    # Create peers and session for orchestration
    user_peer = honcho.peer(user_id)
    assistant_peer = honcho.peer("assistant")
    session = honcho.session(session_id)

    # Create an agent with memory tools
    agent = Agent(
        name="Programming Mentor",
        model=OpenAIChat(id=os.getenv("OPENAI_MODEL", "gpt-4o")),
        tools=[honcho_tools],
        description="A programming mentor that remembers user interests and progress.",
        instructions=[
            "Use the honcho_chat tool to understand the user's preferences and interests",
            "Use honcho_get_context if you need raw conversation history",
        ],
    )

    # Add user messages via orchestration (not toolkit)
    print("Adding user messages to conversation...")
    session.add_messages([
        user_peer.message("I'm learning Python programming"),
        user_peer.message("I'm also interested in web development with FastAPI"),
    ])

    # The agent can now query memories and provide personalized responses
    # user_id and session_id flow through RunContext to the tools
    print("\nAsking the agent for recommendations...")
    response = agent.run(
        "Based on what you know about the user, what should they learn next? "
        "Use the honcho_chat tool to understand their interests first.",
        user_id=user_id,
        session_id=session_id,
    )

    # Save the assistant's response via orchestration
    assistant_response = str(response.content) if response.content else ""
    if assistant_response:
        session.add_messages([assistant_peer.message(assistant_response)])

    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(response.content)

    # Show the full context using the honcho client directly
    print("\n" + "=" * 60)
    print("SESSION CONTEXT")
    print("=" * 60)
    context = session.get_context()
    print(context)


if __name__ == "__main__":
    main()
