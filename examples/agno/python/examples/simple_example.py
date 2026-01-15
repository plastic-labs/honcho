"""
Simple Honcho + Agno Example

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
    session_id = f"simple-{uuid.uuid4().hex[:8]}"

    # Initialize Honcho client
    honcho = Honcho(workspace_id="agno-demo")

    # Initialize HonchoTools - creates peer and session internally
    honcho_tools = HonchoTools(
        app_id="agno-demo",
        peer_id="assistant",
        session_id=session_id,
        honcho_client=honcho,
    )

    # Create user peer (toolkit's peer is "assistant")
    user_peer = honcho.peer("user")

    # Create an agent with memory tools
    agent = Agent(
        name="Programming Mentor",
        model=OpenAIChat(id=os.getenv("OPENAI_MODEL", "gpt-4o")),
        tools=[honcho_tools],
        description="A programming mentor that remembers user interests and progress.",
        instructions=[
            "Use the chat tool to understand the user's preferences and interests",
            "Use get_context if you need raw conversation history",
        ],
    )

    # Add user messages
    print("Adding user messages to conversation...")
    honcho_tools.session.add_messages([
        user_peer.message("I'm learning Python programming"),
        user_peer.message("I'm also interested in web development with FastAPI"),
    ])

    # The agent can now query memories and provide personalized responses
    print("\nAsking the agent for recommendations...")
    response = agent.run(
        "Based on what you know about the user, what should they learn next? "
        "Use the chat tool to understand their interests first."
    )

    # Save the assistant's response to Honcho
    assistant_response = str(response.content) if response.content else ""
    if assistant_response:
        honcho_tools.session.add_messages([honcho_tools.peer.message(assistant_response)])

    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(response.content)

    # Show the full context
    print("\n" + "=" * 60)
    print("SESSION CONTEXT")
    print("=" * 60)
    print(honcho_tools.get_context())


if __name__ == "__main__":
    main()
