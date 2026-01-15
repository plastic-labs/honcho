"""
Simple Honcho + Agno Example

Environment Variables:
    OPENAI_API_KEY or LLM_OPENAI_API_KEY: OpenAI API key
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

# Support both OPENAI_API_KEY and LLM_OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and (llm_key := os.getenv("LLM_OPENAI_API_KEY")):
    os.environ["OPENAI_API_KEY"] = llm_key


def main():
    # Create shared session
    session_id = f"simple-{uuid.uuid4().hex[:8]}"

    # Initialize Honcho directly for managing user messages
    honcho = Honcho(workspace_id="agno-demo")
    session = honcho.session(session_id)
    user_peer = honcho.peer("user")

    # Initialize HonchoTools - this IS the assistant's identity
    honcho_tools = HonchoTools(
        app_id="agno-demo",
        peer_id="assistant",  # The toolkit speaks as "assistant"
        session_id=session_id,  # Same session as user
        honcho_client=honcho,  # Reuse client
    )

    # Create an agent with memory tools
    agent = Agent(
        name="Programming Mentor",
        model=OpenAIChat(id=os.getenv("OPENAI_MODEL", "gpt-4o")),
        tools=[honcho_tools],
        description="A programming mentor that remembers user interests and progress.",
        instructions=[
            "Use get_context to understand the conversation history",
            "Use query_peer to ask about the user's preferences",
            "Use add_message to save your responses to the conversation",
        ],
    )

    # Add user messages via Honcho directly
    print("Adding user messages to conversation...")
    session.add_messages([
        user_peer.message("I'm learning Python programming"),
        user_peer.message("I'm also interested in web development with FastAPI"),
    ])

    # The agent can now query memories and provide personalized responses
    print("\nAsking the agent for recommendations...")
    response = agent.run(
        "Based on what you know about the user, what should they learn next? "
        "Use get_context to see the conversation history first."
    )

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
