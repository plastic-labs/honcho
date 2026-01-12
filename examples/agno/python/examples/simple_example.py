"""
Simple Honcho + Agno Example

A minimal example showing how to use HonchoTools with Agno agents.

Environment Variables:
    OPENAI_API_KEY or LLM_OPENAI_API_KEY: OpenAI API key
    HONCHO_ENVIRONMENT: 'local' or 'production' (default: production)
    HONCHO_API_KEY: Required for production environment
"""

import os

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from honcho_agno import HonchoTools

load_dotenv()

# Support both OPENAI_API_KEY and LLM_OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("LLM_OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("LLM_OPENAI_API_KEY")


def main():
    # Get environment settings
    honcho_env = os.getenv("HONCHO_ENVIRONMENT", "production")

    # Initialize Honcho tools with user context
    honcho_tools = HonchoTools(
        app_id="agno-demo",
        user_id="demo-user",
        environment=honcho_env,
    )

    # Add some conversation history manually
    print("Adding conversation history...")
    honcho_tools.add_message("I'm learning Python programming", role="user")
    honcho_tools.add_message(
        "I'm also interested in web development with FastAPI", role="user"
    )

    # Create an agent with memory tools
    agent = Agent(
        name="Programming Mentor",
        model=OpenAIChat(id=os.getenv("OPENAI_MODEL", "gpt-4o")),
        tools=[honcho_tools],
        description="A programming mentor that remembers user interests and progress.",
        instructions=[
            "Use the memory tools to understand the user's background",
            "Provide personalized recommendations based on their interests",
        ],
    )

    # The agent can now query memories and provide personalized responses
    print("\nAsking the agent for recommendations...")
    response = agent.run(
        "Based on what you know about me, what should I learn next? "
        "Use your memory tools to check my interests first."
    )

    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(response.content)


if __name__ == "__main__":
    main()
