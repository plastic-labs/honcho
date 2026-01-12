"""
Honcho Multi-Tool Example

Demonstrates using all Honcho tools with an Agno agent:
- add_message: Store conversation history
- get_context: Retrieve session context
- search_messages: Semantic search
- query_user: Dialectic API queries

Environment Variables:
    OPENAI_API_KEY or LLM_OPENAI_API_KEY: OpenAI API key
    OPENAI_MODEL: Model to use (default: gpt-4o)
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
    print("=" * 70)
    print("HONCHO TOOLS + AGNO EXAMPLE")
    print("=" * 70 + "\n")

    # Get environment settings
    honcho_env = os.getenv("HONCHO_ENVIRONMENT", "production")

    # Setup Honcho tools with a specific session
    honcho_tools = HonchoTools(
        app_id="travel-app",
        user_id="traveler-42",
        session_id="trip-planning-session",
        environment=honcho_env,
    )

    # Pre-populate with travel preferences
    print("Adding travel preferences to memory...")
    messages = [
        "I'm planning a trip to Japan in March",
        "I love trying authentic local cuisine",
        "My budget is around $3000 for 10 days",
        "I prefer ryokans over hotels",
        "I'm interested in both traditional temples and modern Tokyo",
    ]

    for msg in messages:
        result = honcho_tools.add_message(msg, role="user")
        print(f"  Added: {msg[:50]}...")

    print("\n" + "-" * 70 + "\n")

    # Create travel planning agent with memory tools
    agent = Agent(
        name="Travel Planner",
        model=OpenAIChat(id=os.getenv("OPENAI_MODEL", "gpt-4o")),
        tools=[honcho_tools],
        description=(
            "A travel planning expert with access to memory tools. "
            "Use get_context for recent conversation, search_messages to find "
            "specific preferences, and query_user to understand the traveler."
        ),
        instructions=[
            "Always retrieve relevant context before making recommendations",
            "Use search to find specific preferences mentioned",
            "Query the user's knowledge to understand their travel style",
            "Be specific and actionable in your recommendations",
        ],
    )

    # Run the agent with a planning request
    print("Asking agent to create a personalized itinerary...\n")
    response = agent.run(
        "Create a 3-day Tokyo itinerary for me. First, use the memory tools to "
        "understand my preferences (budget, accommodation style, interests), "
        "then create a personalized plan that matches what I've told you."
    )

    print("=" * 70)
    print("RESPONSE")
    print("=" * 70)
    print(response.content)

    # Demonstrate search capability
    print("\n" + "=" * 70)
    print("DIRECT TOOL USAGE: Searching for budget info...")
    print("=" * 70)
    search_result = honcho_tools.search_messages("budget money cost", limit=5)
    print(search_result)


if __name__ == "__main__":
    main()
