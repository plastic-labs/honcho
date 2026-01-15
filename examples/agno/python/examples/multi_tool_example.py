"""
Honcho Multi-Tool Example

Demonstrates using Honcho tools with an Agno agent:
- chat: Ask questions about the conversation (recommended)
- get_context: Retrieve raw session context
- search_messages: Semantic search through messages

The chat tool is the recommended way to understand users
It reasons over conversation context and provides synthesized insights.

Environment Variables:
    LLM_OPENAI_API_KEY: OpenAI API key (matches honcho .env)
    OPENAI_MODEL: Model to use
    HONCHO_API_KEY: Required for Honcho API access
"""

import os

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
    print("=" * 70)
    print("HONCHO TOOLS + AGNO EXAMPLE")
    print("=" * 70 + "\n")

    # Initialize Honcho client
    honcho = Honcho(workspace_id="travel-app")

    # Setup Honcho tools - creates peer and session internally
    honcho_tools = HonchoTools(
        app_id="travel-app",
        peer_id="travel-assistant",
        session_id="trip-planning-session",
        honcho_client=honcho,
    )

    # Create user peer (the toolkit's peer is "travel-assistant")
    user_peer = honcho.peer("traveler-42")

    # Pre-populate with user's travel preferences
    print("Adding user's travel preferences to memory...")
    messages = [
        "I'm planning a trip to Japan in March",
        "I love trying authentic local cuisine",
        "My budget is around $3000 for 10 days",
        "I prefer ryokans over hotels",
        "I'm interested in both traditional temples and modern Tokyo",
    ]

    for msg in messages:
        honcho_tools.session.add_messages([user_peer.message(msg)])
        print(f"  [traveler-42]: {msg[:50]}...")

    print("\n" + "-" * 70 + "\n")

    # Create travel planning agent with memory tools
    agent = Agent(
        name="Travel Planner",
        model=OpenAIChat(id=os.getenv("OPENAI_MODEL", "gpt-4o")),
        tools=[honcho_tools],
        description=(
            "A travel planning expert with access to Honcho memory tools. "
            "Use chat to understand the traveler's preferences and travel style."
        ),
        instructions=[
            "Use the chat tool to understand the user's preferences and travel style",
            "Ask both broad and specific questions like 'What is their travel style?' or 'What is their budget?'",
            "Only use get_context or search_messages if you need raw message history",
            "Be specific and actionable in your recommendations",
        ],
    )

    # Run the agent with a planning request
    print("Asking agent to create a personalized itinerary...\n")
    response = agent.run(
        "Create a 3-day Tokyo itinerary for me. Use the chat tool to ask about "
        "my budget, accommodation preferences, and interests, then create "
        "a personalized plan that matches my travel style."
    )

    # Save the assistant's response to Honcho (using toolkit's peer and session)
    assistant_response = str(response.content) if response.content else ""
    if assistant_response:
        honcho_tools.session.add_messages([honcho_tools.peer.message(assistant_response)])

    print("=" * 70)
    print("RESPONSE")
    print("=" * 70)
    print(response.content)

    # Demonstrate chat (recommended)
    print("\n" + "=" * 70)
    print("DIRECT TOOL USAGE: chat (recommended)")
    print("=" * 70)
    chat_result = honcho_tools.chat(
        "What are the traveler's key preferences and constraints?"
    )
    print(chat_result)

    # Demonstrate search capability
    print("\n" + "=" * 70)
    print("DIRECT TOOL USAGE: search_messages")
    print("=" * 70)
    search_result = honcho_tools.search_messages("budget money cost", limit=5)
    print(search_result)

    # Show full conversation context
    print("\n" + "=" * 70)
    print("DIRECT TOOL USAGE: get_context")
    print("=" * 70)
    print(honcho_tools.get_context())


if __name__ == "__main__":
    main()
