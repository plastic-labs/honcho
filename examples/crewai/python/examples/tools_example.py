"""
Honcho Tools with CrewAI Example

Demonstrates how to equip CrewAI agents with Honcho's memory tools:
- HonchoGetContextTool: Retrieve session context with token limits
- HonchoDialecticTool: Query representations about peers
- HonchoSearchTool: Perform semantic search across session messages

These tools give agents explicit control over memory retrieval, beyond the
automatic memory provided by ExternalMemory.
"""

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from honcho import Honcho
from honcho_crewai import (
    HonchoGetContextTool,
    HonchoDialecticTool,
    HonchoSearchTool,
)

load_dotenv()


def main():
    """Demonstrate Honcho tools with CrewAI agents."""
    print("=" * 70)
    print("HONCHO TOOLS + CREWAI EXAMPLE")
    print("=" * 70 + "\n")

    # Step 1: Setup session with conversation history
    print("1. Setting up session with conversation history...\n")

    honcho = Honcho()
    user_id = "demo-user-45"
    session_id = "tools-demo-session"

    user = honcho.peer(user_id)
    session = honcho.session(session_id)

    # Add conversation history
    messages = [
        "I'm planning a trip to Japan in March",
        "I love trying authentic local cuisine, especially ramen and sushi",
        "My budget is around $3000 for a 10-day trip",
        "I'm interested in visiting both Tokyo and Kyoto",
        "I prefer staying in traditional ryokans over hotels",
    ]

    for msg in messages:
        print(f"   • {msg}")
        session.add_messages([user.message(msg)])

    print("\n   ✓ Session created with 5 messages\n")

    # Step 2: Create Honcho tools
    print("2. Creating Honcho memory tools...\n")

    context_tool = HonchoGetContextTool(
        honcho=honcho, session_id=session_id, peer_id=user_id
    )
    print("   ✓ get_session_context - Retrieve conversation context")

    dialectic_tool = HonchoDialecticTool(
        honcho=honcho, session_id=session_id, peer_id=user_id
    )
    print("   ✓ query_peer_knowledge - Ask about user preferences")

    search_tool = HonchoSearchTool(honcho=honcho, session_id=session_id)
    print("   ✓ search_session_messages - Semantic search messages\n")

    # Step 3: Create agent with tools
    print("3. Creating travel planning agent with memory tools...\n")

    travel_agent = Agent(
        role="Travel Planning Specialist",
        goal="Create personalized travel recommendations using memory tools",
        backstory=(
            "You are an expert travel planner with access to conversation memory tools. "
            "Use the tools to understand the user's preferences before making recommendations."
        ),
        tools=[context_tool, dialectic_tool, search_tool],
        verbose=True,
        allow_delegation=False
    )
    print("   ✓ Agent created with 3 Honcho tools\n")

    # Step 4: Create task
    print("4. Creating task...\n")

    task = Task(
        description=(
            "Create a personalized 3-day Tokyo itinerary. "
            "Use the memory tools to understand:\n"
            "  • Food preferences (search for 'cuisine' or 'food')\n"
            "  • Travel style and budget (query user knowledge)\n"
            "  • Recent context (get conversation context)\n"
            "Then create a detailed plan matching their interests."
        ),
        expected_output=(
            "A 3-day Tokyo itinerary with:\n"
            "  • Daily activities matching user interests\n"
            "  • Restaurant recommendations\n"
            "  • Accommodation suggestions\n"
            "  • Budget considerations"
        ),
        agent=travel_agent
    )
    print("   ✓ Task created\n")

    # Step 5: Execute
    print("5. Executing crew (agent will use tools to retrieve memory)...\n")
    print("-" * 70 + "\n")

    crew = Crew(
        agents=[travel_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    crew.kickoff()

if __name__ == "__main__":
    main()
