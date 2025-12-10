"""
Hybrid Memory Example: Combining Automatic Memory + Explicit Tools

Demonstrates combining automatic memory (HonchoStorage) with explicit memory tools.
The agent gets baseline context automatically but can also make targeted queries.
"""

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.memory.external.external_memory import ExternalMemory
from honcho import Honcho
from honcho_crewai import (
    HonchoStorage,
    HonchoSearchTool,
    HonchoDialecticTool,
)

load_dotenv()


def main():
    """Hybrid memory example with automatic baseline + explicit tools."""
    # Initialize Honcho
    honcho = Honcho()
    user_id = "hybrid-demo-user"
    session_id = "hybrid-demo-session"

    # Setup automatic memory
    storage = HonchoStorage(
        user_id=user_id,
        session_id=session_id,
        honcho_client=honcho
    )
    external_memory = ExternalMemory(storage=storage)

    # Add conversation history
    messages = [
        ("user", "I'm planning a trip to Japan next spring"),
        ("assistant", "How exciting! Japan is beautiful in spring."),
        ("user", "I love Japanese cuisine, especially ramen and sushi"),
        ("assistant", "You'll find amazing food there!"),
        ("user", "My budget is around $3000 for the whole trip"),
        ("assistant", "That's a good budget for a memorable trip."),
        ("user", "I prefer cultural experiences over touristy attractions"),
    ]

    for role, message in messages:
        external_memory.save(message, metadata={"agent": role})

    # Create memory tools for targeted queries
    search_tool = HonchoSearchTool(honcho=honcho, session_id=session_id)
    dialectic_tool = HonchoDialecticTool(
        honcho=honcho, session_id=session_id, peer_id=user_id
    )

    # Create agent with both automatic memory AND tools
    travel_agent = Agent(
        role="Travel Advisor",
        goal="Create personalized travel recommendations using memory",
        backstory=(
            "You are a travel advisor with access to conversation history. "
            "You can use tools to search for specific details or understand preferences."
        ),
        tools=[search_tool, dialectic_tool],
        verbose=True,
        allow_delegation=False
    )

    # Create task
    task = Task(
        description=(
            "Create a 3-day Tokyo itinerary for the user.\n\n"
            "Use search_tool to find their budget and food preferences.\n"
            "Use query_peer_knowledge to understand their travel style.\n"
            "Then create a personalized itinerary with activities and restaurant recommendations."
        ),
        expected_output="A 3-day Tokyo itinerary with daily activities and dining suggestions",
        agent=travel_agent
    )

    # Execute with hybrid memory: automatic baseline + explicit tools
    crew = Crew(
        agents=[travel_agent],
        tasks=[task],
        process=Process.sequential,
        external_memory=external_memory,  # Automatic memory!
        verbose=True
    )

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(result.raw)


if __name__ == "__main__":
    main()
