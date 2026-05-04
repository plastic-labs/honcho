"""
Simple Honcho + CrewAI Example

A minimal example showing how to use Honcho-backed unified Memory with CrewAI agents.
This demonstrates the basic pattern for persistent conversation memory.
"""

from crewai import Agent, Crew, Memory, Process, Task
from dotenv import load_dotenv
from honcho_crewai import HonchoMemoryStorage

load_dotenv()


def main():
    """Simple example of CrewAI agent with Honcho memory."""
    user_id = "simple-demo-user"

    # Initialize CrewAI unified memory backed by Honcho
    storage = HonchoMemoryStorage(
        peer_id=user_id,
        session_id="simple-demo-session",
    )
    memory = Memory(storage=storage)

    # Add some conversation history
    messages = [
        ("user", "I'm learning Python programming"),
        ("assistant", "Great! Python is an excellent language to learn."),
        ("user", "I'm particularly interested in web development"),
    ]

    for role, message in messages:
        memory.remember(
            message,
            scope=f"/users/{user_id}/conversation",
            categories=["conversation"],
            metadata={"role": role},
        )

    # Create agent with memory
    agent = Agent(
        role="Programming Mentor",
        goal="Help users learn programming by remembering their interests and progress",
        backstory=(
            "You are a patient programming mentor who remembers what students "
            "have told you about their learning journey and interests."
        ),
        verbose=True,
        allow_delegation=False,
    )

    # Create task
    task = Task(
        description=(
            "Based on what you know about the user's interests, "
            "suggest a simple web development project they could build to practice Python."
        ),
        expected_output="A specific project suggestion with brief explanation",
        agent=agent,
    )

    # Execute with memory - CrewAI automatically retrieves relevant context.
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        memory=memory,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(result.raw)


if __name__ == "__main__":
    main()
