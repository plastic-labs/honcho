"""
Simple Honcho + CrewAI Example

A minimal example showing how to use Honcho's ExternalMemory with CrewAI agents.
This demonstrates the basic pattern for persistent conversation memory.
"""

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.memory.external.external_memory import ExternalMemory
from honcho_crewai import HonchoStorage

load_dotenv()


def main():
    """Simple example of CrewAI agent with Honcho memory."""
    # Initialize Honcho storage
    storage = HonchoStorage(user_id="simple-demo-user")
    external_memory = ExternalMemory(storage=storage)

    # Add some conversation history
    messages = [
        ("user", "I'm learning Python programming"),
        ("assistant", "Great! Python is an excellent language to learn."),
        ("user", "I'm particularly interested in web development"),
    ]

    for role, message in messages:
        external_memory.save(message, metadata={"agent": role})

    # Create agent with memory
    agent = Agent(
        role="Programming Mentor",
        goal="Help users learn programming by remembering their interests and progress",
        backstory=(
            "You are a patient programming mentor who remembers what students "
            "have told you about their learning journey and interests."
        ),
        verbose=True,
        allow_delegation=False
    )

    # Create task
    task = Task(
        description=(
            "Based on what you know about the user's interests, "
            "suggest a simple web development project they could build to practice Python."
        ),
        expected_output="A specific project suggestion with brief explanation",
        agent=agent
    )

    # Execute with memory - CrewAI automatically retrieves relevant context!
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        external_memory=external_memory,
        verbose=True
    )

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(result.raw)


if __name__ == "__main__":
    main()
