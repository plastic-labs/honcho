"""
CrewAI Integration with Honcho and OpenAI

This example demonstrates how to build AI agents with persistent memory using
CrewAI for agent orchestration, OpenAI for the AI model, and Honcho for memory
management via the honcho_crewai package.
"""

from typing import Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.memory.external.external_memory import ExternalMemory
from honcho_crewai import HonchoStorage

load_dotenv()


def run_conversation_turn(
    user_id: str,
    user_input: str,
    session_id: Optional[str] = None,
    storage: Optional[HonchoStorage] = None
) -> tuple[str, HonchoStorage]:
    """
    Run a single conversation turn with the CrewAI agent.

    Args:
        user_id: Unique identifier for the user
        user_input: User's message
        session_id: Optional session ID for conversation continuity
        storage: Optional existing HonchoStorage instance

    Returns:
        Tuple of (agent_response, storage_instance)
    """
    # Initialize or reuse storage
    if storage is None:
        if not session_id:
            session_id = f"session_{user_id}"
        storage = HonchoStorage(user_id=user_id, session_id=session_id)

    # Create ExternalMemory wrapper for automatic context retrieval
    external_memory = ExternalMemory(storage=storage)

    # Save user input to memory
    external_memory.save(user_input, metadata={"agent": "user"})

    # Create an agent with memory
    agent = Agent(
        role="AI Assistant",
        goal="Help users with their questions and remember context from previous conversations",
        backstory=(
            "You are a helpful AI assistant with the ability to remember past conversations. "
            "You use context from previous interactions to provide personalized and relevant responses."
        ),
        verbose=False,
        allow_delegation=False
    )

    # Create task for the agent
    task = Task(
        description=f"Respond to the user's message: {user_input}",
        expected_output="A helpful and contextually relevant response that considers conversation history",
        agent=agent
    )

    # Create crew with external memory - enables automatic context retrieval
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        external_memory=external_memory,
        verbose=False
    )

    # Execute - CrewAI automatically retrieves relevant context from Honcho
    result = crew.kickoff()

    # Save assistant response back to memory
    response_text = str(result.raw)
    external_memory.save(response_text, metadata={"agent": "assistant"})

    return response_text, storage


def main():
    """Interactive chat loop with CrewAI agent powered by Honcho memory."""
    print("Welcome to the AI Assistant powered by CrewAI and Honcho!")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    user_id = "demo-user-123"
    storage = None

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            response, storage = run_conversation_turn(
                user_id=user_id,
                user_input=user_input,
                storage=storage
            )
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
