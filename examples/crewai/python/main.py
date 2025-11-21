"""
CrewAI Integration with Honcho and OpenAI

This module demonstrates how to build AI agents with persistent memory using
CrewAI for agent orchestration, OpenAI for the AI model, and Honcho for memory
management. It creates agents that remember conversations across sessions.
"""

import os
import uuid
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.memory.external.external_memory import ExternalMemory
from crewai.memory.storage.interface import Storage
from honcho import Honcho

load_dotenv()


class HonchoStorage(Storage):
    """
    Honcho-backed storage provider for CrewAI external memory.
    Implements CrewAI's Storage interface using Honcho's session-based memory.
    """

    def __init__(
        self,
        user_id: str,
        session_id: Optional[str] = None,
    ):
        """
        Initialize Honcho storage for a specific user and session.

        Args:
            user_id: Unique identifier for the user
            session_id: Optional session ID. If not provided, one will be generated
        """
        self.honcho = Honcho()

        # Initialize user and assistant peers
        self.user = self.honcho.peer(user_id)
        self.assistant = self.honcho.peer("assistant")

        # Create or use existing session
        if not session_id:
            session_id = f"session_{uuid.uuid4()}"
        self.session = self.honcho.session(session_id)
        self.session_id = session_id

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """
        Save a message to Honcho session.

        Args:
            value: Message content to save
            metadata: Metadata dict that may contain 'role', 'agent', or 'type' info
        """
        # Determine if this is from user or assistant based on metadata
        role = metadata.get("role", metadata.get("agent", "assistant"))
        is_user = role == "user"
        peer = self.user if is_user else self.assistant

        # Add message to session
        self.session.add_messages([peer.message(str(value))])

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant messages in Honcho session.

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum relevance score

        Returns:
            List of message dictionaries in CrewAI expected format
        """
        # Token limit approximation: ~100 tokens per message
        token_limit = limit * 100

        # Get context from Honcho
        context = self.session.get_context(tokens=token_limit)
        messages = context.messages

        # Convert to CrewAI expected format
        # CrewAI's external memory expects a 'memory' key
        results = []
        for msg in messages[:limit]:
            results.append({
                "memory": msg.content,
                "context": msg.content,
                "metadata": {
                    "peer_id": msg.peer_id,
                    "created_at": str(msg.created_at) if hasattr(msg, 'created_at') else None
                }
            })

        return results

    def reset(self) -> None:
        """Create a new session, effectively resetting memory."""
        new_session_id = f"session_{uuid.uuid4()}"
        self.session = self.honcho.session(new_session_id)
        self.session_id = new_session_id


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
    response_text = str(result)
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
