"""
Honcho CrewAI Integration

This package provides seamless integration between Honcho and CrewAI,
enabling AI agents to maintain persistent memory across conversations.

Example:
    ```python
    from honcho_crewai import HonchoStorage
    from crewai.memory.external.external_memory import ExternalMemory
    from crewai import Agent, Task, Crew

    # Initialize storage
    storage = HonchoStorage(user_id="user123")
    external_memory = ExternalMemory(storage=storage)

    # Create agent with memory
    agent = Agent(
        role="AI Assistant",
        goal="Help users with persistent memory",
        backstory="You remember past conversations.",
    )

    # Create crew with external memory
    crew = Crew(
        agents=[agent],
        tasks=[task],
        external_memory=external_memory
    )
    ```
"""

from honcho_crewai.storage import HonchoStorage

__version__ = "0.1.0"
__all__ = ["HonchoStorage"]
