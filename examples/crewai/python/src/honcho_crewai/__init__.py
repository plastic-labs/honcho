"""
Honcho CrewAI Integration

This package provides seamless integration between Honcho and CrewAI,
enabling AI agents to maintain persistent memory across conversations.

Example:
    ```python
    from honcho_crewai import HonchoMemoryStorage, HonchoSearchTool, HonchoGetContextTool, HonchoDialecticTool
    from crewai import Agent, Task, Crew, Memory
    from honcho import Honcho

    # Initialize Honcho client and CrewAI memory
    honcho = Honcho()
    storage = HonchoMemoryStorage(
        peer_id="user123",
        session_id="session123",
        honcho_client=honcho,
    )
    memory = Memory(storage=storage)

    # Create tools for agents
    search_tool = HonchoSearchTool(honcho=honcho, session_id=storage.session_id)
    context_tool = HonchoGetContextTool(honcho=honcho, session_id=storage.session_id, peer_id="user123")
    dialectic_tool = HonchoDialecticTool(honcho=honcho, session_id=storage.session_id, peer_id="user123")

    # Create agent with memory and tools
    agent = Agent(
        role="AI Assistant",
        goal="Help users with persistent memory",
        backstory="You remember past conversations.",
        tools=[search_tool, context_tool, dialectic_tool],
    )

    # Define a task for the crew
    task = Task(
        description="Help the user with their request",
        expected_output="A helpful response",
        agent=agent,
    )

    # Create crew with unified memory
    crew = Crew(
        agents=[agent],
        tasks=[task],
        memory=memory
    )
    ```
"""

from honcho_crewai.exceptions import HonchoDependencyError
from honcho_crewai.storage import HonchoMemoryStorage, HonchoStorage
from honcho_crewai.tools import (
    HonchoDialecticTool,
    HonchoGetContextTool,
    HonchoSearchTool,
)

__version__ = "0.3.0"
__all__ = [
    "HonchoDependencyError",
    "HonchoDialecticTool",
    "HonchoGetContextTool",
    "HonchoMemoryStorage",
    "HonchoSearchTool",
    "HonchoStorage",
]
