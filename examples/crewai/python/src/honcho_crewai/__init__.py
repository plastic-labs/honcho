"""
Honcho CrewAI Integration

This package provides seamless integration between Honcho and CrewAI,
enabling AI agents to maintain persistent memory across conversations.

Example:
    ```python
    from honcho_crewai import HonchoStorage
    from crewai.memory.external.external_memory import ExternalMemory
    from crewai import Agent, Task, Crew
    from honcho import Honcho

    # Initialize Honcho client and storage
    honcho = Honcho()
    storage = HonchoStorage(user_id="user123", honcho_client=honcho)
    external_memory = ExternalMemory(storage=storage)

    # Create tools for agents
    search_tool = create_search_tool(honcho, session_id=storage.session_id)
    context_tool = create_get_context_tool(honcho, session_id=storage.session_id, peer_id="user123")
    dialectic_tool = create_dialectic_tool(honcho, session_id=storage.session_id, peer_id="user123")

    # Create agent with memory and tools
    agent = Agent(
        role="AI Assistant",
        goal="Help users with persistent memory",
        backstory="You remember past conversations.",
        tools=[search_tool, context_tool, dialectic_tool],
    )

    # Create crew with external memory
    crew = Crew(
        agents=[agent],
        tasks=[task],
        external_memory=external_memory
    )
    ```
"""

from honcho_crewai.exceptions import HonchoDependencyError
from honcho_crewai.storage import HonchoStorage
from honcho_crewai.tools import (
    HonchoDialecticTool,
    HonchoGetContextTool,
    HonchoSearchTool,
    create_dialectic_tool,
    create_get_context_tool,
    create_search_tool,
)

__version__ = "0.1.0"
__all__ = [
    "HonchoStorage",
    "HonchoGetContextTool",
    "HonchoDialecticTool",
    "HonchoSearchTool",
    "create_get_context_tool",
    "create_dialectic_tool",
    "create_search_tool",
    "HonchoDependencyError",
]
