"""
Honcho Agno Integration

This package provides seamless integration between Honcho and Agno,
enabling AI agents to maintain persistent memory across conversations.

Each HonchoTools instance represents ONE agent identity (peer). The toolkit
speaks as that peer when adding messages or querying the dialectic.

Example:
    ```python
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from honcho_agno import HonchoTools

    # Create Honcho tools
    honcho_tools = HonchoTools(
        app_id="my-app",
        peer_id="assistant",  # The identity for the agent using this toolkit
        session_id="session-123",
    )

    # Create agent with memory
    agent = Agent(
        name="Memory Agent",
        model=OpenAIChat(id="gpt-4o"),
        tools=[honcho_tools],
        description="An assistant with persistent memory powered by Honcho.",
    )

    # Run the agent - messages saved via add_message() are attributed to "assistant"
    response = agent.run("What do you know about the user?")
    ```
"""

from honcho_agno.exceptions import (
    HonchoDependencyError,
    HonchoSessionError,
    HonchoToolError,
)
from honcho_agno.tools import HonchoTools

__version__ = "0.1.0"
__all__ = [
    "HonchoTools",
    "HonchoDependencyError",
    "HonchoSessionError",
    "HonchoToolError",
]
