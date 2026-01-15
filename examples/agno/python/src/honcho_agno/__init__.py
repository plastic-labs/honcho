"""
Honcho Agno Integration

This package provides seamless integration between Honcho and Agno,
enabling AI agents to maintain persistent memory across conversations.

Each HonchoTools instance represents ONE agent identity (peer). The toolkit
provides read access to Honcho for querying conversation context.
Orchestration code handles saving messages to avoid duplicates.

Example:
    ```python
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from honcho import Honcho
    from honcho_agno import HonchoTools

    # Shared Honcho client
    honcho = Honcho(workspace_id="my-app")

    # Create Honcho tools for the assistant
    honcho_tools = HonchoTools(
        peer_id="assistant",
        session_id="session-123",
        honcho_client=honcho,
    )

    # Create user peer for orchestration
    user_peer = honcho.peer("user")

    # Create agent with memory
    agent = Agent(
        name="Memory Agent",
        model=OpenAIChat(id="gpt-4o"),
        tools=[honcho_tools],
        description="An assistant with persistent memory powered by Honcho.",
    )

    # Save user message via orchestration
    honcho_tools.session.add_messages([user_peer.message("I prefer Python over JavaScript")])

    # Run the agent
    response = agent.run("What programming language does the user prefer?")

    # Save assistant response via orchestration
    honcho_tools.session.add_messages([honcho_tools.peer.message(str(response.content))])
    ```
"""

from honcho_agno.tools import HonchoTools

__version__ = "0.1.0"
__all__ = [
    "HonchoTools",
]
