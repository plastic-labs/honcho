"""
Honcho Agno Integration

This package provides seamless integration between Honcho and Agno,
enabling AI agents to maintain persistent memory across conversations.

The toolkit provides read access to Honcho for querying conversation context.
Orchestration code handles saving messages using the Honcho client directly.

Example:
    ```python
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from honcho import Honcho
    from honcho_agno import HonchoTools

    # Initialize Honcho client
    honcho = Honcho(workspace_id="my-app")

    # Create Honcho tools for the agent
    honcho_tools = HonchoTools(honcho_client=honcho)

    # Create peers and session for orchestration
    user_peer = honcho.peer("user-123")
    assistant_peer = honcho.peer("assistant")
    session = honcho.session("session-123")

    # Create agent with memory tools
    agent = Agent(
        name="Memory Agent",
        model=OpenAIChat(id="gpt-4o"),
        tools=[honcho_tools],
    )

    # Save user message via orchestration (using honcho client directly)
    session.add_messages([user_peer.message("I prefer Python over JavaScript")])

    # Run agent - user_id and session_id flow through RunContext to tools
    response = agent.run(
        "What programming language does the user prefer?",
        user_id="user-123",
        session_id="session-123",
    )

    # Save assistant response via orchestration
    session.add_messages([assistant_peer.message(str(response.content))])
    ```
"""

from honcho_agno.tools import HonchoTools

__version__ = "0.1.0"
__all__ = [
    "HonchoTools",
]
