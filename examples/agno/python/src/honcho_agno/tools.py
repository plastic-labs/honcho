"""
Honcho Tools for Agno

This module provides a Toolkit that allows Agno agents to interact with Honcho's
memory system, including session context, semantic search, and chat.

Designed for Agno's user/assistant architecture:
- user_id from RunContext → Honcho peer (the human user)
- agent_id from init → Honcho peer (the AI assistant)
- session_id from RunContext → Honcho session (shared conversation)

Cross-run memory: Unlike Agno Teams which only share context within a run,
Honcho persists memory across runs. Agent A can remember what Agent B
learned last week.
"""

import logging

from agno.run import RunContext
from agno.tools import Toolkit
from honcho import Honcho

logger = logging.getLogger(__name__)


class HonchoTools(Toolkit):
    """
    Honcho toolkit for Agno agents.

    Maps to Agno's user/assistant model:
    - user_id from RunContext → Honcho peer (the human being queried about)
    - agent_id from init → Honcho peer (the AI assistant's identity)
    - session_id from RunContext → Honcho session (the conversation)

    Tools query Honcho about the USER, not the agent. When the agent asks
    "What does this user prefer?", Honcho returns insights about the human
    user identified by run_context.user_id.

    Example:
        ```python
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from honcho_agno import HonchoTools

        # Initialize toolkit with agent identity
        honcho_tools = HonchoTools(
            workspace_id="my-app",
            agent_id="travel-assistant",
        )

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[honcho_tools],
        )

        # At runtime, pass user_id and session_id
        agent.run("Plan my trip", user_id="user-123", session_id="conv-456")
        ```
    """

    def __init__(
        self,
        workspace_id: str = "default",
        agent_id: str = "assistant",
        honcho_client: Honcho | None = None,
    ) -> None:
        """
        Initialize the Honcho toolkit for a specific agent identity.

        Args:
            workspace_id: Workspace ID for creating an internal Honcho client.
                Ignored if honcho_client is provided.
            agent_id: The agent's identity in Honcho. Used for message attribution
                when the orchestration code saves messages.
            honcho_client: Optional pre-configured Honcho client instance.
                When provided, uses this client directly (workspace_id is ignored).
        """
        super().__init__(name="honcho")

        # Initialize Honcho client
        if honcho_client is not None:
            self.honcho = honcho_client
        else:
            self.honcho = Honcho(workspace_id=workspace_id)

        self.agent_id: str = agent_id

        # Register tools with honcho_ prefix to avoid conflicts with other toolkits
        self.register(self.honcho_get_context)
        self.register(self.honcho_search_messages)
        self.register(self.honcho_chat)

    def honcho_get_context(
        self,
        run_context: RunContext,
        tokens: int | None = None,
        include_summary: bool = True,
    ) -> str:
        """
        Retrieve recent conversation context within token limits.

        Uses run_context.session_id to identify which conversation to retrieve
        context from.

        Args:
            run_context: Agno RunContext providing session_id (auto-injected).
            tokens: Maximum number of tokens to include. If not specified,
                returns all available context.
            include_summary: Whether to include session summary in the context.

        Returns:
            Formatted string containing conversation context.
        """
        try:
            session = self.honcho.session(run_context.session_id)
            result = session.get_context(
                summary=include_summary,
                tokens=tokens,
            )
            return str(result)
        except Exception as e:
            logger.exception("Error retrieving context")
            return f"Error retrieving context: {e!s}"

    def honcho_search_messages(
        self,
        run_context: RunContext,
        query: str,
        limit: int = 10,
    ) -> str:
        """
        Search through session messages using semantic similarity.

        Use this tool to find relevant past messages from the conversation
        history based on semantic meaning rather than exact keyword matching.

        Args:
            run_context: Agno RunContext providing session_id (auto-injected).
            query: Search query for semantic matching.
            limit: Number of results to return (1-100).

        Returns:
            Formatted string with search results.
        """
        try:
            session = self.honcho.session(run_context.session_id)
            messages = session.search(query=query, limit=limit)

            if not messages:
                return f"No messages found matching '{query}'"

            results = [f"=== Search Results for '{query}' ({len(messages)} found) ==="]
            for i, msg in enumerate(messages, 1):
                results.append(f"\n{i}. [{msg.peer_id}] {msg.content}")
                if hasattr(msg, "created_at") and msg.created_at:
                    results.append(f"   Created: {msg.created_at}")

            return "\n".join(results)

        except Exception as e:
            logger.exception("Error searching messages")
            return f"Error searching messages: {e!s}"

    def honcho_chat(self, run_context: RunContext, query: str) -> str:
        """
        Ask Honcho what it knows about the current user.

        Queries the USER's peer (run_context.user_id) to get synthesized insights
        about the human user based on their conversation history. This is how
        the agent learns about user preferences, past discussions, and context.

        Args:
            run_context: Agno RunContext providing user_id and session_id (auto-injected).
            query: Natural language question about the user.
                Examples: "What are the user's preferences?",
                "What topics has the user discussed?",
                "What should I know about this user?"

        Returns:
            Synthesized response about the user based on Honcho's memory.
        """
        try:
            user_id = run_context.user_id
            if not user_id:
                return "Error: No user_id provided in RunContext"

            # Query the USER's peer - this is who we want to learn about
            user_peer = self.honcho.peer(user_id)
            response = user_peer.chat(
                query=query,
                stream=False,
                session=run_context.session_id,
            )

            return str(response) if response else "No relevant information found."

        except Exception as e:
            logger.exception("Error querying user information")
            return f"Error querying user information: {e!s}"
