"""
Honcho Tools for Agno

This module provides a Toolkit that allows Agno agents to interact with Honcho's
memory system, including session context, semantic search, and chat.

Each HonchoTools instance represents ONE agent identity (peer). The toolkit
provides read access to Honcho for querying conversation context.
Orchestration code will handle saving messages to avoid duplicates.
"""

import logging
import uuid
from typing import TYPE_CHECKING

from agno.tools import Toolkit
from honcho import Honcho

if TYPE_CHECKING:
    from honcho.peer import Peer
    from honcho.session import Session

logger = logging.getLogger(__name__)


class HonchoTools(Toolkit):
    """
    Honcho toolkit for Agno agents.

    Each toolkit instance is for ONE agent identity.

    For multi-peer conversations:
    - Create one HonchoTools per agent, each with a different peer_id
    - Share the same session_id across toolkits
    - Messages are saved to Honcho by the orchestration code, not the toolkit

    Example:
        ```python
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from honcho_agno import HonchoTools

        honcho_tools = HonchoTools(
            workspace_id="my-app",
            peer_id="assistant",
            session_id="shared-session",
        )

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[honcho_tools],
        )
        ```
    """

    def __init__(
        self,
        workspace_id: str = "default",
        peer_id: str = "assistant",
        session_id: str | None = None,
        honcho_client: Honcho | None = None,
    ) -> None:
        """
        Initialize the Honcho toolkit for a specific agent identity.

        Args:
            workspace_id: Workspace ID for creating an internal Honcho client.
                Ignored if honcho_client is provided.
            peer_id: The identity this toolkit represents. This is who
                the agent "is" when querying peer knowledge.
            session_id: Optional session ID. If not provided, a new UUID
                will be generated. Share this across toolkits for multi-peer
                conversations.
            honcho_client: Optional pre-configured Honcho client instance.
                When provided, uses this client directly (workspace_id is ignored).
        """
        super().__init__(name="honcho")

        # Initialize Honcho client
        if honcho_client is not None:
            self.honcho = honcho_client
        else:
            self.honcho = Honcho(workspace_id=workspace_id)

        self.peer_id: str = peer_id
        self.session_id: str = session_id or str(uuid.uuid4())

        # Create the peer this toolkit represents
        self.peer: Peer = self.honcho.peer(peer_id)

        # Create or get session
        self.session: Session = self.honcho.session(self.session_id)

        # Register tools
        self.register(self.get_context)
        self.register(self.search_messages)
        self.register(self.chat)

    def get_context(
        self,
        tokens: int | None = None,
        include_summary: bool = True,
    ) -> str:
        """
        Retrieve recent conversation context within token limits.

        Args:
            tokens: Maximum number of tokens to include. If not specified,
                returns all available context.
            include_summary: Whether to include session summary in the context.

        Returns:
            Formatted string containing conversation context.
        """
        try:
            context = self.session.get_context(
                summary=include_summary,
                tokens=tokens,
            )
            return str(context)
        except Exception as e:
            logger.exception("Error retrieving context")
            return f"Error retrieving context: {e!s}"

    def search_messages(
        self,
        query: str,
        limit: int = 10,
    ) -> str:
        """
        Search through session messages using semantic similarity.

        Use this tool to find relevant past messages from the conversation
        history based on semantic meaning rather than exact keyword matching.

        Args:
            query: Search query for semantic matching.
            limit: Number of results to return (1-100).

        Returns:
            Formatted string with search results.
        """
        try:
            messages = self.session.search(query=query, limit=limit)

            if not messages:
                return f"No messages found matching '{query}'"

            result = [f"=== Search Results for '{query}' ({len(messages)} found) ==="]
            for i, msg in enumerate(messages, 1):
                result.append(f"\n{i}. [{msg.peer_id}] {msg.content}")
                if hasattr(msg, "created_at") and msg.created_at:
                    result.append(f"   Created: {msg.created_at}")

            return "\n".join(result)

        except Exception as e:
            logger.exception("Error searching messages")
            return f"Error searching messages: {e!s}"

    def chat(self, query: str) -> str:
        """
        Ask a question about what was discussed in this conversation.

        Use this tool to query session-specific context and facts.
        The system uses Honcho reasoning to provide synthesized
        insights based on the conversation history.

        Args:
            query: Natural language question about the conversation.
                Examples: "What did we discuss?", "What preferences should I be aware of?",
                "What topics came up?"

        Returns:
            Synthesized response based on the session context.
        """
        try:
            response = self.peer.chat(
                query=query,
                stream=False,
                session=self.session_id,
            )

            return str(response) if response else "No relevant information found."

        except Exception as e:
            logger.exception("Error querying conversation")
            return f"Error querying conversation: {e!s}"
