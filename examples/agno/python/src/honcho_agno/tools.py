"""
Honcho Tools for Agno

This module provides a Toolkit that allows Agno agents to interact with Honcho's
memory system, including session context, semantic search, and dialectic API.

Each HonchoTools instance represents ONE agent identity (peer). The toolkit
speaks as that peer when adding messages or querying the dialectic. For
multi-peer conversations, create separate toolkit instances or use Honcho
directly to manage other peers.
"""

import logging
import uuid
from typing import Any, Literal

from agno.tools import Toolkit
from honcho import Honcho

logger = logging.getLogger(__name__)


class HonchoTools(Toolkit):
    """
    Honcho toolkit for Agno agents.

    Each toolkit instance represents ONE agent identity. The peer_id parameter
    defines who this toolkit "speaks as" - all messages added through this
    toolkit are attributed to that peer.

    For multi-peer conversations:
    - Create one HonchoTools per agent, each with a different peer_id
    - Share the same session_id across toolkits
    - Use Honcho directly for peers not represented by an agent

    Example:
        ```python
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from honcho_agno import HonchoTools

        # This toolkit IS the assistant - it speaks as "assistant"
        honcho_tools = HonchoTools(
            app_id="my-app",
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
        app_id: str = "default",
        peer_id: str = "assistant",
        session_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        environment: Literal["local", "production"] | None = "production",
        honcho_client: Honcho | None = None,
    ) -> None:
        """
        Initialize the Honcho toolkit for a specific agent identity.

        Args:
            app_id: Application/workspace ID for scoping operations.
                Maps to Honcho's workspace_id.
            peer_id: The identity this toolkit represents. All messages
                added through this toolkit are attributed to this peer.
                This is who the agent "is" in the conversation.
            session_id: Optional session ID. If not provided, a new UUID
                will be generated. Share this across toolkits for multi-peer
                conversations.
            api_key: Optional API key for Honcho. If not provided, will
                attempt to read from HONCHO_API_KEY environment variable.
            base_url: Optional base URL for the Honcho API.
            environment: Environment to use. Options: "local", "production".
                Defaults to "production".
            honcho_client: Optional pre-configured Honcho client instance.
                If provided, other connection parameters are ignored.
        """
        super().__init__(name="honcho")

        # Initialize Honcho client
        self.honcho: Honcho
        if honcho_client is not None:
            self.honcho = honcho_client
        else:
            client_kwargs: dict[str, Any] = {"workspace_id": app_id}
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            if environment is not None:
                client_kwargs["environment"] = environment
            self.honcho = Honcho(**client_kwargs)

        # Store identifiers
        self.app_id: str = app_id
        self.peer_id: str = peer_id
        self.session_id: str = session_id or str(uuid.uuid4())

        # Create the peer this toolkit represents
        # This is THE identity of this toolkit - one toolkit = one voice
        self.peer: Peer = self.honcho.peer(peer_id)

        # Create or get session
        self.session: Session = self.honcho.session(self.session_id)

        # Register tools
        self.register(self.add_message)
        self.register(self.get_context)
        self.register(self.search_messages)
        self.register(self.query_peer)

    def add_message(self, content: str) -> str:
        """
        Store a message in the current session as this agent.

        Use this tool to save your responses or important information
        to the conversation history. The message is attributed to this
        toolkit's peer identity.

        Args:
            content: The message content to store.

        Returns:
            Confirmation message indicating the memory was saved.
        """
        try:
            self.session.add_messages([self.peer.message(content)])
            return f"Message saved as '{self.peer_id}' to session {self.session_id}"
        except Exception as e:
            logger.exception("Error saving message")
            return f"Error saving message: {e!s}"

    def get_context(
        self,
        tokens: int | None = None,
        include_summary: bool = True,
    ) -> str:
        """
        Retrieve recent conversation context within token limits.

        Use this tool to get optimized context from the current session,
        including messages and optional summary, that fits within token budgets.

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

            result: list[str] = []

            # Add summary if present
            if context.summary:
                result.append("=== Session Summary ===")
                result.append(context.summary.content)
                result.append("")

            # Add peer representation if present
            if context.peer_representation:
                result.append("=== Peer Representation ===")
                result.append(context.peer_representation)
                result.append("")

            # Add peer card if present
            if context.peer_card:
                result.append("=== Peer Card ===")
                result.extend(context.peer_card)
                result.append("")

            # Add messages
            if context.messages:
                result.append(f"=== Messages ({len(context.messages)}) ===")
                for msg in context.messages:
                    result.append(f"{msg.peer_id}: {msg.content}")

            return "\n".join(result) if result else "No context available"

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

    def query_peer(self, query: str, target_peer_id: str | None = None) -> str:
        """
        Query the system's knowledge about a peer in the conversation.

        Use this tool to ask questions about any participant's preferences,
        interests, or past interactions. The system uses dialectic reasoning
        to provide insights based on the peer's long-term representation.

        Args:
            query: Natural language question about the peer.
                Examples: "What does the user like?", "What are their preferences?"
            target_peer_id: Optional peer ID to query about. If not provided,
                queries about this toolkit's own peer identity.

        Returns:
            Response from the dialectic API with insights about the peer.
        """
        try:
            # Query about a specific peer, or self if not specified
            if target_peer_id:
                target = self.honcho.peer(target_peer_id)
            else:
                target = self.peer

            response = target.chat(
                query=query,
                stream=False,
                session=self.session_id,
            )

            return str(response) if response else "No relevant information found."

        except Exception as e:
            logger.exception("Error querying peer knowledge")
            return f"Error querying peer knowledge: {e!s}"

    def reset_session(self) -> str:
        """
        Create a new session, clearing the conversation history.

        Use this tool to start a fresh conversation while maintaining
        the user's long-term memory and representation.

        Returns:
            Confirmation with the new session ID.
        """
        try:
            self.session_id = str(uuid.uuid4())
            self.session = self.honcho.session(self.session_id)
            return f"Session reset. New session ID: {self.session_id}"
        except Exception as e:
            logger.exception("Error resetting session")
            return f"Error resetting session: {e!s}"
