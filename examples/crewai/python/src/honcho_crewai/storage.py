"""
Honcho Storage for CrewAI External Memory

This module provides a Honcho-backed storage provider for CrewAI's external memory
system, enabling AI agents to maintain persistent conversation memory across sessions.
"""

import uuid
from typing import Any, Dict, List, Optional

from crewai.memory.storage.interface import Storage
from honcho import Honcho


class HonchoStorage(Storage):
    """
    Honcho-backed storage provider for CrewAI external memory.

    Implements CrewAI's Storage interface using Honcho's session-based memory,
    allowing agents to maintain context across conversations.

    Attributes:
        honcho: The Honcho client instance
        user: Peer representing the user
        assistant: Peer representing the AI assistant
        session: The conversation session
        session_id: Unique identifier for the session

    Example:
        ```python
        from honcho_crewai import HonchoStorage
        from crewai.memory.external.external_memory import ExternalMemory

        # Initialize storage
        storage = HonchoStorage(user_id="user123")

        # Use with CrewAI's external memory
        external_memory = ExternalMemory(storage=storage)
        ```
    """

    def __init__(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        honcho_client: Optional[Honcho] = None,
    ):
        """
        Initialize Honcho storage for a specific user and session.

        Args:
            user_id: Unique identifier for the user
            session_id: Optional session ID. If not provided, one will be generated
            honcho_client: Optional Honcho client instance. If not provided, creates a new one
        """
        self.honcho = honcho_client or Honcho()

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

        This method is called by CrewAI to store messages and context. Messages
        are associated with the appropriate peer (user or assistant) based on
        the metadata.

        Args:
            value: Message content to save
            metadata: Metadata dict that may contain 'role', 'agent', or 'type' info
                     Common keys: 'role', 'agent', 'type'
        """
        # Determine if this is from user or assistant based on metadata
        # Check various metadata keys that might indicate the role
        role = metadata.get("role", metadata.get("agent", "assistant"))
        is_user = role == "user"
        peer = self.user if is_user else self.assistant

        # Add message to session
        self.session.add_messages([peer.message(str(value))])

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant messages in Honcho session.

        This method is called by CrewAI to retrieve relevant conversation context.
        It uses Honcho's get_context() method to fetch messages within the
        specified token limit.

        Args:
            query: Search query (currently not used for filtering, but required by interface)
            limit: Maximum number of messages to retrieve
            score_threshold: Minimum relevance score (currently not used, but required by interface)

        Returns:
            List of message dictionaries in CrewAI expected format.
            Each dict contains:
                - memory: The message content (required by CrewAI)
                - context: Same as memory (for compatibility)
                - metadata: Additional message metadata (peer_id, created_at)
        """
        # Token limit approximation: ~100 tokens per message
        # This is a rough estimate to control context size
        token_limit = limit * 100

        # Get context from Honcho
        # get_context() retrieves relevant conversation history
        context = self.session.get_context(tokens=token_limit)
        messages = context.messages

        # Convert to CrewAI expected format
        results = []
        for msg in messages[:limit]:
            results.append(
                {
                    "content": msg.content,
                    "memory": msg.content,
                    "context": msg.content,
                    "metadata": {
                        "peer_id": msg.peer_id,
                        "created_at": str(msg.created_at) if hasattr(msg, "created_at") else None,
                    },
                }
            )

        return results

    def reset(self) -> None:
        """
        Create a new session, effectively resetting memory.

        This creates a new Honcho session with a fresh UUID, allowing the agent
        to start a new conversation without the previous context.
        """
        new_session_id = f"session_{uuid.uuid4()}"
        self.session = self.honcho.session(new_session_id)
        self.session_id = new_session_id
